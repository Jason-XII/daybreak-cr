#!/usr/bin/env python3
"""Export raw traces with independent cache reuse, seed search, and prior-failure policies."""
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert as C
from cr import observation as O
from cr.raw_trace import SCHEMA, atomic_json, check_trace, digest_file, read_trace, trace_path, write_trace


def terminal_path(out, tag):
    return trace_path(out / 'outcomes', tag).with_suffix('.json')


def payload_at(job, streams, payloads=None):
    name = job['part']
    if name not in streams:
        if len(streams) >= 8:
            streams.pop(next(iter(streams))).close()
        streams[name] = ((payloads or ROOT / 'hf-ingest/payloads') / name).open('rb')
    source = streams[name]
    source.seek(job['offset'])
    encoded = source.readline()
    item = json.loads(encoded)
    if item['tag'] != job['tag'] or item['payload']['source']['replay_tag'] != job['tag']:
        raise ValueError('canonical assignment/payload mismatch')
    return item['payload'], hashlib.sha256(encoded).hexdigest()


def cached_options(meta):
    if meta.get('kept') is not True or 'cut_at' in meta:
        raise ValueError('previous result is rejected or partial')
    if meta.get('hp_error') != 0 or meta.get('winner') is not True or meta.get('crowns') is not True:
        raise ValueError('previous exact guard is absent')
    search = meta.get('search') or {}
    seed = search.get('seed')
    if seed is not None and (type(seed) is not int or seed <= 0 or seed % 7919):
        raise ValueError('invalid cached search seed')
    if seed is None and search.get('seed_dependent'):
        raise ValueError('seed-dependent result has no cached success seed')
    return {'seed': seed, 'kings': {int(k): v for k, v in (search.get('king') or {}).items()} or None,
            'press_choices': search.get('presses'), 'calibration_seeds': 0}


def replay_policy(args):
    return {'reuse_cache': args.reuse_cache, 'allow_seed_search': args.allow_seed_search,
            'skip_failed': args.skip_failed, 'seeds': args.seeds, 'calibration_seeds': args.calibration_seeds}


def full_exact_metadata(meta):
    return (meta.get('kept') is True and 'cut_at' not in meta and meta.get('hp_error') == 0 and
            meta.get('winner') is True and meta.get('crowns') is True)


CALIBRATION_MISSES = ('no compatible native omitted-card layout',
                      'native deal calibration failed after ten configurations',
                      'native deal calibration repeated a deck configuration')


def replay_with_policy(recorder, payload, options, args):
    """Always try the selected recipe without seed enumeration; search only after an explicitly allowed miss."""
    try:
        result = C.play_replay(recorder, payload, **options)
    except ValueError as error:
        if str(error) not in CALIBRATION_MISSES:
            raise
        if not args.allow_seed_search:
            raise
        result = None
    if args.allow_seed_search and (result is None or not C.exact(result)):
        result = C.play_exact(recorder, payload, seeds=args.seeds, calibration_seeds=args.calibration_seeds)
        return result, True
    return result, False


def packed_features(game):
    vocab, legal = O.Vocab.load(), O.cached_legal()
    out = {}
    for owner in (0, 1):
        x, y = C.game_features(game, owner, vocab, legal)
        out[owner] = C._pack(x, C.targets(x, y))
    return out


def asset_tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('asset file resolves outside its declared root')
            digest.update(str(path.relative_to(root)).encode() + b'\0' + digest_file(path).encode() + b'\n')
    return digest.hexdigest()


def compare_features(left, right):
    import numpy as np
    errors = []
    for owner in (0, 1):
        if set(left[owner]) != set(right[owner]):
            errors.append(f'p{owner}: array keys differ')
        for key in set(left[owner]) & set(right[owner]):
            a, b = left[owner][key], right[owner][key]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                errors.append(f'p{owner}:{key}')
    return errors


def export_one(job, env, args, fingerprints, streams, search_attempts=None):
    archive = args.source / job['file']
    if digest_file(archive) != job['source_archive_sha256']:
        raise ValueError('source archive hash differs from verified exact manifest')
    meta = C.read_metadata(archive)
    if meta['tag'] != job['tag'] or meta['engine'] not in (job['engine'], job.get('alt')):
        raise ValueError('source archive identity/version mismatch')
    payload, payload_sha = payload_at(job, streams, getattr(args, 'payloads', None))
    previous_exact = full_exact_metadata(meta)
    if not previous_exact and args.skip_failed:
        return {'status': 'skipped', 'reason': 'previous conversion failed; retry disabled',
                'skip_code': 'previous_failure_skipped'}
    options = cached_options(meta) if args.reuse_cache and previous_exact else {
        'seed': None, 'kings': None, 'press_choices': None, 'calibration_seeds': 0}
    cached_search = meta.get('search', {}) if args.reuse_cache else {'runs': 1}
    recipe_cache_hit = False
    cache_skip = None
    if args.recipe_cache is not None:
        outcome = terminal_path(args.recipe_cache, job['tag'])
        if outcome.exists():
            prior = json.loads(outcome.read_text())
            if prior.get('source_archive_sha256') != job['source_archive_sha256']:
                raise ValueError('cached outcome source binding mismatch')
            if prior.get('status') in ('skipped', 'failed') and args.skip_failed:
                if (prior.get('payload_row_sha256') != payload_sha or
                        any(prior.get('provenance', {}).get(key) != fingerprints.get(key) for key in
                            ('engine_sha256', 'data_fingerprint_sha256', 'data_tree_sha256',
                             'bindings_sha256', 'converter_sha256', 'reader_sha256'))):
                    raise ValueError('skipped recipe cache is not compatible with the current execution')
                cache_skip = {'status': 'skipped', 'reason': prior['reason'], 'skip_cache_hit': True,
                              'skip_code': 'previous_failure_skipped',
                              'previous_status': prior['status']}
        cache = trace_path(args.recipe_cache / 'traces', job['tag'])
        if args.reuse_cache and cache_skip is None and cache.exists():
            cached = read_trace(cache)
            check_trace(cached)
            cached_meta = cached['metadata']
            if (cached_meta['source_archive_sha256'] != job['source_archive_sha256'] or
                    cached_meta['payload_row_sha256'] != payload_sha or
                    any(cached['provenance'][key] != fingerprints[key] for key in
                        ('engine_sha256', 'data_fingerprint_sha256', 'data_tree_sha256', 'bindings_sha256'))):
                raise ValueError('cached recipe provenance mismatch')
            options['prepared_game'] = cached['prepared_game']
            options['seed'] = None
            cached_search = cached_meta.get('search') or {}
            options['press_choices'] = cached_search.get('presses')
            recipe_cache_hit = True
    if cache_skip is not None:
        return cache_skip
    recorder = C.Recorder(env)
    if search_attempts is not None:
        recorder.search_attempts = search_attempts
    try:
        result, search_used = replay_with_policy(recorder, payload, options, args)
    except ValueError as error:
        if str(error) != 'no compatible native omitted-card layout':
            raise
        return {'status': 'skipped', 'reason': str(error), 'skip_code': 'missing_implicit_calibration_seed'}
    if result['failures']:
        raise ValueError('native command execution failures: ' + str(result['failures'][:3]))
    if not C.exact(result):
        return {'status': 'skipped', 'reason': 'reconstruction is not exact under the selected search policy',
                'skip_code': 'cached_recipe_not_exact', 'fidelity': result['fidelity'],
                'seed_search_attempts': recorder.search_attempts['seed'],
                'calibration_seed_search_attempts': recorder.search_attempts['calibration_seed']}
    dropped = [h for h in result['holds'] if h[4] == 'dropped']
    if dropped or not recorder.decisions:
        return {'status': 'skipped', 'reason': 'reconstruction contains dropped plays or no decisions',
                'skip_code': 'dropped_or_empty_reconstruction'}
    if previous_exact and args.reuse_cache and not args.allow_seed_search and len(recorder.decisions) != meta['decisions']:
        raise ValueError('raw decision coverage differs from the previous full-exact archive')
    played = result['game']
    resolved_search = result.get('search', cached_search)
    prepared = copy.deepcopy(played)
    holds = {(h[0], h[1], h[2]): h for h in result['holds']}
    for command in played['commands']:
        if command['kind'] == 'card':
            h = holds.get((command['tick'], command['owner'], command['card_id']))
            command['executed_tick'] = command['tick'] + h[3] if h and h[4] == 'played' else None
    metadata = {**meta, 'part': job['part'], 'row': job['row'], 'actual_seed': played['match']['rndSeed'],
                'source_archive_sha256': job['source_archive_sha256'], 'payload_row_sha256': payload_sha,
                'sampling_ticks': 5, 'seed_search_attempts': recorder.search_attempts['seed'],
                'calibration_seed_search_attempts': recorder.search_attempts['calibration_seed'],
                'decisions': len(recorder.decisions), 'kept': True, 'hp_error': 0, 'winner': True, 'crowns': True,
                'search': resolved_search, 'previous_full_exact': previous_exact, 'replay_policy': replay_policy(args)}
    if 'expert' not in metadata:
        team_owner = played['episode']['source_team_owner']
        metadata['expert'] = {('team' if entry['owner'] == team_owner else 'opponent'): entry.get('expert')
                              for entry in meta.get('entries', [])}
    metadata.pop('entries', None)
    metadata.pop('why', None)
    document = {'schema': SCHEMA, 'metadata': metadata, 'provenance': fingerprints,
                'prepared_game': C._jsonable(prepared), 'fidelity': C._jsonable(result['fidelity']),
                'holds': C._jsonable(result['holds']), 'presses': C._jsonable(result['presses']),
                'failures': C._jsonable(result['failures']),
                'field_layout': {k: hex(v) for k, v in env.reader.f.items()},
                'game': {'header': {'tag': job['tag'], 'commands': C._jsonable(played['commands']),
                                    'terminal': C._jsonable(played['terminal']), 'episode': C._jsonable(played['episode'])},
                         'decisions': C._jsonable(recorder.decisions)}}
    check_trace(document)
    path = trace_path(args.out / 'traces', job['tag'])
    sizes = write_trace(path, document)
    checks = {}
    if args.feature_checks:
        restored = read_trace(path)
        if document != restored:
            raise ValueError('raw serialization is not lossless')
        fresh = packed_features(document['game'])
        errors = compare_features(fresh, packed_features(restored['game']))
        if errors:
            raise ValueError('raw feature roundtrip differs: ' + str(errors))
        checks['raw_feature_roundtrip'] = True
        import numpy as np
        if previous_exact:
            with np.load(archive, allow_pickle=False) as previous:
                old = {owner: {k[len(f'p{owner}_'):]: previous[k] for k in previous.files
                               if k.startswith(f'p{owner}_')} for owner in (0, 1)}
            checks['previous_feature_differences'] = compare_features(fresh, old)
            if checks['previous_feature_differences']:
                raise ValueError('previous feature arrays differ: ' + str(checks['previous_feature_differences']))
        # A recipe must also recreate the trajectory in a fresh native match without calibration or search.
        recipe_recorder = C.Recorder(env)
        recipe_result = C.play_replay(recipe_recorder, payload, prepared_game=prepared,
                                      press_choices=resolved_search.get('presses', options['press_choices']),
                                      calibration_seeds=0)
        if not C.exact(recipe_result) or recipe_recorder.decisions != recorder.decisions:
            raise ValueError('saved reconstruction recipe did not reproduce every raw decision')
        checks['prepared_recipe_roundtrip'] = True
    return {'status': 'accepted', 'file': str(path.relative_to(args.out)), **sizes,
            'actual_seed': metadata['actual_seed'],
            'cached_search_seed': meta.get('search', {}).get('seed') if args.reuse_cache and not search_used else None,
            'recipe_cache_hit': recipe_cache_hit, 'search_fallback_used': search_used,
            'seed_search_attempts': metadata['seed_search_attempts'],
            'calibration_seed_search_attempts': metadata['calibration_seed_search_attempts'],
            'validation': C.held_out(job['tag']), 'checks': checks}


def worker(index, engine, jobs, args, startup, fingerprints):
    log = (args.out / f'worker-{index}.log').open('a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.path.insert(0, str(args.clapha_engine / 'crx'))
    libdir, assets = args.engines[engine]
    os.environ['CRX_ASSETS'] = str(assets)
    from env import CrxEnv
    import loader
    if loader.ASSET_ROOT.resolve() != assets:
        raise ValueError('effective native asset root mismatch')
    env, streams = None, {}
    current = args.out / f'current-{index}.json'
    progress = {'worker': index, 'engine': engine, 'pid': os.getpid(), 'finished': 0,
                'accepted': 0, 'skipped': 0, 'failed': 0, 'raw_bytes': 0, 'decisions': 0}
    worker_started = time.time()
    while True:
        try:
            job = jobs.get(timeout=1)
        except queue.Empty:
            break
        atomic_json(current, {'job': job, 'pid': os.getpid()})
        began = time.perf_counter()
        search_attempts = {'seed': 0, 'calibration_seed': 0}
        try:
            if env is None:
                with startup:
                    env = CrxEnv(libdir,
                                 args.out / 'native-work' / f'{index}-{engine}',
                                 assets)
                    env.reader.capture_raw = True
            result = export_one(job, env, args, fingerprints[engine], streams, search_attempts=search_attempts)
        except Exception as error:
            result = {'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}
        if result['status'] == 'skipped':
            _, payload_sha = payload_at(job, streams, args.payloads)
            result.update(payload_row_sha256=payload_sha, provenance=fingerprints[engine])
        result.update(tag=job['tag'], engine=job['actual_engine'], seconds=time.perf_counter() - began,
                      source_archive_sha256=job['source_archive_sha256'],
                      seed_search_attempts=search_attempts['seed'],
                      calibration_seed_search_attempts=search_attempts['calibration_seed'])
        atomic_json(terminal_path(args.out, job['tag']), result)
        current.unlink(missing_ok=True)
        progress['finished'] += 1
        progress[result['status']] += 1
        progress['raw_bytes'] += result.get('raw_bytes', 0)
        progress['decisions'] += result.get('decisions', 0)
        if progress['finished'] == 1 or progress['finished'] % 32 == 0:
            atomic_json(args.out / f'progress-{index}.json',
                        {**progress, 'elapsed_seconds': time.time() - worker_started, 'updated_unix': time.time()})
        print(json.dumps(result), flush=True)
    atomic_json(args.out / f'progress-{index}.json',
                {**progress, 'status': 'worker_complete', 'elapsed_seconds': time.time() - worker_started,
                 'updated_unix': time.time()})
    for source in streams.values():
        source.close()
    sys.stdout.flush()
    os._exit(0)


def make_jobs(args):
    assignment = [json.loads(line) for line in args.assignment.open()]
    by_tag = {row['tag']: row for row in assignment}
    if len(by_tag) != len(assignment):
        raise ValueError('duplicate assignment tags')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('selection') != 'full_exact_all_formats':
        raise ValueError('requires independently verified full-exact-all-formats manifest')
    selected, seen = [], set()
    owners = {}
    for entry in manifest['entries']:
        owners.setdefault(entry['tag'], []).append(entry['owner'])
        if entry['owner'] != 0:
            continue
        tag = entry['tag']
        if tag in seen or tag not in by_tag:
            raise ValueError('invalid accepted tag identity')
        seen.add(tag)
        selected.append({**by_tag[tag], 'file': entry['file'], 'source_archive_sha256': entry['archive_sha256'],
                         'previous_full_exact': True})
    if len(manifest['entries']) != 2 * len(selected) or any(sorted(o) != [0, 1] for o in owners.values()):
        raise ValueError('invalid dual-owner full exact manifest')
    if not args.skip_failed:
        previous_manifest = args.source / 'manifest.json'
        if digest_file(previous_manifest) != manifest.get('source_manifest_sha256'):
            raise ValueError('retry source manifest is not bound to the full-exact audit')
        previous = json.loads(previous_manifest.read_text())
        for rejected in previous['rejected']:
            tag = rejected['tag']
            if tag in seen or tag not in by_tag:
                raise ValueError('invalid or duplicate previous rejected tag')
            seen.add(tag)
            archive = C.archive_path(args.source, tag)
            metadata = C.read_metadata(archive)
            if metadata.get('tag') != tag or metadata.get('kept') is not False:
                raise ValueError('rejected manifest/archive mismatch')
            selected.append({**by_tag[tag], 'file': archive.name,
                             'source_archive_sha256': digest_file(archive), 'previous_full_exact': False})
    if args.limit:
        rng = random.Random(20261105)
        selected = rng.sample(selected, min(args.limit, len(selected)))
    # Index canonical payload rows once per shard; workers seek directly rather than decoding entire parts.
    requested = {}
    for job in selected:
        requested.setdefault(job['part'], {})[job['row']] = job
    for part, wanted in requested.items():
        with (getattr(args, 'payloads', ROOT / 'hf-ingest/payloads') / part).open('rb') as source:
            row = 0
            while wanted:
                offset = source.tell()
                line = source.readline()
                if not line:
                    raise ValueError('payload row is missing')
                if row in wanted:
                    wanted.pop(row)['offset'] = offset
                row += 1
    return assignment, selected


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='data root for default payload and engine paths')
    parser.add_argument('--payloads', type=Path, help='canonical JSONL payload directory')
    parser.add_argument('--clapha-engine', type=Path, help='clapha-engine checkout with raw_fields support')
    parser.add_argument('--engine', nargs=3, action='append', metavar=('NAME', 'LIBDIR', 'ASSETS'),
                        help='repeat for each data version, as in convert.py')
    parser.add_argument('--assignment', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--startup-workers', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--feature-checks', action='store_true')
    parser.add_argument('--recipe-cache', type=Path,
                        help='previous raw export holding complete successful reconstruction recipes')
    parser.add_argument('--reuse-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='reuse successful seed/King/press and full recipes; independent of retry/search policy')
    parser.add_argument('--allow-seed-search', action=argparse.BooleanOptionalAction, default=False,
                        help='allow dice and omitted-card calibration seed searches after the initial replay misses')
    parser.add_argument('--skip-failed', action=argparse.BooleanOptionalAction, default=True,
                        help='skip previous rejected/skipped/failed results; disabling explicitly permits retry')
    parser.add_argument('--seeds', type=int, default=16,
                        help='dice seed candidates per search, used only with --allow-seed-search')
    parser.add_argument('--calibration-seeds', type=int, default=64,
                        help='omitted-card seed candidates per calibration, used only with --allow-seed-search')
    return parser


def main():
    global ROOT
    parser = argument_parser()
    args = parser.parse_args()
    ROOT = args.root.expanduser().resolve()
    args.payloads = (args.payloads or ROOT / 'hf-ingest/payloads').expanduser().resolve()
    args.clapha_engine = (args.clapha_engine or ROOT / 'clapha-engine').expanduser().resolve()
    configured = args.engine or [(name, ROOT / 'conversion-kit/ENGINES/160402020', ROOT / 'conversion-kit/DATA' / name)
                                 for name in ('d19', 'd22')]
    args.engines = {name: (Path(lib).expanduser().resolve(), Path(assets).expanduser().resolve())
                    for name, lib, assets in configured}
    if len(args.engines) != len(configured):
        parser.error('engine names must be unique')
    if args.seeds < 0 or args.calibration_seeds < 0 or args.workers < 1 or args.startup_workers < 1 or args.limit < 0:
        parser.error('search limits and --limit must be nonnegative; worker counts must be positive')
    for name in ('assignment', 'source', 'manifest', 'out', 'recipe_cache'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.out == args.source or args.out == args.recipe_cache:
        parser.error('raw output must be separate from its source and recipe cache')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assignment, selected = make_jobs(args)
    old_full_exact_count = len(json.loads(args.manifest.read_text())['entries']) // 2
    fingerprints = {}
    for engine, (libdir, assets) in args.engines.items():
        fingerprints[engine] = {'engine_sha256': digest_file(libdir / 'libg.so'),
                                'data_fingerprint_sha256': digest_file(assets / 'fingerprint.json'),
                                'data_tree_sha256': asset_tree_digest(assets),
                                'converter_sha256': digest_file(Path(C.__file__)),
                                'reader_sha256': digest_file(args.clapha_engine / 'crx/observe.py'),
                                'exporter_sha256': digest_file(Path(__file__)),
                                'raw_schema_sha256': digest_file(Path(C.__file__).parent / 'cr/raw_trace.py'),
                                'bindings_sha256': asset_tree_digest(args.clapha_engine / 'crx/bindings'),
                                'effective_asset_root': str(assets.resolve()),
                                'payload_root': str(args.payloads),
                                'assignment_sha256': digest_file(args.assignment),
                                'source_manifest_sha256': digest_file(args.manifest),
                                'replay_policy': replay_policy(args),
                                'recipe_cache': str(args.recipe_cache.resolve()) if args.recipe_cache is not None else None,
                                'retry_source_manifest_sha256': digest_file(args.source / 'manifest.json')
                                    if not args.skip_failed else None}
    prior_run = args.out / 'run.json'
    if prior_run.exists() and json.loads(prior_run.read_text()).get('provenance') != fingerprints:
        raise ValueError('resume provenance differs; use a new output version')
    for progress in args.out.glob('progress-*.json'):
        progress.unlink()
    missing = []
    for job in selected:
        terminal = terminal_path(args.out, job['tag'])
        if terminal.exists():
            result = json.loads(terminal.read_text())
            if result.get('source_archive_sha256') != job['source_archive_sha256']:
                raise ValueError('resume outcome bound to a different source archive')
            if result.get('status') == 'accepted':
                path = trace_path(args.out / 'traces', job['tag'])
                if not path.is_file() or path.stat().st_size != result.get('raw_bytes'):
                    raise ValueError('resume raw artifact is absent or incomplete')
                continue
            if args.skip_failed:
                continue
            backup = terminal.with_name(terminal.name + f'.before-retry-{time.time_ns()}')
            backup.write_bytes(terminal.read_bytes())
        meta = C.read_metadata(args.source / job['file'])
        job['actual_engine'] = meta['engine']
        if job['actual_engine'] not in fingerprints:
            raise ValueError('unsupported cached engine version')
        missing.append(job)
    started = time.time()
    atomic_json(args.out / 'run.json', {'status': 'running', 'pid': os.getpid(), 'started_unix': started,
                                      'selected': len(selected), 'remaining': len(missing),
                                      'old_rejected_or_partial_skipped': len(assignment) - len(selected)
                                          if args.limit == 0 else None,
                                      'previous_full_exact_games': old_full_exact_count,
                                      'workers': args.workers,
                                      'replay_policy': replay_policy(args),
                                      'seed_search_limit': args.seeds if args.allow_seed_search else 0,
                                      'calibration_seed_search_limit': args.calibration_seeds if args.allow_seed_search else 0,
                                      'provenance': fingerprints})
    random.Random(20261105).shuffle(missing)
    context = mp.get_context('fork')
    startup = context.BoundedSemaphore(args.startup_workers)
    processes = []
    # Native libraries install process-global hooks: each worker must serve exactly one data version.
    for engine in args.engines:
        group = [job for job in missing if job['actual_engine'] == engine]
        if not group:
            continue
        jobs = context.Queue()
        for job in group:
            jobs.put(job)
        count = min(len(group), max(1, round(args.workers * len(group) / max(1, len(missing)))))
        for _ in range(count):
            processes.append(context.Process(target=worker,
                args=(len(processes), engine, jobs, args, startup, fingerprints)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    # A native crash leaves its claimed job visible. Do not repeatedly rerun the same crashing game.
    for current in args.out.glob('current-*.json'):
        job = json.loads(current.read_text())['job']
        terminal = terminal_path(args.out, job['tag'])
        if not terminal.exists():
            atomic_json(terminal, {'tag': job['tag'], 'status': 'failed', 'reason': 'native worker crash; not retried',
                                   'source_archive_sha256': job['source_archive_sha256']})
    results = []
    for job in selected:
        terminal = terminal_path(args.out, job['tag'])
        if not terminal.exists():
            raise RuntimeError('unfinished assignment; restart to resume missing outcomes')
        results.append(json.loads(terminal.read_text()))
    counts = Counter(row['status'] for row in results)
    manifest = {'schema': SCHEMA, 'selection': 'raw_full_exact', 'source_manifest_sha256': digest_file(args.manifest),
                'assignment_sha256': digest_file(args.assignment), 'provenance': fingerprints,
                'entries': [row for row in results if row['status'] == 'accepted'],
                'skipped': [row for row in results if row['status'] != 'accepted']}
    atomic_json(args.out / 'manifest.json', manifest)
    report = {'status': 'failed' if counts.get('failed') else 'export_complete_pending_verification',
              'selected': len(selected), 'counts': dict(counts),
              'old_rejected_or_partial_skipped': len(assignment) - len(selected) if args.limit == 0 else None,
              'wall_seconds': time.time() - started,
              'raw_bytes': sum(row.get('raw_bytes', 0) for row in results),
              'decisions': sum(row.get('decisions', 0) for row in results),
              'replay_policy': replay_policy(args),
              'seed_search_attempts': sum(row.get('seed_search_attempts', 0) for row in results),
              'calibration_seed_search_attempts': sum(row.get('calibration_seed_search_attempts', 0) for row in results),
              'manifest_sha256': digest_file(args.out / 'manifest.json'),
              'skip_reasons': dict(Counter(row.get('reason') for row in results if row['status'] != 'accepted'))}
    atomic_json(args.out / 'export.done.json', report)
    print(json.dumps(report), flush=True)
    raise SystemExit(1 if counts.get('failed') else 0)


if __name__ == '__main__':
    main()
