#!/usr/bin/env python3
"""Independently verify raw coverage, provenance, exact guards, hashes, and decoded fields."""
import argparse
from collections import Counter
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr.raw_trace import SCHEMA, atomic_json, check_trace, digest_file, read_trace, trace_path


def verify_entry(item):
    root, entry, provenance, expected_decisions = item
    try:
        path = root / entry['file']
        if path.resolve() != trace_path(root / 'traces', entry['tag']).resolve():
            raise ValueError('raw path/identity mismatch')
        if digest_file(path) != entry['raw_sha256'] or path.stat().st_size != entry['raw_bytes']:
            raise ValueError('raw file hash/size mismatch')
        document = read_trace(path)
        check_trace(document)
        metadata = document['metadata']
        if metadata['tag'] != entry['tag'] or metadata['actual_seed'] != entry['actual_seed']:
            raise ValueError('raw manifest metadata mismatch')
        if metadata['source_archive_sha256'] != entry['source_archive_sha256']:
            raise ValueError('raw source archive mismatch')
        if document['provenance'] != provenance[entry['engine']]:
            raise ValueError('raw provenance mismatch')
        policy = provenance[entry['engine']].get('replay_policy')
        if policy is not None and metadata.get('replay_policy') != policy:
            raise ValueError('raw replay policy differs from its execution provenance')
        allow_search = policy is not None and policy.get('allow_seed_search') is True
        for field in ('seed_search_attempts', 'calibration_seed_search_attempts'):
            attempts = metadata.get(field)
            if type(attempts) is not int or attempts < 0 or (not allow_search and attempts != 0):
                raise ValueError('unexpected seed search')
        if (len(document['game']['decisions']) != entry['decisions'] or
                (expected_decisions is not None and entry['decisions'] != expected_decisions)):
            raise ValueError('decision count mismatch')
        if entry.get('checks', {}).get('previous_feature_differences'):
            raise ValueError('recorded feature parity failure')
        return {'ok': True, 'raw_bytes': entry['raw_bytes'], 'decisions': entry['decisions']}
    except Exception as error:
        return {'ok': False, 'tag': entry['tag'], 'error': str(error)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--allow-bounded-sample', action='store_true')
    p.add_argument('--assignment', type=Path)
    p.add_argument('--payloads', type=Path, help='canonical JSONL payload directory')
    p.add_argument('--allow-excluded-failures', action='store_true',
                   help='audit unresolved failed exports without accepting their trajectories or resolving their causes')
    args = p.parse_args()
    if args.workers < 1:
        p.error('worker count must be positive')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    run = json.loads((args.data / 'run.json').read_text())
    source = json.loads(args.source_manifest.read_text())
    expected = {entry['tag']: entry['archive_sha256'] for entry in source['entries']}
    expected_decisions = {entry['tag']: entry['decisions'] for entry in source['entries']}
    source_decisions = expected_decisions.copy()
    policies = [provenance.get('replay_policy') for provenance in manifest['provenance'].values()]
    policy = policies[0]
    if any(value != policy for value in policies):
        raise ValueError('data versions disagree on the replay policy')
    if policy is not None and (not policy['reuse_cache'] or policy['allow_seed_search']):
        expected_decisions = {tag: None for tag in expected_decisions}
    if policy is not None and not policy['skip_failed']:
        retry_source = args.source_manifest.parent / 'manifest.json'
        if digest_file(retry_source) != source.get('source_manifest_sha256'):
            raise ValueError('retry source is not bound to the old full-exact manifest')
        for provenance in manifest['provenance'].values():
            if provenance.get('retry_source_manifest_sha256') != digest_file(retry_source):
                raise ValueError('retry source differs from execution provenance')
        previous = json.loads(retry_source.read_text())
        import numpy as np
        for row in previous['rejected']:
            tag = row['tag']
            if tag in expected:
                raise ValueError('duplicate rejected tag in retry source')
            archive = args.source_manifest.parent / (hashlib.sha256(tag.encode()).hexdigest() + '.npz')
            with np.load(archive, allow_pickle=False) as values:
                metadata = json.loads(values['_metadata'].tobytes())
            if metadata.get('tag') != tag or metadata.get('kept') is not False:
                raise ValueError('rejected source archive identity mismatch')
            expected[tag] = digest_file(archive)
            expected_decisions[tag] = None
    results = manifest['entries'] + manifest['skipped']
    seen = Counter(entry['tag'] for entry in results)
    errors = []
    if manifest['provenance'] != run['provenance']:
        errors.append('runtime provenance differs from the raw manifest')
    if manifest['source_manifest_sha256'] != digest_file(args.source_manifest):
        errors.append('source manifest hash mismatch')
    coverage_ok = set(seen) <= set(expected) if args.allow_bounded_sample else set(seen) == set(expected)
    if not coverage_ok or any(n != 1 for n in seen.values()):
        errors.append('source full-exact coverage/uniqueness mismatch')
    if manifest['schema'] != SCHEMA:
        errors.append('raw schema mismatch')
    if source.get('selection') != 'full_exact_all_formats':
        errors.append('invalid full exact source selection')
    if not manifest['entries']:
        errors.append('no raw games accepted')
    before_path = args.data / 'manifest.before-data-quarantine.json'
    original_path = before_path if before_path.exists() else args.data / 'manifest.json'
    original = json.loads(original_path.read_text())
    original_by_tag = {row['tag']: row for row in original['entries'] + original['skipped']}
    done = json.loads((args.data / 'export.done.json').read_text())
    if digest_file(original_path) != done['manifest_sha256']:
        errors.append('original export manifest binding mismatch')
    if original['entries'] != manifest['entries']:
        errors.append('accepted entries were modified during quarantine')
    for entry in source['entries']:
        if entry['archive_sha256'] != expected[entry['tag']] or entry['decisions'] != source_decisions[entry['tag']]:
            errors.append('source owners disagree on archive or decision count')
    for engine, provenance in manifest['provenance'].items():
        if (provenance.get('source_manifest_sha256') != digest_file(args.source_manifest) or
                provenance.get('assignment_sha256') != manifest['assignment_sha256'] or
                not Path(provenance.get('effective_asset_root', '')).is_absolute()):
            errors.append('invalid execution provenance binding: ' + engine)
    if before_path.exists():
        from quarantine_raw_failures import classify
        quarantine = json.loads((args.data / 'quarantine.done.json').read_text())
        if (quarantine['original_manifest_sha256'] != digest_file(before_path) or
                quarantine['manifest_sha256'] != digest_file(args.data / 'manifest.json') or
                quarantine['policy_sha256'] != digest_file(Path(__file__).with_name('quarantine_raw_failures.py'))):
            errors.append('quarantine before/after/policy binding mismatch')
    else:
        classify = None
    failures = []
    for entry in results:
        if entry['source_archive_sha256'] != expected.get(entry['tag']):
            errors.append('source archive binding mismatch: ' + entry['tag'])
        original_entry = original_by_tag.get(entry['tag'])
        if original_entry is None:
            errors.append('missing original export evidence: ' + entry['tag'])
        if entry['status'] == 'failed':
            failures.append(entry)
            if entry != original_entry or not args.allow_excluded_failures:
                errors.append('unresolved infrastructure or export failure: ' + entry['tag'])
        elif entry['status'] not in ('accepted', 'skipped'):
            errors.append('invalid terminal export status: ' + entry['tag'])
        if entry['status'] == 'skipped':
            if (entry.get('skip_code') not in ('missing_implicit_calibration_seed', 'cached_recipe_not_exact',
                                              'dropped_or_empty_reconstruction', 'cached_ability_source_unresolvable',
                                              'cached_deal_configuration_unresolvable', 'previous_failure_skipped') or
                    entry.get('provenance') != manifest['provenance'].get(entry['engine']) or
                    not entry.get('payload_row_sha256')):
                errors.append('invalid or unbound skip outcome: ' + entry['tag'])
            if entry.get('quarantined_previous_status') == 'failed':
                if (classify is None or original_entry.get('status') != 'failed' or
                        original_entry.get('reason') != entry['reason'] or
                        classify(entry['reason']) != entry.get('skip_code') or
                        entry.get('quarantine_policy_sha256') != quarantine['policy_sha256']):
                    errors.append('invalid classified failure evidence: ' + entry['tag'])
            elif entry != original_entry:
                errors.append('original skipped outcome was altered: ' + entry['tag'])
        if entry['status'] == 'accepted':
            validation = int(hashlib.sha1(entry['tag'].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.02
            if entry.get('validation') is not validation:
                errors.append('validation split differs from the canonical game split')
    if args.assignment:
        payload_roots = {value.get('payload_root', '/tmp/DBCR/hf-ingest/payloads')
                         for value in manifest['provenance'].values()}
        if len(payload_roots) != 1:
            raise ValueError('data versions disagree on the canonical payload root')
        payloads = args.payloads or Path(next(iter(payload_roots)))
        if digest_file(args.assignment) != manifest['assignment_sha256']:
            errors.append('assignment hash mismatch')
        wanted = {entry['tag']: entry for entry in manifest['skipped'] if entry['status'] == 'skipped'}
        parts = {}
        for line in args.assignment.open():
            row = json.loads(line)
            if row['tag'] in wanted:
                parts.setdefault(row['part'], {})[row['row']] = row['tag']
        checked = set()
        for part, rows in parts.items():
            with (payloads / part).open('rb') as payload:
                for index, line in enumerate(payload):
                    if index in rows:
                        tag = rows[index]
                        decoded = json.loads(line)
                        if (decoded['tag'] != tag or decoded['payload']['source']['replay_tag'] != tag or
                                hashlib.sha256(line).hexdigest() != wanted[tag]['payload_row_sha256']):
                            errors.append('skip canonical payload binding mismatch: ' + tag)
                        checked.add(tag)
        if checked != set(wanted):
            errors.append('skip payload coverage incomplete')
    elif before_path.exists():
        errors.append('quarantine verification requires canonical assignment')
    items = ((args.data, entry, manifest['provenance'], expected_decisions[entry['tag']])
             for entry in manifest['entries'])
    raw_bytes = decisions = valid = 0
    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(verify_entry, items, chunksize=16), 1):
            if not result['ok']:
                errors.append(result)
            else:
                raw_bytes += result['raw_bytes']
                decisions += result['decisions']
                valid += 1
            if i % 5000 == 0:
                print(json.dumps({'verified': i, 'valid': valid, 'errors': len(errors)}), flush=True)
    status = ('complete_with_excluded_failures' if failures else 'complete') if not errors and valid > 0 else 'failed'
    known_skipped = [entry for entry in manifest['skipped'] if entry['status'] == 'skipped']
    report = {'schema': 'daybreak-cr-raw-verification.v1', 'ok': not errors and valid > 0, 'status': status,
              'expected_old_full_exact_games': len(source_decisions), 'expected_candidate_games': len(expected),
              'raw_games': valid,
              'processed_games': len(results), 'full_scope_verified': not args.allow_bounded_sample,
              'skipped_games': len(known_skipped), 'failed_games': len(failures),
              'excluded_games': len(manifest['skipped']), 'unresolved_failures': failures,
              'raw_bytes': raw_bytes, 'decisions': decisions,
              'raw_retention': valid / max(1, len(results)), 'errors': errors,
              'skip_reasons': dict(Counter(entry['reason'] for entry in known_skipped)),
              'failed_reasons': dict(Counter(entry['reason'] for entry in failures)),
              'manifest_sha256': digest_file(args.data / 'manifest.json'),
              'source_manifest_sha256': digest_file(args.source_manifest),
              'verifier_sha256': digest_file(Path(__file__))}
    atomic_json(args.data / 'verify.done.json', report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
