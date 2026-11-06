import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / 'cr/inputs_v2'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr.inputs_v2.v2_contract import numeric_frame, player_frame, read_episode
from cr.inputs_v2.v2_numeric import flatten
from cr.inputs_v2.verification_gate import verified_manifest


def bounded_results(pool, function, items, limit):
    iterator = iter(items)
    pending = set()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < limit and not exhausted:
            try:
                pending.add(pool.submit(function, next(iterator)))
            except StopIteration:
                exhausted = True
        if pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()


def inspect_episode(item):
    path, sha256 = item
    episode = read_episode(path, sha256)
    globals_paths, entity_paths = set(), set()
    frames = len(episode['frames'])
    max_entities, max_pending = 0, 0
    for world in episode['frames']:
        max_entities = max(max_entities, len(world['entities']))
        for owner in (0, 1):
            frame = numeric_frame(player_frame(world, owner), owner)
            for entity in frame.pop('entities'):
                entity_paths.update(path for path, value in flatten(entity))
            globals_paths.update(path for path, value in flatten(frame))
            max_pending = max(max_pending, sum(c['kind'] == 1 and c['side'] == owner and
                                              c['t2'] > world['game_tick'] for c in world['queue']))
    return dict(globals=globals_paths, entities=entity_paths, frames=frames,
                max_entities=max_entities, max_pending=max_pending)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verifier-sha256', required=True)
    args = parser.parse_args()
    verification = json.loads((args.root/'verify.done.json').read_text())
    source_paths = [args.root/'manifest.json', args.root/'verify.done.json',
                    ROOT/'v2_contract.py', ROOT/'v2_numeric.py', ROOT/'input_spec.py', Path(__file__)]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    manifest_bytes = (args.root/'manifest.json').read_bytes()
    manifest = verified_manifest(manifest_bytes, verification, args.verifier_sha256)
    entries = manifest['entries']
    if verification['verified_games'] != len(entries) or verification['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Verification does not cover the current complete manifest')
    if not entries:
        raise ValueError('Empty verified dataset')
    items = []
    for entry in entries:
        path = args.root/entry['file']
        if not path.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('Episode path escapes dataset')
        items.append((str(path), entry['sha256']))
    paths = {'globals': set(), 'entities': set()}
    frames, maximum_entities, maximum_pending = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(bounded_results(pool, inspect_episode, items, max(1, args.workers)*2), 1):
            for name in paths:
                paths[name].update(row[name])
            frames += row['frames']
            maximum_entities = max(maximum_entities, row['max_entities'])
            maximum_pending = max(maximum_pending, row['max_pending'])
            if index % 1000 == 0:
                print(json.dumps(dict(processed=index, total=len(items), frames=frames)), flush=True)
    if maximum_pending > 2:
        raise ValueError('Verified dataset contains pending queues exceeding fixed slot capacity')
    for path, expected in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Layout source changed during scan: '+path)
    result = dict(scope='full_verified_dataset_shard', manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  verification_sha256=hashlib.sha256((args.root/'verify.done.json').read_bytes()).hexdigest(),
                  episodes=len(items), frames=frames, max_entities=maximum_entities, max_pending=maximum_pending,
                  canonicalization='known-coordinates-own-view-v1', encoding='named-signed-log-valid-v1',
                  source_hashes=source_hashes)
    for name, values in paths.items():
        result[name] = dict(paths=sorted(values), width=2*len(values))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.out)
    print('LAYOUT_FINISHED '+json.dumps(dict(episodes=len(items), frames=frames,
         global_width=result['globals']['width'], entity_width=result['entities']['width'])), flush=True)


if __name__ == '__main__':
    main()
