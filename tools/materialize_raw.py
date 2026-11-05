#!/usr/bin/env python3
"""Build feature archives from raw traces without loading or simulating the native engine."""
import argparse
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert as C
from cr import observation as O
from cr.raw_trace import atomic_json, check_trace, digest_file, read_trace


def build_one(item):
    raw, out, entry, recipe = item
    source = raw / entry['file']
    if digest_file(source) != entry['raw_sha256']:
        raise ValueError('raw archive hash mismatch')
    document = read_trace(source)
    check_trace(document)
    target = C.archive_path(out, entry['tag'])
    if target.exists():
        old = C.read_metadata(target)
        if old.get('raw_sha256') != entry['raw_sha256'] or old.get('feature_recipe') != recipe:
            raise ValueError('feature cache was built with a different recipe or raw source')
    else:
        metadata = {**document['metadata'], 'raw_sha256': entry['raw_sha256'], 'feature_recipe': recipe}
        C.write_archive(target, document['game'], O.Vocab.load(), O.cached_legal(), metadata)
    return [{**value, 'archive_sha256': digest_file(target), 'raw_sha256': entry['raw_sha256']}
            for value in C.read_metadata(target)['entries']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error('worker count must be positive and --limit nonnegative')
    if args.raw.resolve() == args.out.resolve():
        raise ValueError('raw data and derived feature cache must be separate')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.materialize.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((args.raw / 'manifest.json').read_text())
    selected = manifest['entries'][:args.limit] if args.limit else manifest['entries']
    recipe = {'converter_sha256': digest_file(Path(C.__file__)),
              'observation_sha256': digest_file(Path(O.__file__)),
              'assets': {p.name: digest_file(p) for p in sorted(O.ASSETS.glob('*.json'))},
              'native_feature_sha256': digest_file(Path(O._native_features.__file__))
                                       if O._native_features is not None else None}
    entries = []
    with mp.Pool(args.workers) as pool:
        for group in pool.imap_unordered(build_one,
                ((args.raw, args.out, entry, recipe) for entry in selected), chunksize=4):
            entries.extend(group)
    entries.sort(key=lambda entry: (entry['tag'], entry['owner']))
    atomic_json(args.out / 'manifest.json', {
        'schema': {'vocab': list(O.Vocab.load().index), 'legal': True, 'presses_at_tap': True},
        'selection': 'full_exact_modern_from_raw', 'entries': entries, 'rejected': [],
        'raw_manifest_sha256': digest_file(args.raw / 'manifest.json'), 'feature_recipe': recipe})
    print(json.dumps({'games': len(entries) // 2, 'engine_replays': 0}), flush=True)


if __name__ == '__main__':
    main()
