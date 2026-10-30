"""One manifest.json for a dataset converted as several convert.py outputs side by side (one folder per stream).

    python tools/merge_manifests.py ~/il-data/v1 's?-?' [--workers 16]

A folder's own manifest.json lists only the games of its last convert.py run (each run writes the manifest of its own
assignment), so the entries are read back from every archive's metadata. Their files are named relative to ROOT; the
vocabulary must be the same in every folder. Presses are labelled at the tap (convert.py since PR #14): presses_at_tap.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def metadata(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['_metadata'].tobytes())
    folder = path.parent.name
    return ([dict(e, file=f'{folder}/{e["file"]}') for e in meta['entries']],
            None if meta['kept'] else dict(tag=meta['tag'], why=meta.get('why')))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path)
    ap.add_argument('folders', help='glob of the output folders under ROOT')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    folders = sorted(p for p in args.root.glob(args.folders) if (p / 'manifest.json').exists())
    vocabs = {json.dumps(json.loads((p / 'manifest.json').read_text())['schema']['vocab']) for p in folders}
    if len(vocabs) != 1:
        raise SystemExit(f'{len(vocabs)} different vocabularies in {len(folders)} folders')
    paths = [p for f in folders for p in sorted(f.glob('*.npz'))]
    print(f'{len(folders)} folders, {len(paths)} archives', flush=True)
    entries, rejected = [], []
    with Pool(args.workers) as pool:
        for e, r in pool.imap(metadata, paths, chunksize=256):
            entries += e
            if r:
                rejected.append(r)
    tags = [e['tag'] for e in entries]
    assert len(set(zip(tags, (e['owner'] for e in entries)))) == len(entries), 'a game appears twice'
    manifest = dict(schema=dict(vocab=json.loads(vocabs.pop()), legal=True, presses_at_tap=True), entries=entries,
                    rejected=rejected)
    (args.root / 'manifest.tmp').write_text(json.dumps(manifest))
    (args.root / 'manifest.tmp').replace(args.root / 'manifest.json')
    print(f'{len(set(tags))} games kept, {len(rejected)} rejected, {sum(e["validation"] for e in entries)} validation '
          f'views -> {args.root / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
