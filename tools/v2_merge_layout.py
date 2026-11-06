import argparse
import hashlib
import json
from pathlib import Path


def merge(shards):
    if not shards:
        raise ValueError('No verified layout shards')
    identities = [shard['manifest_sha256'] for shard in shards]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate dataset layout shard')
    for shard in shards:
        if shard['scope'] != 'full_verified_dataset_shard':
            raise ValueError('Sample layouts cannot be merged for production')
        if shard['canonicalization'] != 'known-coordinates-own-view-v1' or shard['encoding'] != 'named-signed-log-valid-v1':
            raise ValueError('Layout feature semantics differ')
    result = dict(scope='full_verified_dataset', canonicalization='known-coordinates-own-view-v1',
                  encoding='named-signed-log-valid-v1', episodes=sum(s['episodes'] for s in shards),
                  frames=sum(s['frames'] for s in shards),
                  max_entities=max(s['max_entities'] for s in shards),
                  max_pending=max(s['max_pending'] for s in shards),
                  source_shards=[{key: s[key] for key in ('manifest_sha256', 'verification_sha256', 'episodes', 'frames')}
                                 for s in shards])
    for name in ('globals', 'entities'):
        paths = sorted({path for shard in shards for path in shard[name]['paths']})
        if any(shard[name]['width'] != 2*len(shard[name]['paths']) for shard in shards):
            raise ValueError('Invalid shard channel width')
        result[name] = dict(paths=paths, width=2*len(paths))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shards', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = merge([json.loads(path.read_text()) for path in args.shards])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2).encode()
    temporary = args.out.with_suffix('.tmp')
    temporary.write_bytes(encoded)
    temporary.replace(args.out)
    print(json.dumps(dict(sha256=hashlib.sha256(encoded).hexdigest(), episodes=result['episodes'],
                         globals=result['globals']['width'], entities=result['entities']['width'])))


if __name__ == '__main__':
    main()
