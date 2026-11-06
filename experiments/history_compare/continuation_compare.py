import argparse
import json
from pathlib import Path


def compare(baseline, split):
    protocols = [json.loads((path / 'protocol.json').read_text()) for path in (baseline, split)]
    if [p['arm'] for p in protocols] != ['baseline', 'split']:
        raise ValueError('Pair must be baseline then split')
    keys = ('seed', 'gbs', 'mbs', 'workers', 'real_ddp', 'optimizer_reset', 'origin_manifest_sha256',
            'origin_checkpoint_sha256', 'order_hashes', 'settings', 'config', 'source_sha256', 'compile', 'bf16',
            'gradient_checkpointing', 'validation_views', 'core_sha256', 'continuation_sha256')
    def normalize(protocol, key):
        value = dict(protocol[key]) if key in ('config', 'settings') else protocol[key]
        if key in ('config', 'settings'):
            value.pop('split_query', None)
            value.pop('split_alpha', None)
        return value
    matched = {key: normalize(protocols[0], key) == normalize(protocols[1], key) for key in keys}
    if not all(matched.values()):
        raise ValueError(f'Paired continuation contract differs: {matched}')
    if protocols[0]['index'] != protocols[1]['index']:
        raise ValueError('Use the same resume index for the controlled33200 comparison')
    results = {}
    for arm, path in (('baseline', baseline), ('split', split)):
        validation = json.loads((path / 'validation-03200.json').read_text())
        if validation['step'] != 33200 or validation['index'] != 3200:
            raise ValueError('Require exact33200 validation')
        if validation['entries'] != protocols[0]['validation_views'] or len(validation['entries']) < 32:
            raise ValueError('Validation views differ')
        if not (path / 'checkpoint-03200.pt').is_file():
            raise ValueError('Missing exact33200 checkpoint')
        results[arm] = dict(metrics=validation['metrics'], tile_labels=validation['tile_labels'],
                           snapshot=str(path / 'checkpoint-03200.pt'))
    if results['baseline']['tile_labels'] != results['split']['tile_labels']:
        raise ValueError('Paired tile-label denominator differs')
    return dict(index=3200, step=33200, resumed_index=protocols[0]['index'], contract_equal=matched, results=results,
                limitation='JSON contracts and snapshot existence verified, not tensor contents or CUDA bitwise replay')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = compare(args.baseline, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2))
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
