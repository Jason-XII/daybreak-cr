import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from cr.splitquery import migrate_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Model-only migration to the experimental SplitQuery tile head')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    result = migrate_checkpoint(saved, args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as output:
        torch.save(result, output)


if __name__ == '__main__':
    main()
