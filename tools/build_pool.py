"""A PPO deck pool from converted human battles (27 Oct: the fresh run on the newest engine plays October's decks): each
battle of the given data versions, through convert.prepare (the same match the conversion played), becomes one pool
pair: both decks with their card levels and forms, tower troops, king and experience levels. A battle prepare() refuses
is left out.

    python tools/build_pool.py OUT.json ~/il-data/hf-770k/s*/assignment.jsonl --engines d22 --sample 40000

The assignment's rows name their payload part; payloads are read from the 'payloads' folder beside each assignment.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert  # noqa: E402


def pair_of(match):
    battle = match['battle']
    return dict(decks=[battle[f'deck{o}']['sp'] for o in (0, 1)], sc=[battle[f'deck{o}']['sc'] for o in (0, 1)],
                lvlcap=battle['lvlcap'], cardlvlmin=battle['cardlvlmin'],
                kt=[battle['hbd'][o]['kt'] for o in (0, 1)], exp=[battle[f'avatar{o}']['expLevel'] for o in (0, 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path)
    ap.add_argument('assignments', type=Path, nargs='+')
    ap.add_argument('--engines', default='d22', help='data versions to take battles from (comma-separated)')
    ap.add_argument('--sample', type=int, default=40000, help='at most this many pairs, a random sample (0: all)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    names = set(args.engines.split(','))
    jobs = [dict(json.loads(line), root=a.parent) for a in args.assignments for line in a.open()]
    jobs = [j for j in jobs if j['engine'] in names]
    rng = random.Random(args.seed)
    rng.shuffle(jobs)
    if args.sample:
        jobs = jobs[:int(args.sample * 1.1)]  # a few are refused; trimmed below
    pool, why, tables = [], Counter(), {}
    for job in sorted(jobs, key=lambda j: (str(j['root']), j['part'], j['row'])):  # each part read once
        try:
            payload = convert.payload_of(dict(job, path=str(job['root'] / 'payloads' / job['part'])), tables)
            pool.append(pair_of(convert.prepare(payload)['match']))
        except (ValueError, KeyError) as error:
            why[type(error).__name__ + ': ' + str(error)[:60]] += 1
    rng.shuffle(pool)
    pool = pool[:args.sample] if args.sample else pool
    args.out.write_text(json.dumps(pool))
    print(json.dumps(dict(read=len(jobs), pairs=len(pool), refused=sum(why.values()), why=why.most_common(8))))


if __name__ == '__main__':
    main()
