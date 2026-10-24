"""What a conversion produced, per battle and in total: convert.py's index-*.jsonl (one line per try; an update-day battle may
have two) and hf_ingest.py's ingest/ rows (every battle, kept or left out and why).

    python tools/conversion_report.py CONVERT_OUT INGEST_OUT [--ledger LEDGER.jsonl]

Status per battle: exact / exact with a seed / dice: needs the true seed (no seed in --seeds reproduces it; the true seed is
not in RoyaleAPI's data -- docs/CONVERSION.md) / not reproduced / refused (the converter's reason) / crashed / not converted
yet / left out (the ingest reason). Exact rate by day: a drop flags a game update or mechanic not yet known.
"""
import argparse
import collections
import json
from pathlib import Path

ORDER = ['exact', 'exact with a seed', 'dice: needs the true seed', 'not reproduced', 'refused', 'crashed']


def status(e):
    s = e.get('search') or {}
    if e.get('kept') or (e.get('hp_error') == 0 and e.get('winner') and e.get('crowns')):
        return ('exact with a seed' if s.get('seed') else 'exact'), None
    why = e.get('why') or ''
    if why == 'native engine crashed twice':
        return 'crashed', why
    if 'hp_error' not in e:
        return 'refused', why[:120]
    if s.get('seed_dependent'):
        return 'dice: needs the true seed', f'HP off {e.get("hp_error")}'
    return 'not reproduced', f'HP off {e.get("hp_error")}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('convert_out', type=Path)
    ap.add_argument('ingest_out', type=Path)
    ap.add_argument('--ledger', type=Path)
    args = ap.parse_args()
    best = {}
    for f in args.convert_out.glob('index-*.jsonl'):
        for line in open(f):
            if not line.strip():
                continue
            e = json.loads(line)
            st = status(e)
            if e['tag'] not in best or ORDER.index(st[0]) < ORDER.index(best[e['tag']][0][0]):
                best[e['tag']] = (st, e)
    rows = []
    for f in sorted((args.ingest_out / 'ingest').glob('*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if r['status'] == 'kept':
                got = best.get(r['tag'])
                if got:
                    (st, why), e = got
                    r.update(status=st, reason=why, engine=e.get('engine'), search=e.get('search'))
                else:
                    r.update(status='not converted yet')
            rows.append(r)
    if args.ledger:
        args.ledger.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    c = collections.Counter(r['status'] for r in rows)
    done = [r for r in rows if r['status'] in ORDER]
    ok = sum(1 for r in done if r['status'].startswith('exact'))
    print(f'{len(rows)} battles; {len(done)} converted; exact {ok} ({100 * ok / max(1, len(done)):.1f}% of converted)\n')
    for s, n in c.most_common():
        print(f'  {n:8}  {s}')
    print('\nleft out:')
    for s, n in collections.Counter(r['reason'] for r in rows if r['status'] == 'left out').most_common(10):
        print(f'  {n:8}  {s}')
    print('\nrefused:')
    for s, n in collections.Counter(r['reason'] for r in rows if r['status'] == 'refused').most_common(8):
        print(f'  {n:8}  {s}')
    by = collections.defaultdict(lambda: [0, 0])
    for r in done:
        by[(r['battle_time'] or '')[:10]][0] += 1
        by[(r['battle_time'] or '')[:10]][1] += r['status'].startswith('exact')
    print('\nexact by day (UTC):')
    for d, (n, k) in sorted(by.items()):
        print(f'  {d}  {k:7}/{n:7}  {100 * k / n:5.1f}%')


if __name__ == '__main__':
    main()
