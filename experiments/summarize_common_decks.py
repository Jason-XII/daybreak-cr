"""Summarize a complete, seat-balanced common-deck round robin."""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from common_deck_arena import tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args(); root = args.root
    meta = json.loads((root / 'decks.json').read_text())
    rows = [json.loads(line) for f in sorted((root / 'results').glob('games-*.jsonl'))
            for line in f.read_text().splitlines()]
    expected = {r['id']: r for r in tasks(meta)}
    if len(rows) != len(expected) or {r['id'] for r in rows} != set(expected):
        raise RuntimeError(f'incomplete/duplicate round robin: {len(rows)}/{len(expected)}')
    paired = collections.defaultdict(dict)
    for row in rows:
        if any(row[k] != expected[row['id']][k] for k in ('a','b','seed','a_owner')):
            raise RuntimeError('result task does not match schedule')
        paired[(row['a'],row['b'],row['seed'])][row['a_owner']] = row['score']
    samples = collections.defaultdict(list)
    for (a,b,seed), seats in paired.items():
        if set(seats) != {0,1}:
            raise RuntimeError('unpaired seats')
        score = (seats[0]+seats[1])/2
        samples[a,b].append(score); samples[b,a].append(1-score)
    n = len(meta['decks']); matrix = [[None]*n for _ in range(n)]; ranking = []
    rng = np.random.default_rng(1029)
    for a,d in enumerate(meta['decks']):
        opponents = [b for b in range(n) if b != a]
        weights = np.array([meta['decks'][b]['usage'] for b in opponents]); weights /= weights.sum()
        bootstrap = np.zeros(4000)
        for b,w in zip(opponents,weights):
            values = np.array(samples[a,b]); assert len(values) == 10
            matrix[a][b] = float(values.mean())
            bootstrap += w * values[rng.integers(10,size=(4000,10))].mean(1)
        score = float(sum(weights[k]*matrix[a][b] for k,b in enumerate(opponents)))
        ranking.append(dict(index=a, names=d['names'], cards=d['cards'], usage=d['usage'],
                            games=20*(n-1), weighted_score=score,
                            equal_weight_score=float(np.mean([matrix[a][b] for b in opponents])),
                            ci95=np.quantile(bootstrap,[.025,.975]).tolist()))
    ranking.sort(key=lambda r:r['weighted_score'])
    summary = dict(games=len(rows),decks=n,total_human_views=meta['total_views'],
                   common_deck_usage=sum(d['usage'] for d in meta['decks']),ranking=ranking,matrix=matrix,
                   note='Scores measure this frozen model with each deck against common decks, not intrinsic skill separated from deck matchup strength. Draw=0.5; human-frequency weights normalized over other qualifying decks; 95% intervals bootstrap paired seeds within each matchup.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'matchups.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['deck']+list(range(n)))
        w.writerows([[a]+['' if v is None else round(100*v,2) for v in row] for a,row in enumerate(matrix)])
    lines=['# Common-deck self-play diagnosis: general gen-r6e v320','',
           f'{len(rows):,} games; {n} decks; 20 games per distinct pair (10 seeds × both seats).',
           f"Eligibility: ≥0.5% of {meta['total_views']:,} unique October human player-deck appearances.",
           'Level16, Princess Tower, modal hero/evolution forms, temperature0.05, overlap enabled, no forced ability exploration.','',
           '| Rank | ID | Human usage | Weighted score | 95% interval | Equal-weight score | Cards |',
           '|---:|---:|---:|---:|---:|---:|---|']
    for rank,r in enumerate(ranking,1):
        lo,hi=r['ci95']
        lines.append(f"| {rank} | {r['index']} | {100*r['usage']:.2f}% | {100*r['weighted_score']:.1f}% | {100*lo:.1f}–{100*hi:.1f}% | {100*r['equal_weight_score']:.1f}% | {', '.join(r['names'])} |")
    lines += ['',summary['note'],'','The intervals cover game sampling noise, conditional on these decks and configurations; they do not cover uncertainty in human usage or variant choice.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
