"""Paired target-deck and broad-deck evaluations for the automated curriculum."""
import argparse
import collections
import copy
import json
from pathlib import Path
import random

from common_deck_arena import match, tasks


def make_plan(meta, pool, target, candidate, reference, frozen):
    result = []
    if target is not None:
        for task in tasks(meta):
            if target not in (task['a'], task['b']):
                continue
            owner = task['a_owner'] if task['a'] == target else 1-task['a_owner']
            opponent = task['b'] if task['a'] == target else task['a']
            for kind, policy in [('target', candidate), ('target_reference', reference)]:
                result.append(dict(id=len(result), kind=kind, opponent_deck=opponent,
                                   pair=match(meta, task), seed=task['seed'], a_owner=owner,
                                   candidate=str(policy), opponent=str(frozen)))
    rng = random.Random(291029)
    # Independent human deck appearances; same sampled decks, hand seed and both seats.
    for i in range(500):
        pair = copy.deepcopy(rng.choice(pool))
        pair['decks'] = [copy.deepcopy(rng.choice(pool)['decks'][rng.randrange(2)]) for _ in range(2)]
        pair['sc'] = [meta['tower'],meta['tower']]
        for owner in (0,1):
            seated = copy.deepcopy(pair)
            if owner:
                for key in ('decks','sc','kt','exp'):
                    seated[key].reverse()
            result.append(dict(id=len(result),kind='general',pair_index=i,pair=seated,
                               seed=29100000+i,a_owner=owner,candidate=str(candidate),opponent=str(reference)))
    return dict(tasks=result,target=target,candidate=str(candidate),reference=str(reference),frozen=str(frozen),
                usages={str(d['index']):d['usage'] for d in meta['decks']})


def summarize(root):
    import numpy as np
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    rows=[json.loads(l) for f in sorted((root/'results').glob('games-*.jsonl')) for l in f.read_text().splitlines()]
    expected={t['id']:t for t in plan['tasks']}
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):
        raise RuntimeError(f'incomplete/duplicate evaluation: {len(rows)}/{len(expected)}')
    for r in rows:
        if any(r[k]!=expected[r['id']][k] for k in ('seed','a_owner','kind','candidate','opponent')):
            raise RuntimeError('evaluation task mismatch')
    grouped=collections.defaultdict(dict)
    for r in rows:
        key=r.get('pair_index') if r['kind']=='general' else (r['opponent_deck'],r['seed'])
        grouped[r['kind']][key]=grouped[r['kind']].get(key,{})|{r['a_owner']:r['score']}
    if any(set(seats)!={0,1} for by_seed in grouped.values() for seats in by_seed.values()):
        raise RuntimeError('missing paired seat')
    general=np.array([sum(s.values())/2 for s in grouped['general'].values()])
    rng=np.random.default_rng(291029)
    gen_ci=np.quantile(general[rng.integers(len(general),size=(4000,len(general)))].mean(1),[.025,.975]).tolist()
    summary=dict(games=len(rows),general_score=float(general.mean()),general_ci95=gen_ci,
                 target_score=None,target_reference_score=None,target_gain=None,
                 refused=sum(sum(r['refused']) for r in rows))
    if plan['target'] is not None:
        weights={int(k):v for k,v in plan['usages'].items() if int(k)!=plan['target']}
        total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
        scores={}
        for kind in ('target','target_reference'):
            by_deck=collections.defaultdict(list)
            for (deck,seed),seats in grouped[kind].items():by_deck[deck].append(sum(seats.values())/2)
            scores[kind]=sum(weights[d]*float(np.mean(v)) for d,v in by_deck.items())
        summary.update(target_score=scores['target'],target_reference_score=scores['target_reference'],
                       target_gain=scores['target']-scores['target_reference'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'report.md').write_text('# Curriculum evaluation\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();summarize(a.root)
