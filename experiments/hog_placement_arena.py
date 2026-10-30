"""Paired Hog placement swaps: specialist vs its general-policy initialization.

Four arms share opponent decks/seeds/seats. Both hybrid networks keep independent recurrent
state on the actual hybrid history; only timing/card/ability outputs or placement are exchanged.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from placement_arena import Hybrid, digest, self_test

ARMS = ('baseline', 'specialist', 'base_specialist_tile', 'specialist_base_tile')


def match_pair(template, settings, seed, owner):
    pair = copy.deepcopy(template)
    pair['decks'][0] = random.Random(seed).sample(settings['deck'], len(settings['deck']))
    pair['sc'][0] = copy.deepcopy(settings['deck_sc'])
    if owner == 1:
        for field in ('decks', 'sc', 'kt', 'exp'):
            pair[field].reverse()
    return pair


def verify_pairing():
    template = dict(decks=[[{'d': 1}], [{'d': 2}]], sc=[[{'d': 3}], [{'d': 4}]], kt=[11, 12], exp=[11, 12])
    settings = dict(deck=[{'d': 10, 'el': 2}, {'d': 11, 'el': 1}], deck_sc=[{'d': 99}])
    a, b = (match_pair(template, settings, 7, owner) for owner in (0, 1))
    assert all(a[k] == list(reversed(b[k])) for k in ('decks', 'sc', 'kt', 'exp'))
    assert a['decks'][1] == template['decks'][1] and template['decks'][0] == [{'d': 1}]
    self_test()
    print('Matched opponent/deck/seed seat reversal: passed', flush=True)


def summarize(root, games):
    n = games // 2
    expected = {(i, owner) for i in range(n) for owner in (0, 1)}
    scores, summaries = {}, {}
    indices = np.random.default_rng(2901).integers(n, size=(10000, n))
    for arm in ARMS:
        rows = [json.loads(line) for p in sorted((root / 'results').glob(arm + '-s*.jsonl'))
                for line in p.read_text().splitlines()]
        keyed = {(r['pair_index'], r['a_owner']): r for r in rows}
        if len(rows) != games or set(keyed) != expected:
            raise RuntimeError(f'{arm}: incomplete/duplicate panel: {len(rows)} rows')
        pairs = np.array([(keyed[(i, 0)]['score'] + keyed[(i, 1)]['score']) / 2 for i in range(n)])
        scores[arm] = pairs
        summaries[arm] = dict(games=games, score=float(pairs.mean()),
                              ci95=np.quantile(pairs[indices].mean(1), [.025, .975]).tolist(),
                              wins=sum(r['score'] == 1 for r in rows), draws=sum(r['score'] == .5 for r in rows),
                              losses=sum(r['score'] == 0 for r in rows))
    effects = {}
    for name, a, b in [('total_specialization', 'specialist', 'baseline'),
                       ('placement_added_to_baseline', 'base_specialist_tile', 'baseline'),
                       ('specialist_placement_removed', 'specialist', 'specialist_base_tile')]:
        diff = scores[a] - scores[b]
        effects[name] = dict(points=float(diff.mean() * 100),
                             ci95_points=(np.quantile(diff[indices].mean(1), [.025, .975]) * 100).tolist())
    result = dict(arms=summaries, effects=effects,
                  note='All arms play Hog against frozen general gen-r6e v140 with varied decks. Bootstrap resamples entire matched seed/deck pairs across both seats and all arms.')
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Hog v120 placement arena', '', '| Arm | Score | 95% paired interval |', '|---|---:|---:|']
    for arm, r in summaries.items():
        lo, hi = r['ci95']; lines.append(f'| {arm} | {100*r["score"]:.1f}% | {100*lo:.1f}–{100*hi:.1f}% |')
    lines += ['', 'Placement effects (percentage points):']
    for name, r in effects.items():
        lo, hi = r['ci95_points']; lines.append(f'- {name}: {r["points"]:+.1f}, 95% interval {lo:+.1f} to {hi:+.1f}.')
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'baseline', 'specialist', 'engine', 'engine_lib', 'engine_assets', 'pool', 'settings', 'out'):
        p.add_argument('--' + key.replace('_', '-'), type=Path)
    p.add_argument('--games', type=int, default=400)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--seed', type=int, default=291029)
    p.add_argument('--temperature', type=float, default=.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--summary', action='store_true')
    a = p.parse_args()
    if a.self_test:
        return verify_pairing()
    if a.games < 2 or a.games % 2:
        p.error('games must be positive and even')
    if a.summary:
        return summarize(a.out, a.games)
    if any(getattr(a, k) is None for k in ('source', 'baseline', 'specialist', 'engine', 'pool', 'settings', 'out')):
        p.error('source, checkpoints, engine, pool, settings and out are required')
    shard, shards = map(int, a.shard.split('/'))
    if not 0 <= shard < shards:
        p.error('invalid shard')
    sys.path.insert(0, str(a.source))
    from cr import policy
    from cr.engine import Engine, Game
    torch.set_num_threads(1)
    baseline, vocab, _ = policy.load(a.baseline, a.device)
    specialist, other, _ = policy.load(a.specialist, a.device)
    assert vocab.index == other.index
    models = dict(baseline=baseline, specialist=specialist,
                  base_specialist_tile=Hybrid(baseline, specialist),
                  specialist_base_tile=Hybrid(specialist, baseline))
    for model in models.values():
        model.eval(); model.temperature = a.temperature; model.mix = 0.; model.overlap = True
    pool, settings = json.loads(a.pool.read_text()), json.loads(a.settings.read_text())
    indices = np.random.default_rng(a.seed).integers(len(pool), size=a.games // 2)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                hashes={k: digest(getattr(a, k)) for k in ('baseline', 'specialist', 'pool', 'settings')},
                script_sha256=digest(Path(__file__)), hybrid_sha256=digest(Path(__file__).with_name('placement_arena.py')),
                arms=list(ARMS), ability_explore=0., overlap=True)
    metadata = a.out / f'metadata-s{shard}.json'
    if metadata.exists():
        assert json.loads(metadata.read_text()) == meta, 'resume requires identical scripts/checkpoints/settings'
    else:
        metadata.write_text(json.dumps(meta, indent=2) + '\n')
    done, streams = {}, {}
    for arm in ARMS:
        path = a.out / f'{arm}-s{shard}.jsonl'
        done[arm] = {(r['pair_index'], r['a_owner']) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
        streams[arm] = path.open('a')
    sim = Engine(a.engine, lib=a.engine_lib, assets=a.engine_assets)
    for i in range(shard, len(indices), shards):
        for owner in (0, 1):
            for arm, model in models.items():
                if (i, owner) in done[arm]:
                    continue
                started = time.perf_counter(); seed = a.seed + i
                action_seed = a.seed + 2*i + owner
                random.seed(action_seed); np.random.seed(action_seed); torch.manual_seed(action_seed)
                sim.overlap = [True, True]
                pair = match_pair(pool[int(indices[i])], settings, seed, owner)
                game = Game(sim, pair, seed, {0: vocab, 1: vocab})
                agents = [policy.Agent(model if o == owner else baseline, o, a.device) for o in (0, 1)]
                refused = [0, 0]
                while not game.over:
                    views = game.views()
                    submitted, transition = game.step({agent.owner: agent.act(views[agent.owner]) for agent in agents})
                    for action, ok in zip(submitted, transition.executed):
                        refused[action.owner] += not ok
                state = game.state
                assert state.finalized
                score = .5 if state.result not in (0, 1) else float(state.result == owner)
                row = dict(arm=arm, pair_index=i, pool_index=int(indices[i]), seed=seed, a_owner=owner,
                           score=score, crowns_a=state.crowns[owner], crowns_b=state.crowns[1-owner],
                           tick=state.tick, refused_a=refused[owner], refused_b=refused[1-owner],
                           seconds=round(time.perf_counter()-started, 3))
                streams[arm].write(json.dumps(row)+'\n'); streams[arm].flush(); print(json.dumps(row), flush=True)
    for stream in streams.values(): stream.close()
    print('completed', a.shard, flush=True); sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
