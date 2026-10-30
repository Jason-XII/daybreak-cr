"""What are hero / champion presses worth in the simulator? IL with its ability choices masked (arm il_noab) against
plain IL, matched deck pairs, seeds and both seats, as placement_arena.py. Score < 0.5: IL's presses win games.
Each row also counts confirmed activations per seat. Run inside a Slurm allocation.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn


class NoAbility(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, state=None):
        out = dict(self.base(x, state))
        what = out['what'].clone()
        what[..., 2:] = float('-inf')  # ABILITY1, ABILITY2
        out['what'] = what
        return out

    def tiles(self, out, x, slot):
        return self.base.tiles(out, x, slot)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--il', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--engine-lib', type=Path)
    ap.add_argument('--engine-assets', type=Path)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--games', type=int, default=800)
    ap.add_argument('--seed', type=int, default=2801)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    shard, shards = map(int, args.shard.split('/'))
    sys.path.insert(0, str(args.source.resolve()))
    from cr import policy
    from cr.engine import Engine, Game

    torch.set_num_threads(1)
    il, vocab, _ = policy.load(args.il, args.device)
    noab = NoAbility(il).eval()
    for model in (il, noab):
        model.temperature, model.mix, model.overlap = args.temperature, 0.0, True
    pool = json.loads(args.pool.read_text())
    indices = np.random.default_rng(args.seed).integers(len(pool), size=args.games // 2)
    args.out.mkdir(parents=True, exist_ok=True)
    engine = Engine(args.engine, lib=args.engine_lib, assets=args.engine_assets)
    path = args.out / f'il_noab-s{shard}.jsonl'
    done = {(r['pair_index'], r['a_owner']) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    with path.open('a') as stream:
        for i in range(shard, len(indices), shards):
            for owner in (0, 1):
                if (i, owner) in done:
                    continue
                started = time.perf_counter()
                action_seed = args.seed + i * 2 + owner
                random.seed(action_seed)
                np.random.seed(action_seed)
                torch.manual_seed(action_seed)
                engine.overlap = [True, True]
                game = Game(engine, pool[int(indices[i])], args.seed + i, {0: vocab, 1: vocab})
                agents = [policy.Agent(noab if o == owner else il, o, args.device) for o in (0, 1)]
                ready = [0, 0]
                while not game.over:
                    for o in (0, 1):
                        ready[o] += any(a.ready for a in game.state.players[o].abilities)
                    views = game.views()
                    game.step({a.owner: a.act(views[a.owner]) for a in agents})
                state = game.state
                if not state.finalized:
                    raise RuntimeError('battle did not finalize')
                stats = engine.ability_stats
                row = dict(pair_index=i, pool_index=int(indices[i]), seed=args.seed + i, a_owner=owner,
                           score=0.5 if state.result not in (0, 1) else float(state.result == owner),
                           crowns_a=state.crowns[owner], crowns_b=state.crowns[1 - owner], tick=state.tick,
                           ready_a=ready[owner], ready_b=ready[1 - owner],
                           confirmed_a=stats[owner]['confirmed'], confirmed_b=stats[1 - owner]['confirmed'],
                           seconds=round(time.perf_counter() - started, 3))
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps(row), flush=True)
    print('completed', args.shard, flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
