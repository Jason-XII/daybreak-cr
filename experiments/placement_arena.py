"""Separate-network placement swaps against IL, with matched deck pairs, seeds and seats.

Arms: ppo (all PPO), il_ppo_tile (IL what/card, PPO tile), ppo_il_tile (PPO what/card, IL tile).
Every network keeps its own recurrent state on the actual hybrid history. Run inside a Slurm allocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn


class Hybrid(nn.Module):
    def __init__(self, base, donor):
        super().__init__()
        self.base, self.donor = base, donor

    def forward(self, x, state=None):
        sb, sd = state if state is not None else (None, None)
        base, tile = self.base(x, sb), self.donor(x, sd)
        out = dict(base)
        for key in ('core', 'grid', 'tokens'):
            if key in tile:
                out[key] = tile[key]
        out['state'] = (base['state'], tile['state'])
        return out

    def tiles(self, out, x, slot):
        return self.donor.tiles(out, x, slot)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def self_test():
    class Stub(nn.Module):
        def __init__(self, marker):
            super().__init__()
            self.marker = marker

        def forward(self, x, state=None):
            return dict(what=self.marker, which=self.marker, value=self.marker,
                        core=self.marker, grid=self.marker, tokens=self.marker,
                        state=(state or 0) + self.marker)

        def tiles(self, out, x, slot):
            assert out['core'] == out['grid'] == out['tokens'] == self.marker
            return self.marker, slot

    for base, donor in ((2, 7), (7, 2)):
        hybrid = Hybrid(Stub(base), Stub(donor))
        out = hybrid(None)
        assert out['what'] == out['which'] == out['value'] == base
        assert out['state'] == (base, donor)
        assert hybrid.tiles(out, None, 3) == (donor, 3)
        assert hybrid(None, out['state'])['state'] == (2 * base, 2 * donor)
    print('Hybrid routing, chosen-card forwarding and independent recurrent states: passed', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--il', type=Path)
    ap.add_argument('--ppo', type=Path)
    ap.add_argument('--engine', type=Path)
    ap.add_argument('--engine-lib', type=Path)
    ap.add_argument('--engine-assets', type=Path)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--seed', type=int, default=2701)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if any(getattr(args, key) is None for key in ('il', 'ppo', 'engine', 'pool', 'out')):
        ap.error('--il, --ppo, --engine, --pool and --out are required')
    shard, shards = map(int, args.shard.split('/'))
    if args.games < 2 or args.games % 2 or not 0 <= shard < shards:
        ap.error('require an even game count and shard i/n with 0 <= i < n')
    sys.path.insert(0, str(args.source.resolve()))
    from cr import policy
    from cr.engine import Engine, Game

    torch.set_num_threads(1)
    il, vocab, _ = policy.load(args.il, args.device)
    ppo, ppo_vocab, _ = policy.load(args.ppo, args.device)
    if vocab.index != ppo_vocab.index:
        raise ValueError('IL and PPO must use identical vocabularies')
    arms = dict(ppo=ppo, il_ppo_tile=Hybrid(il, ppo).eval(), ppo_il_tile=Hybrid(ppo, il).eval())
    for model in (il, *arms.values()):
        model.temperature, model.mix, model.overlap = args.temperature, 0.0, True
    pool = json.loads(args.pool.read_text())
    indices = np.random.default_rng(args.seed).integers(len(pool), size=args.games // 2)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                sha256={key: digest(getattr(args, key)) for key in ('il', 'ppo', 'pool')},
                script_sha256=digest(Path(__file__)), arms=list(arms), overlap=True, ability_explore=0.0,
                random_streams_seeded=['python_command_delay', 'numpy', 'torch', 'native_match'])
    (args.out / f'metadata-s{shard}.json').write_text(json.dumps(meta, indent=2))
    engine = Engine(args.engine, lib=args.engine_lib, assets=args.engine_assets)
    for name, model in arms.items():
        path = args.out / f'{name}-s{shard}.jsonl'
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done.add((r['pair_index'], r['a_owner']))
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
                    agents = [policy.Agent(model if o == owner else il, o, args.device) for o in (0, 1)]
                    played, refused = [0, 0], [0, 0]
                    while not game.over:
                        views = game.views()
                        submitted, transition = game.step({a.owner: a.act(views[a.owner]) for a in agents})
                        for action, ok in zip(submitted, transition.executed):
                            refused[action.owner] += not ok
                        for play in engine.played:
                            played[play['owner']] += 1
                    state = game.state
                    if not state.finalized:
                        raise RuntimeError('battle did not finalize')
                    score = 0.5 if state.result not in (0, 1) else float(state.result == owner)
                    row = dict(arm=name, pair_index=i, pool_index=int(indices[i]), seed=args.seed + i,
                               a_owner=owner, score=score, crowns_a=state.crowns[owner],
                               crowns_b=state.crowns[1 - owner], tick=state.tick,
                               plays_a=played[owner], plays_b=played[1 - owner],
                               refused_a=refused[owner], refused_b=refused[1 - owner],
                               seconds=round(time.perf_counter() - started, 3))
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    print(json.dumps(row), flush=True)
    print('completed', args.shard, flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
