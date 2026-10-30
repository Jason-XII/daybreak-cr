"""Is the placement search's gain real against human-like play? (26 Oct, the user's agreement: before more placement
work, check the ~20-point search gain, measured only against our own bot.) Paired-seat games: A against B, each deck pair
played with A in both seats. With --search, A's card plays go to the best-scoring of its top --k tiles (and the one it
sampled): each tried in a forked copy of the game, played --horizon ticks with no further plays, scored by tower damage
(crowns + tower hp fractions, theirs minus ours) + --material x (our units' hp - theirs, in princess towers). Opponent
plays A can't see yet are taken out of the copies; the opponent's hand never enters (nobody plays in a copy).
Search: cr/search.py. Linux, inside a Slurm allocation.

    python experiments/search_arena.py A.pt B.pt --engine ~/clapha-engine --pool deck_pool.json --games 400 \\
        --out runs/x.jsonl [--search] [--shard 0/4]

Resumable: games already in --out are skipped. The same --seed gives the same deck pairs and game seeds in every run.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr import policy  # noqa: E402
from cr import search  # noqa: E402
from cr.engine import Engine, Game, tower_hp  # noqa: E402


class Searcher(policy.Agent):
    """policy.Agent; its card plays go to the best-scoring candidate tile (same card, same timing)."""

    def __init__(self, model, owner, device, engine, args):
        super().__init__(model, owner, device)
        self.engine, self.args = engine, args
        self.stats = dict(searches=0, changed=0, gain=0.0, failed=0, seconds=0.0)

    @torch.no_grad()
    def act(self, inputs, start):
        x = policy.tensors({k: v[None, None] for k, v in policy.pack(inputs).items()}, self.device)
        out = self.model(x, self.lstm)
        self.lstm = out['state']
        action, _ = policy.sample(self.model, out, x)
        choice = policy.decode(int(action[0]), self.owner)
        if choice[0] != 'card':
            return choice
        t0 = time.perf_counter()
        slot = choice[1]
        f, fx = policy.flat(out, x)
        tiles = policy.tile_logp(self.model, f, fx, torch.tensor([slot], device=self.device))[0].cpu()
        top = [int(c) for c in torch.topk(tiles, min(self.args.k, int(torch.isfinite(tiles).sum()))).indices]
        sampled = (int(action[0]) - policy.CARD) % policy.CELLS
        cells = list(dict.fromkeys([sampled] + top))
        plays = search.plays(self.owner, slot, cells)
        scores = search.search(self.engine, plays, self.owner, start, self.args.horizon, self.args.material)
        self.stats['searches'] += 1
        self.stats['seconds'] += time.perf_counter() - t0
        if any(s is None for s in scores):
            self.stats['failed'] += 1
            return choice
        best = int(np.argmax(scores))
        if best != 0 and scores[best] > scores[0] + 1e-6:
            self.stats['changed'] += 1
            self.stats['gain'] += scores[best] - scores[0]
            p = plays[best]
            return ('card', p.slot, p.x, p.y)
        return choice


def evaluate(engine, models, vocabs, pair, seed, a_owner, args):
    engine.overlap = [True, True]
    game = Game(engine, pair, seed, {a_owner: vocabs['a'], 1 - a_owner: vocabs['b']})
    start = tower_hp(game.state)
    agents = {a_owner: Searcher(models['a'], a_owner, args.device, engine, args) if args.search
              else policy.Agent(models['a'], a_owner, args.device),
              1 - a_owner: policy.Agent(models['b'], 1 - a_owner, args.device)}
    while not game.over:
        rows = game.views()
        game.step({o: a.act(rows[o], start) if isinstance(a, Searcher) else a.act(rows[o])
                   for o, a in agents.items()})
    state = game.state
    if not state.finalized:
        raise RuntimeError('native battle did not finalize before tick 7200')
    winner = None if state.result not in (0, 1) else 'A' if state.result == a_owner else 'B'
    stats = agents[a_owner].stats if args.search else {}
    return dict(winner=winner, a_owner=a_owner, crowns_a=state.crowns[a_owner], crowns_b=state.crowns[1 - a_owner],
                tick=state.tick, **stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a', type=Path)
    ap.add_argument('b', type=Path)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--engine-lib', type=Path, help="an engine build's folder (default: the checkout's)")
    ap.add_argument('--engine-assets', type=Path, help='its game data (default: the checkout\'s)')
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--search', action='store_true', help="A searches its tiles")
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=160)
    ap.add_argument('--material', type=float, default=0.3)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.05, help='both models, card and tile')
    ap.add_argument('--shard', default='0/1', help='i/n: this process plays deck pairs i, i+n, ...')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    i, n = map(int, args.shard.split('/'))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed * 1000 + i)
    pool = json.loads(args.pool.read_text())
    pairs = np.random.default_rng(args.seed).integers(len(pool), size=args.games // 2)
    models, vocabs = {}, {}
    for side in ('a', 'b'):  # each with its own vocabulary (IL r6's has cards v40's lacks)
        models[side], vocabs[side], _ = policy.load(getattr(args, side), args.device)
        models[side].temperature, models[side].overlap = args.temperature, True
    engine = Engine(args.engine, lib=args.engine_lib, assets=args.engine_assets)
    done = set()
    if args.out.exists():
        done = {json.loads(line)['game'] for line in args.out.read_text().splitlines() if line.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('a') as stream:
        for pair_index in range(i, len(pairs), n):
            for seat in (0, 1):
                g = 2 * pair_index + seat
                if g in done:
                    continue
                result = evaluate(engine, models, vocabs, pool[pairs[pair_index]], args.seed + pair_index, seat, args)
                stream.write(json.dumps(dict(game=g, search=args.search, **result)) + '\n')
                stream.flush()


if __name__ == '__main__':
    main()
