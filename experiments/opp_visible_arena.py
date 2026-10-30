"""How much is seeing the opponent's plays before they land worth? (25 Oct, the user's question.) Paired-seat games, the same
model in both seats: A sees an opponent play only from --a-visible ticks before it lands (0: at landing, as a human sees it),
B from OPP_VISIBLE (13) as trained. Both may tap while a card is in flight (gen-h and later). Linux, inside a Slurm allocation.

    python experiments/opp_visible_arena.py M.pt --engine ~/clapha-engine --pool deck_pool.json --games 400 \
        --out runs/oppvis.jsonl [--a-visible 0] [--shard 0/4]

Resumable: games already in --out are skipped.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr import observation as C  # noqa: E402
from cr import policy  # noqa: E402
from cr.engine import Engine, Game  # noqa: E402

TRAINED = C.OPP_VISIBLE
_step = C.PlayerView.step


def blinded(owner, visible):
    """PlayerView.step with OPP_VISIBLE = `visible` for `owner`'s view only (PlayerView.step reads the module global)."""
    def step(self, *args, **kwargs):
        C.OPP_VISIBLE = visible if self.owner == owner else TRAINED
        try:
            return _step(self, *args, **kwargs)
        finally:
            C.OPP_VISIBLE = TRAINED
    return step


def evaluate(engine, model, vocab, pair, seed, a_owner, visible, device):
    C.PlayerView.step = blinded(a_owner, visible)
    engine.overlap = [True, True]
    game = Game(engine, pair, seed, {0: vocab, 1: vocab})
    agents = [policy.Agent(model, owner, device) for owner in (0, 1)]
    played, refused = [0, 0], [0, 0]
    while not game.over:
        rows = game.views()
        submitted, transition = game.step({a.owner: a.act(rows[a.owner]) for a in agents})
        for action, ok in zip(submitted, transition.executed):
            refused[action.owner] += not ok
        for play in engine.played:
            played[play['owner']] += 1
    state = game.state
    if not state.finalized:
        raise RuntimeError('native battle did not finalize before tick 7200')
    winner = None if state.result not in (0, 1) else 'A' if state.result == a_owner else 'B'
    return dict(winner=winner, a_owner=a_owner, crowns_a=state.crowns[a_owner], crowns_b=state.crowns[1 - a_owner],
                plays_a=played[a_owner], plays_b=played[1 - a_owner], refused_a=refused[a_owner],
                refused_b=refused[1 - a_owner], tick=state.tick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', type=Path)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--a-visible', type=int, default=0, help="ticks before landing A sees an opponent play (trained: 13)")
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--shard', default='0/1', help='i/n: this process plays deck pairs i, i+n, ...')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    i, n = map(int, args.shard.split('/'))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed * 1000 + i)
    pool = json.loads(args.pool.read_text())
    pairs = np.random.default_rng(args.seed).integers(len(pool), size=args.games // 2)
    model, vocab = policy.load(args.model, args.device)[:2]
    model.temperature, model.overlap = args.temperature, True
    engine = Engine(args.engine)
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
                result = evaluate(engine, model, vocab, pool[pairs[pair_index]], args.seed + pair_index, seat,
                                  args.a_visible, args.device)
                stream.write(json.dumps(dict(game=g, a_visible=args.a_visible, **result)) + '\n')
                stream.flush()


if __name__ == '__main__':
    main()
