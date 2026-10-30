"""Does deciding every tick help? (25 Oct, the user's question.) Paired-seat games: A decides every tick, B every 5 ticks
as trained; same model in both seats by default. Linux, inside a Slurm allocation.

    python experiments/tick_arena.py A.pt B.pt --engine ~/clapha-engine --pool deck_pool.json --mode hazard \
        --games 400 --out runs/tick-hazard.jsonl [--shard 0/4]

A's --mode:
  naive   the policy as is, every tick: its LSTM steps every tick, and P(play) per tick is its per-decision P (~5x the plays
          it was trained to make, given the elixir).
  hazard  each tick is decided as if it were the 5-tick decision: the LSTM commits only every 5 ticks (the ticks in between
          start from the same state), and P(not wait) per tick = 1 - P(wait) ** (1/5): over 5 ticks it plays as often as
          it would once, but picks the tick.
  five    every 5 ticks through this loop (control: should be ~50%).
Both players' object velocities (dx, dy) stay measured over 5 ticks, as trained. Both may tap a card while one is in flight
(gen-h and later are trained so). Resumable: games already in --out are skipped.
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
from cr.engine import Engine, Game, PLAY_DELAY, plays  # noqa: E402

FIRST = 90  # Engine.reset steps to tick 90; decisions every 5 ticks from there


@torch.no_grad()
def act(model, lstm, inputs, owner, device, k=1):
    """One decision from LSTM state `lstm` -> (decode()'s action, the state after it). policy.sample() with, for k > 1,
    P(not wait) = 1 - P(wait) ** (1/k), split among play and the abilities as the model splits it."""
    x = policy.tensors({key: v[None, None] for key, v in policy.pack(inputs).items()}, device)
    out = model(x, lstm)
    state = out['state']
    out, x = policy.flat(out, x)
    what_lp = policy.what_logp(out['what'], x, 0.0, out['which'], model.overlap)
    if k > 1:
        p = what_lp.exp()
        wait = p[:, C.WAIT]
        go = 1 - wait
        p = p * ((1 - wait ** (1 / k)) / go.clamp(min=1e-12))[:, None]
        p[:, C.WAIT] = wait ** (1 / k)
        what_lp = p.clamp(min=1e-30).log().masked_fill(p == 0, float('-inf'))
    t = model.temperature
    which_lp = policy.sharpen(policy.which_logp(out['which'], x, model.overlap), t)
    what, slot = policy._gumbel_argmax(what_lp), policy._gumbel_argmax(which_lp)
    cell = policy._gumbel_argmax(policy.sharpen(policy.tile_logp(model, out, x, slot), t))
    action = int(policy.CARD + slot * C.CELLS + cell) if int(what) == C.PLAY else int(what)
    return policy.decode(action, owner), state


def evaluate(engine, model, vocab, pair, seed, a_owner, mode, device):
    engine.overlap = [True, True]
    game = Game(engine, pair, seed, {0: vocab, 1: vocab})
    views = game.inputs.views
    b = policy.Agent(model, 1 - a_owner, device)
    lstm = base = None
    seen = {}  # tick -> object positions then: each input's velocities over the last 5 ticks, as at 5-tick decisions
    played, refused = [0, 0], [0, 0]
    while not game.over:
        tick = game.state.tick
        boundary = (tick - FIRST) % 5 == 0
        for v in views.values():
            v.prev = seen.get(tick - 5, {})
        rows = game.views()
        seen[tick] = views[a_owner].prev
        seen.pop(tick - 6, None)
        choices = {}
        if mode == 'naive':
            choices[a_owner], lstm = act(model, lstm, rows[a_owner], a_owner, device)
        elif mode == 'hazard':
            if boundary:
                base = lstm
                choices[a_owner], lstm = act(model, base, rows[a_owner], a_owner, device, k=5)
            else:
                choices[a_owner], _ = act(model, base, rows[a_owner], a_owner, device, k=5)
        else:
            choices[a_owner], lstm = act(model, lstm, rows[a_owner], a_owner, device)
        if boundary:
            choices[b.owner] = b.act(rows[b.owner])
        submitted = plays(choices)
        transition = engine.step(submitted, 5 if mode == 'five' else 1, delay=PLAY_DELAY)
        game.state = transition.state
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
    ap.add_argument('a', type=Path)
    ap.add_argument('b', type=Path)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--mode', choices=('naive', 'hazard', 'five'), required=True)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--shard', default='0/1', help='i/n: this process plays deck pairs i, i+n, ...')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.a.resolve() != args.b.resolve():
        ap.error('A and B must be the same model (one network object serves both seats)')
    i, n = map(int, args.shard.split('/'))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed * 1000 + i)
    pool = json.loads(args.pool.read_text())
    pairs = np.random.default_rng(args.seed).integers(len(pool), size=args.games // 2)
    model, vocab = policy.load(args.a, args.device)[:2]
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
                result = evaluate(engine, model, vocab, pool[pairs[pair_index]], args.seed + pair_index, seat, args.mode,
                                  args.device)
                stream.write(json.dumps(dict(game=g, mode=args.mode, **result)) + '\n')
                stream.flush()


if __name__ == '__main__':
    main()
