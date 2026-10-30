"""How often does a model put two cards in flight in the engine, and how often does the overlap mask refuse a card it
wanted (28 Oct)? Model vs itself, overlap on, T 0.05, a few games. Per seat-game: plays, overlapped plays (tapped while
another own card was pending), and decisions where the unmasked policy picked PLAY but the mask blocked every card,
split by why (a card held for elixir, two in flight, unaffordable after reservations). Run inside Slurm."""
import argparse
import json
from pathlib import Path
import os
import random
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('source', 'model', 'engine', 'lib', 'assets', 'pool'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--games', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    sys.path.insert(0, args.source)
    from cr import policy
    from cr.engine import ABILITY, Engine, Game
    torch.set_num_threads(1)
    model, vocab, _ = policy.load(args.model, args.device)
    model.temperature, model.mix, model.overlap = 0.05, 0.0, True
    pool = json.loads(Path(args.pool).read_text())
    rng = random.Random(2810)
    engine = Engine(args.engine, lib=args.lib, assets=args.assets)
    total = {}
    for g in range(args.games):
        engine.overlap = [True, True]
        game = Game(engine, rng.choice(pool), rng.randrange(1 << 30), {0: vocab, 1: vocab})
        agents = [policy.Agent(model, o, args.device) for o in (0, 1)]
        for a in agents:
            a.prev = None
        c = {o: dict(plays=0, overlapped=0, wanted_blocked=0, held=0, two=0, budget=0, flight_decisions=0, p_play_sum=0.0,
                     p_play_kept_sum=0.0, p_on_flying_slot_sum=0.0) for o in (0, 1)}
        while not game.over:
            for a in agents:
                a.prev = a.lstm
            views = game.views()
            choices = {}
            for a in agents:
                x = views[a.owner]
                mine = [p for p in engine.pending if p['owner'] == a.owner and p['slot'] != ABILITY]
                choices[a.owner] = ch = a.act(x)
                if mine:  # P(play) the sampler turns into a wait: the card head's mass on cards the mask refuses
                    t = policy.tensors({k: v[None, None] for k, v in policy.pack(x).items()}, args.device)
                    with torch.no_grad():
                        out = model(t, a.prev)
                    which = out['which'].reshape(4).float().masked_fill(~t['legal'].reshape(4, -1).any(-1), float('-inf'))
                    ok = torch.as_tensor(x['live_playable'], device=which.device)
                    q = which.softmax(-1)
                    k = c[a.owner]
                    k['flight_decisions'] += 1
                    k['p_play_sum'] += a.p_what[policy.PLAY]
                    k['p_play_kept_sum'] += a.p_what[policy.PLAY] * float(q[ok].sum())
                    flying = {p['slot'] for p in mine}
                    k['p_on_flying_slot_sum'] += a.p_what[policy.PLAY] * float(sum(q[s] for s in flying))
                if ch[0] == 'card':
                    c[a.owner]['plays'] += 1
                    c[a.owner]['overlapped'] += bool(mine)
                # the model's own preference: P(play) without the engine mask, when the mask leaves no card
                if mine and not x['live_playable'].any() and a.p_what[policy.PLAY] > 0.5:
                    k = c[a.owner]
                    k['wanted_blocked'] += 1
                    k['held' if any(p['held'] for p in mine) else 'two' if len(mine) >= 2 else 'budget'] += 1
            game.step(choices)
        for o in (0, 1):
            for k, v in c[o].items():
                total[k] = total.get(k, 0) + v
        print(json.dumps(dict(game=g, **{f'{k}_{o}': v for o in (0, 1) for k, v in c[o].items()})), flush=True)
    print(json.dumps(dict(model=args.model, seat_games=2 * args.games,
                          **{k: round(v / (2 * args.games), 2) for k, v in total.items()})), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
