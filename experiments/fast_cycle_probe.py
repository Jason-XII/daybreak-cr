"""Native probe (28 Oct): can a player tap cards on consecutive decisions? Elixir pinned at 10 in a pool deck with
cheap cards; every decision taps the cheapest card the engine's mask allows. Logs per decision: tick, playable mask,
accepted, cards pending, elixir; and the tick each card lands. Run inside a Slurm allocation."""
import argparse
import copy
import ctypes
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cr import engine, observation as C


def spot(card, owner):
    rows, cols = np.nonzero(np.asarray(C.legal_rows(card, owner)))
    k = np.argmin((cols - 9) ** 2 + (rows - (5 if owner == 0 else 26)) ** 2)
    return int(cols[k]) * 1000 + 500, int(rows[k]) * 1000 + 500


def pin(sim, owner, elixir):
    r = sim.env.reader
    player = r.player(r.logic(sim.env.battle), owner)
    ctypes.c_int32.from_address(player + r.f['player.elixir_raw']).value = int(elixir * 10000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('engine', 'lib', 'assets', 'pool'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--overlap', type=int, default=1)
    args = p.parse_args()
    pool = json.loads(Path(args.pool).read_text())
    owner = 0
    pair = min(pool, key=lambda q: sum(sorted(C.COSTS.get(c['d'], 3) for c in q['decks'][owner])[:4]))
    print('deck', [(c['d'], C.COSTS.get(c['d'], 3)) for c in pair['decks'][owner]], flush=True)
    sim = engine.Engine(args.engine, lib=args.lib, assets=args.assets)
    sim.reset(match=engine.make_match(copy.deepcopy(pair), 2810))
    sim.overlap = [bool(args.overlap)] * 2
    for _ in range(20):
        sim.step([], 5)
    pin(sim, owner, 10)
    sim.env.step(1)
    sim.state = sim._capture()
    for decision in range(14):
        me = sim.state.players[owner]
        mask = sim.playable(owner)
        options = [h for h in me.hand if mask[h.slot] and h.card != C.MIRROR]
        h = min(options, key=lambda h: h.cost) if options else None
        actions = [engine.Play(owner, h.slot, *spot(h.card, owner))] if h else []
        pending = [(q['card'], q['due'], q['held']) for q in sim.pending if q['owner'] == owner]
        t = sim.step(actions, 5, delay=engine.PLAY_DELAY)
        print(json.dumps(dict(tick=me.tick if hasattr(me, 'tick') else sim.state.tick - 5, elixir=me.elixir / 10000,
                              mask=mask.tolist(), tapped=h and h.card, cost=h and h.cost, accepted=t.executed,
                              pending_before=pending, landed=[(q['card'], q['tick']) for q in sim.played])), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
