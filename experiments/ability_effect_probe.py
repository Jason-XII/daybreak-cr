"""Does a press change the native game? (28 Oct) For each pool hero / champion: the same seeded scene twice, a passive
opponent, the hero deployed by owner 0's bridge; at its first ready decision one run presses, the other waits. Diffs
at +100/+200/+400 ticks: own elixir, enemy towers' hp, own and enemy unit hp, entity counts. Run inside Slurm."""
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
    target = 14 if owner == 0 else 17  # by the bridge, left lane
    k = np.argmin((cols - 3) ** 2 + (rows - target) ** 2)
    return int(cols[k]) * 1000 + 500, int(rows[k]) * 1000 + 500


def snapshot(sim, owner):
    s = sim.state
    ents = s.entities
    towers = sum(max(e.hp or 0, 0) for e in ents if e.card == -1 and e.entity_type == 'building' and e.owner != owner)
    units = [sum(max(e.hp or 0, 0) for e in ents if e.card != -1 and e.owner == o) for o in (owner, 1 - owner)]
    return dict(tick=s.tick, elixir=round(s.players[owner].elixir / 10000, 3), enemy_towers=towers,
                own_units_hp=units[0], enemy_units_hp=units[1], entities=len(ents))


def scene(sim, pair, target, owner, press):
    sim.reset(match=engine.make_match(copy.deepcopy(pair), 281028))
    sim.overlap = [True, True]
    r = sim.env.reader
    deployed = False
    for _ in range(400):
        me = sim.state.players[owner]
        ready = next((i for i, a in enumerate(me.abilities) if a.ready), None)
        if deployed and ready is not None and not sim.pending:
            break
        actions = []
        if not deployed and not sim.pending:
            ctypes.c_int32.from_address(r.player(r.logic(sim.env.battle), owner) + r.f['player.elixir_raw']).value = 100000
            h = next((h for h in me.hand if h.card == target), None)
            if h is None:  # cycle a cheap non-hero card until the target is in hand
                h = min((h for h in me.hand if h.card != C.MIRROR and C.RARITY.get(str(h.card)) != 'Champion'
                         and not any(c['d'] == h.card and c.get('el') == 2 for c in pair['decks'][owner])),
                        key=lambda h: h.cost)
                where = (9500, 2500 if owner == 0 else 29500)  # behind the king tower: out of the way
                if not np.asarray(C.legal_rows(h.card, owner))[where[1] // 1000, where[0] // 1000]:
                    where = spot(h.card, owner)
            else:
                where = spot(h.card, owner)
                deployed = True
            actions = [engine.Play(owner, h.slot, *where)]
        sim.step(actions, 5, delay=22)
    else:
        return None
    ability = sim.state.players[owner].abilities[ready]
    out = dict(ability=ability.name, cost=ability.cost, at=snapshot(sim, owner))
    t = sim.step([engine.Play(owner, engine.ABILITY, ready, 0)] if press else [], 5, delay=22)
    if press:
        out['accepted'] = t.executed[0]
    start = sim.state.tick
    for mark in (100, 200, 400):
        while sim.state.tick < start + mark and not sim.state.finalized:
            sim.step([], 5)
        out[f'+{mark}'] = snapshot(sim, owner)
    out['stats'] = dict(sim.ability_stats[owner])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('engine', 'lib', 'assets', 'pool', 'out'):
        p.add_argument('--' + k, required=True)
    args = p.parse_args()
    pool = json.loads(Path(args.pool).read_text())
    targets = {}
    for pair in pool:
        for card in pair['decks'][0]:
            if card.get('el') == 2 or C.RARITY.get(str(card['d'])) == 'Champion':
                targets.setdefault(card['d'], pair)
    sim = engine.Engine(args.engine, lib=args.lib, assets=args.assets)
    rows = []
    for target, pair in sorted(targets.items()):
        runs = {press: scene(sim, pair, target, 0, press) for press in (True, False)}
        again = scene(sim, pair, target, 0, False)  # determinism control: two waits must match
        if None in runs.values():
            print(json.dumps(dict(card=target, skipped=True)), flush=True)
            continue
        on, off = runs[True], runs[False]
        row = dict(card=target, ability=on['ability'], cost=on['cost'], accepted=on['accepted'],
                   confirmed=on['stats'].get('confirmed', 0), same_start=on['at'] == off['at'], deterministic=again == off,
                   **{m: {k: on[m][k] - off[m][k] for k in on[m] if k != 'tick'} for m in ('+100', '+200', '+400')})
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1) + '\n')
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
