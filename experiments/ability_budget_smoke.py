"""Native d22 validation: deploy each pool hero/champion and test ability/card reservations."""
import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cr import engine, observation as C


def spot(card, owner):
    rows, cols = np.nonzero(np.asarray(C.legal_rows(card, owner)))
    if not len(rows): return None
    x, y = cols * 1000 + 500, rows * 1000 + 500
    target_y = 5500 if owner == 0 else 26500
    k = np.argmin((x - 9000) ** 2 + (y - target_y) ** 2)
    return int(x[k]), int(y[k])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True)
    p.add_argument('--lib', required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    pool = json.loads(Path(args.pool).read_text())
    targets = {}
    for pair in pool:
        for owner, deck in enumerate(pair['decks']):
            for card in deck:
                if card.get('el') == 2 or C.RARITY.get(str(card['d'])) == 'Champion':
                    targets.setdefault((card['d'], card.get('el', 0)), (pair, owner))
    sim = engine.Engine(args.engine, lib=args.lib, assets=args.assets)
    results = []
    skipped = []
    for (target, form), (pair, owner) in sorted(targets.items()):
        sim.reset(match=engine.make_match(copy.deepcopy(pair), 271027))
        sim.overlap = [True, True]
        deployed = False
        controller = None
        for _ in range(600):
            me = sim.state.players[owner]
            controller = next((i for i, a in enumerate(me.abilities) if a.ready), None)
            if deployed and controller is not None and not sim.pending:
                break
            actions = []
            if not deployed and not sim.pending:
                target_hand = next((h for h in me.hand if h.card == target), None)
                if target_hand is not None and me.elixir >= 99900:
                    h = target_hand
                elif target_hand is None:
                    legal = [h for h in me.hand if h.card != C.MIRROR and h.card != target
                             and C.RARITY.get(str(h.card)) != 'Champion'
                             and not any(c['d'] == h.card and c.get('el') == 2 for c in pair['decks'][owner])
                             and spot(h.card, owner) is not None and h.cost * 10000 <= me.elixir]
                    h = min(legal, key=lambda h: h.cost) if legal else None
                else:
                    h = None
                if h is not None:
                    where = spot(h.card, owner)
                    if where:
                        actions = [engine.Play(owner, h.slot, *where)]
                        deployed = h.card == target
            transition = sim.step(actions, 5, delay=22)
            if actions: assert transition.executed == (True,)
        else:
            row = dict(card=target, form=form, owner=owner, reason='no ready carrier in passive-opponent fixture',
                       deployed=deployed, finalized=sim.state.finalized,
                       abilities=[dict(name=a.name, button=a.button, units=a.units) for a in me.abilities])
            skipped.append(row)
            print('SKIP', json.dumps(row), flush=True)
            Path(args.out).with_name('native-skipped.json').write_text(json.dumps(skipped, indent=2)+'\n')
            continue
        me = sim.state.players[owner]
        ability = me.abilities[controller]
        card = min((h for h in me.hand if h.card != C.MIRROR and h.cost >= 2
                    and spot(h.card, owner) is not None), key=lambda h: h.cost)
        # Fixture only: pin the native elixir bar so a cheap ability and card
        # are each affordable, while their combined payments are not.
        deficit = C.regen(sim.state.tick, sim.state.tick + 5 + max(engine.PAYMENT_DELAYS)) / 10000 + .25
        balance = ability.cost + card.cost - deficit
        assert ability.cost <= balance < 10, (ability, card, balance)
        reader = sim.env.reader
        player = reader.player(reader.logic(sim.env.battle), owner)
        ctypes.c_int32.from_address(player + reader.f['player.elixir_raw']).value = int(balance * 10000)
        sim.env.step(1)
        sim.state = sim._capture()
        before = sim.state.tick
        press = sim.step([engine.Play(owner, engine.ABILITY, controller, 0)], 5, delay=22)
        assert press.executed == (True,), (target, ability)
        assert not sim.playable(owner)[card.slot], (target, ability, card, balance)
        rejected = sim.step([engine.Play(owner, card.slot, *spot(card.card, owner))], 5, delay=22)
        assert rejected.executed == (False,)
        for _ in range(20):
            sim.step([], 5)
            if sim.ability_stats[owner]['confirmed']: break
        assert sim.ability_stats[owner]['confirmed'] == 1, (ability, dict(sim.ability_stats[owner]))
        for _ in range(150):
            if sim.playable(owner)[card.slot]: break
            sim.step([], 5)
        assert sim.playable(owner)[card.slot]
        tap = sim.step([engine.Play(owner, card.slot, *spot(card.card, owner))], 5, delay=22)
        assert tap.executed == (True,)
        for _ in range(12): sim.step([], 5)
        assert sim.expired == 0, (target, sim.expired)
        assert not sim.pending
        row = dict(card=target, form=form, owner=owner, ability=ability.name, cost=ability.cost,
                   tick=before, unsafe_card_blocked=True, funded_card_completed=True,
                   ability_stats=dict(sim.ability_stats[owner]), expired_cards=sim.expired)
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2)+'\n')
    assert len(results) >= 15
    print('PASS', len(results), 'native hero/champion cases', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # the native library cannot be safely torn down by Python
