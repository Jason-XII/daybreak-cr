"""Checks of a converted dataset before training on it (31 Oct): each line is a fix that must hold in real archives.

    python tools/check_dataset.py DATASET [--limit 400]

FAIL lines must be fixed before a full conversion; the activity table (share of present objects with each input non-zero)
is the reference a live capture is compared with (play.py diagnostics).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr import observation as C  # noqa: E402
from cr import imitation  # noqa: E402


HERO_BASE = {int(unit): int(card) for card, unit in json.loads((C.ASSETS / 'replay_cards.json').read_text())['hero_units'].items()}


def base(card):
    """A hero unit's card id (203000xxx: what a carrier unit reports) -> its card (what the hand shows). The table, not the
    number: hero Tombstone 27000009 is 203000088, 26000102's hero is 203000076."""
    return HERO_BASE.get(card, 26000000 + card % 1000000 if card // 1000000 == 203 else card)


def check(root, limit):
    files = sorted(Path(root).glob('*.npz'))[:limit]
    ids = {i + 2: v for i, v in enumerate(json.loads((Path(root) / 'manifest.json').read_text())['schema']['vocab'])}
    n = Counter()
    active, rows = np.zeros(C.OBJ_FLOATS), 0
    hero = C.HAND_FLOAT_NAMES.index('hero_form')
    button = C.OBJ_FLOAT_NAMES.index('ability_button')
    for f in files:
        meta = json.loads(np.load(f)['_metadata'].tobytes())
        if not meta.get('kept'):
            n['rejected'] += 1
            continue
        n['games'] += 1
        n['missing_cost_games'] += bool(meta.get('missing_costs'))
        n['unknown_id_games'] += bool(meta.get('unknown_ids'))
        for e in meta['entries']:
            n['entries'] += 1
            n['no_battle_time'] += e.get('battle_time') is None
            n['no_expert_flag'] += e.get('expert') is None
            x = imitation.read(f, e['owner'])
            what, ok, ab = x['t_what'], x['t_ok'], x['x_ability']
            n['decisions'] += len(what)
            n['hand_unknown'] += int((x['x_hand_card'] == 1).any(-1).sum())
            n['queue_unknown'] += int((x['x_cycle_card'] == 1).any(-1).sum())
            for k in (0, 1):
                presses = what == C.ABILITY1 + k
                n[f'press{k + 1}'] += int(presses.sum())
                n[f'press{k + 1}_dropped'] += int((presses & ~ok).sum())
                n[f'pending{k + 1}'] += int((ab[:, k, 4] > 0).sum())
            n['two_buttons'] += bool((ab[:, 1, 0] > 0).any())
            heroes = {ids.get(c) for c in x['x_hand_card'][x['x_hand_float'][..., hero] > 0].tolist()}
            carried = {base(ids.get(c, 0)) for c in x['x_ability_card'][x['x_ability_card'] > 1].tolist()}
            n['button_card_not_a_hero_form'] += len(carried - heroes)  # deck-slot form flags misaligned, or a Clone
            fl = x['x_obj_float']
            active += (np.abs(fl) > 1e-9).sum(0)
            rows += len(fl)
            n['carrier_rows'] += int((fl[:, button] > 0).sum())
    d = max(n['decisions'], 1)

    def line(ok, text):
        print(('ok    ' if ok else 'FAIL  ') + text)
    print(f"{n['games']} games kept, {n['rejected']} rejected, {n['entries']} views, {n['decisions']} decisions")
    line(n['hand_unknown'] == 0 and n['queue_unknown'] == 0,
         f"own hand / queue with '?': {n['hand_unknown']} / {n['queue_unknown']} decisions (live never has them)")
    line(n['press2'] > 0 or n['two_buttons'] == 0, f"presses labelled button 1 / 2: {n['press1']} / {n['press2']} "
         f"({n['two_buttons']} views have two buttons); dropped as unknown button: {n['press1_dropped']} / {n['press2_dropped']}")
    line(n['pending1'] + n['pending2'] > 0 or n['press1'] + n['press2'] == 0,
         f"decisions with our press in flight (button 1 / 2): {n['pending1']} / {n['pending2']}")
    line(n['button_card_not_a_hero_form'] == 0,
         f"button heroes not among the hand's hero-form cards: {n['button_card_not_a_hero_form']}")
    line(n['carrier_rows'] > 0 or n['press1'] == 0, f"object rows flagged as an ability carrier: {n['carrier_rows']}")
    line(n['missing_cost_games'] == 0, f"games with a card missing from card_costs.json: {n['missing_cost_games']}")
    line(n['unknown_id_games'] == 0, f"games with ids the vocabulary lacks: {n['unknown_id_games']}")
    line(n['no_battle_time'] == 0 and n['no_expert_flag'] == 0,
         f"views without battle_time / expert flag: {n['no_battle_time']} / {n['no_expert_flag']}")
    print(f"press rate: {100 * (n['press1'] + n['press2']) / d:.3f}% of decisions")
    print('object input activity (% of object rows non-zero):')
    for name, a in zip(C.OBJ_FLOAT_NAMES, active):
        print(f'  {name:<20}{100 * a / max(rows, 1):8.2f}%')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset', type=Path)
    p.add_argument('--limit', type=int, default=400)
    a = p.parse_args()
    check(a.dataset, a.limit)
