"""Arron's 20-tick delay (24 Oct): convert.py with every card play moved SHIFT ticks (negative: earlier), abilities left
at their replay ticks. If converted games run ~20 ticks late, exact replays peak near SHIFT = -20. Each game's failed
presses / cards and held / dropped plays go to OUT/shift.jsonl.

    cd SRC && python experiments/convert_shift.py SHIFT ...convert.py's arguments...
"""
import json
from pathlib import Path
import sys

SHIFT = int(sys.argv.pop(1))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402

prepare, play_replay = convert.prepare, convert.play_replay
out = Path(sys.argv[sys.argv.index('--out') + 1])


def shifted(payload):
    game = prepare(payload)
    for c in game['commands']:
        if c['kind'] == 'card':
            c['tick'] = max(c['tick'] + SHIFT, 2)
    game['commands'].sort(key=lambda c: c['tick'])  # stable: each player's cards keep their order
    return game


def logged(native, payload):
    result = play_replay(native, payload)
    kinds = [f[1] for f in result['failures']]
    with open(out / 'shift.jsonl', 'a') as f:
        f.write(json.dumps(dict(tag=payload['source']['replay_tag'], shift=SHIFT, hp_error=result['fidelity']['tower_hp_error'],
                                abilities=sum(c['kind'] == 'ability' for c in result['game']['commands']),
                                ability_failed=kinds.count('ability'), card_failed=kinds.count('card'),
                                held=sum(h[3] > 0 for h in result['holds']),
                                dropped=sum(h[4] == 'dropped' for h in result['holds']))) + '\n')
    return result


if __name__ == '__main__':
    convert.prepare, convert.play_replay = shifted, logged
    convert.main()
