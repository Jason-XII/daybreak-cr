"""Fail loudly when ids reach the model as UNKNOWN or without their facts (31 Oct 2026: for weeks hero Ice Wizard, evo
Goblin Barrel's goblin, Minion Giant and October evolutions were one unknown token, and every evolved / hero unit read
zero card stats; nothing said so).

    python tools/check_inputs.py SOURCE [SOURCE ...]

SOURCE is either
  - a conversion manifest.json: its 'unknown_ids' (convert.py counts every id the vocabulary lacked), or
  - an ids-seen file {'ids': {id: {'games': n, 'fields': {'cardId'|'dataGlobalId'|'buff': n}, 'names': {...}}}}, e.g. the
    ids on the boards of rendered real battles (clapha pc-runner job 0127): every id must be in assets/vocab.json, and
    every card id and unit data id must have a row in assets/card_stats.json (projectiles, areas and effects have none
    by design; effects need a property vector in assets/effect_vectors.json via effect_ids.json).
Prints every failure with its name and how many games it is in; exit 1 if any."""
import json
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def check(paths, assets=ASSETS):
    vocab = set(json.loads((assets / 'vocab.json').read_text()))
    stats = set(map(int, json.loads((assets / 'card_stats.json').read_text())['stats']))
    effects = {int(i): n for i, n in json.loads((assets / 'effect_ids.json').read_text())['effects']}
    vectors = set(json.loads((assets / 'effect_vectors.json').read_text())['effects'])
    fails = []
    for path in map(Path, paths):
        data = json.loads(path.read_text())
        if 'unknown_ids' in data:                       # a conversion manifest
            for i, row in data['unknown_ids'].items():
                fails.append((path.name, int(i), 'not in the vocabulary', '', row.get('games', '?')))
            continue
        for i, e in data['ids'].items():
            i, games = int(i), e.get('games', '?')
            names = e.get('names') or {}
            name = max(names, key=names.get) if names else ''
            kinds = {n.split('|')[-1] for n in names}
            if i not in vocab:
                fails.append((path.name, i, 'not in the vocabulary', name, games))
            fields = e.get('fields') or {}
            if (fields.get('cardId') or ('unit' in kinds and fields.get('dataGlobalId'))) and i not in stats:
                fails.append((path.name, i, 'no card stats', name, games))
            if fields.get('buff') and effects.get(i) not in vectors:
                fails.append((path.name, i, 'effect without a property vector', effects.get(i, name), games))
    return fails


if __name__ == '__main__':
    found = check(sys.argv[1:])
    for source, i, what, name, games in sorted(found, key=lambda f: -(f[4] if isinstance(f[4], int) else 0)):
        print(f'FAIL {source}: {i} {name}: {what} (in {games} games)')
    print(f'{len(found)} failures' if found else 'PASS: every id is in the vocabulary, with its facts')
    sys.exit(1 if found else 0)
