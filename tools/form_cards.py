"""Evolution and hero card ids -> their base card (31 Oct 2026). A unit on the board reports the card it came from, and
an evolved or hero unit reports its FORM card (Knight_EV1 = 13000xxx, IceWizard_hero, ...), which replay_cards.json does
not list: card_stats.py had no row for any of them (every evolved / hero unit read zero stats).

    python tools/form_cards.py ENGINE_IDS.json [ENGINE_IDS.json ...] assets/replay_cards.json assets/form_cards.json

ENGINE_IDS.json: every row of the engine's loaded tables as {'tables': {slot: [[data id, name], ...]}} (a dump of the
tables object, e.g. clapha pc-runner job 0126 for 160635001 and 160402020-d22). Slots 0x68 (evolution cards) and 0x658
(hero cards) are read; a form maps to the card whose name (or alias, or engine card row) is its name without
'_EV<n>' / '_hero'. Also 'cards': the engine's card rows (slots 0xd0 / 0xd8 / 0xe0) that replay_cards.json lacks, by their
data name -- e.g. Spirit Empress's board forms MergeMaiden_Normal / _Mounted (26000104 / 26000105, seen on 3% of pro
battles' boards). Exits non-zero if any form row finds no base card."""
import json
import re
import sys
from pathlib import Path

*dumps, CARDS, OUT = map(Path, sys.argv[1:])
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())  # noqa: E731
cards = json.loads(CARDS.read_text())['cards']
by_name = {}
for cid, c in cards.items():
    for n in [c['name'], *c.get('aliases', [])]:
        by_name.setdefault(norm(n), int(cid))
forms, missing, extra = {}, [], {}
for dump in dumps:
    tables = json.loads(dump.read_text())['tables']
    for slot in ('0xd0', '0xd8', '0xe0'):            # the engine's own card names, e.g. 'MinionGiant'
        for i, n in tables.get(slot, []):
            if str(i) in cards:
                by_name.setdefault(norm(n), i)
            else:
                extra[str(i)] = n
    for slot, suffix in (('0x68', r'_EV\d+$'), ('0x658', r'_hero$')):
        for i, n in tables.get(slot, []):
            base = by_name.get(norm(re.sub(suffix, '', n, flags=re.I)))
            if base is None:
                missing.append((i, n))
            else:
                forms[str(i)] = {'name': n, 'base': base}
OUT.write_text(json.dumps({'note': 'evolution / hero card id -> base card id (tools/form_cards.py)',
                           'forms': dict(sorted(forms.items(), key=lambda kv: int(kv[0]))),
                           'cards': dict(sorted(extra.items(), key=lambda kv: int(kv[0])))}, indent=1) + '\n')
print(len(forms), 'forms,', len(extra), 'other card rows;', len(missing), 'without a base card:', missing[:20])
sys.exit(1 if missing else 0)
