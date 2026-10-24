"""ClashAlpha/RoyaleAPI-Replays (private Hugging Face dataset, royaleapi-downloader schema 5) -> convert.py's JSON-lines input.

    python tools/hf_ingest.py OUT --parts 0-773 --token-file ~/.hf_token      (needs pyarrow)

Per part: downloads replays/part-XXXXXX.parquet (checked against the dataset's SHA256SUMS), and writes
  OUT/payloads/part-XXXXXX.jsonl   {tag, battle_time, mode, payload}  -- the battles to convert (royaleapi-battle-actions.v1)
  OUT/ingest/part-XXXXXX.jsonl     {tag, battle_time, mode, status, reason} -- every battle, kept or left out and why
Left out (docs/CONVERSION.md): modes other than Ranked, Ladder, Grand Challenge; battles before July 2026 (no game data of their
time); the Tombstone hero on 2026-09-03/04 (a bug made it free); a play with no position. Resumable: a part already ingested is
skipped. Then: python tools/assign_by_date.py OUT/assignment.jsonl OUT/payloads ; python convert.py ... (docs/CONVERSION.md).
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assign_by_date import data_at  # noqa: E402

REPO = 'https://huggingface.co/datasets/ClashAlpha/RoyaleAPI-Replays/resolve/main/'
KEEP_MODES = {('Ranked', 'pathOfLegend'), ('Ladder', 'trail'), ('Ladder', 'PvP'), ('Grand Challenge', 'trail')}
CHAMPIONS = {'archer-queen', 'golden-knight', 'skeleton-king', 'mighty-miner', 'monk', 'little-prince', 'boss-bandit',
             'goblinstein'}


def get(path, dest, token):
    req = urllib.request.Request(REPO + path, headers={'Authorization': f'Bearer {token}'})
    tmp = Path(str(dest) + '.tmp')
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)


def coordinates(x, y, data_i):
    """RoyaleAPI's team-normalised x / y -> world units (data-i 0: (x, 32000 - y); 1: (18000 - x, y)), as the converter reads them."""
    nx, ny = (x, 32000 - y) if data_i == 0 else (18000 - x, y)
    gx, gy = nx // 1000, ny // 1000
    return {'inside_18x32_arena': 0 <= nx < 18000 and 0 <= ny < 32000, 'inversion_applied': True,
            'native_world_units': {'x': nx, 'y': ny}, 'grid_cell_floor': {'x': gx, 'y': gy},
            'subcell_offset_from_floor_center': {'x': round((nx - gx * 1000 - 500) / 1000, 3),
                                                 'y': round((ny - gy * 1000 - 500) / 1000, 3)}}


def payload(d):
    """One schema-5 battle -> a royaleapi-battle-actions.v1 payload (levels, tower troops and their levels, final tower HP,
    elixir stats, the real length; the side marker data_i is one per battle)."""
    marks = {p['data_i'] for p in d['card_plays'] + list(d.get('ability_plays') or ())}
    if len(marks) != 1:
        raise ValueError(f'side marker data_i not unique: {sorted(marks)}')
    data_i = marks.pop()
    decks = {'team': list(d['team_deck']), 'opponent': list(d['opponent_deck'])}
    plays = sorted([('card', p) for p in d['card_plays']] + [('ability', p) for p in d.get('ability_plays') or ()],
                   key=lambda e: (int(e[1]['time_raw']), int(e[1]['marker_index'])))
    events = []
    for k, (kind, p) in enumerate(plays):
        side, tick = p['side'], int(p['time_raw'])
        card = p.get('card_base') if kind == 'card' else '_invalid'
        source = {'data_i': data_i, 'data_s': 't' if side == 'team' else 'o', 'data_t': tick,
                  'data_x': p.get('x_raw'), 'data_y': p.get('y_raw'), 'data_c': card, 'timeline_card_key': card}
        if kind == 'ability':
            candidates = [c for c in decks[side] if c.endswith('-hero') or c in CHAMPIONS]
            events.append({'kind': 'activate_ability', 'side': side, 'replay_tick_20hz': tick, 'source_index': k,
                           'card_key': None, 'coordinates': None, 'ability_source_authoritative': False,
                           'ability_source_candidates': [c.removesuffix('-hero') for c in candidates],
                           'deck_card_key_candidates': [], 'source_fields': source, 'time_seconds': tick / 20})
            continue
        if p.get('x_raw') in (None, '') or p.get('y_raw') in (None, ''):
            raise ValueError('a play without a position')
        events.append({'kind': 'play_card', 'side': side, 'replay_tick_20hz': tick, 'source_index': k, 'card_key': card,
                       'coordinates': coordinates(int(p['x_raw']), int(p['y_raw']), data_i),
                       'deck_card_key_candidates': [c for c in decks[side] if c.split('-ev1')[0].split('-hero')[0] == card],
                       'form_at_play': 'unknown', 'source_fields': source, 'time_seconds': tick / 20})
    hp = d['final_tower_hp']
    rounds = d['rounds'][0]

    def player(side):
        r = rounds[side][0]
        level = r.get('tower_troop_level')
        h = hp['team' if side == 'team' else 'opponent']
        return {'tag': (d.get(f'{side}_tags') or [''])[0],
                'deck': [{'card_key': c, 'level': int(r['card_levels'].get(c) or 16)} for c in decks[side]],
                'tower_card': {'card_key': r['tower_troop'], 'level': int(level) if level else None},
                'final_tower_hitpoints': {'king': h['king'], 'princess_left': h['princess0'], 'princess_right': h['princess1']}}
    es = d.get('elixir_stats') or {}

    def number(side, key, kind):
        v = ((es.get(side) or {}).get(key) or {}).get('elixir')
        try:
            return kind(v)
        except (TypeError, ValueError):
            return None
    stats = {side: {**{k.lower(): [None, number(side, k, int)] for k in ('Total', 'Troop', 'Building', 'Spell')},
                    'leaked': number(side, 'Leaked', float)} for side in ('team', 'opponent')}
    tc, oc = int(d['team_crowns']), int(d['opponent_crowns'])
    return {'schema_version': 'royaleapi-battle-actions.v1', 'collection_schema_version': 'royaleapi-downloader-5',
            'source': {'site': 'RoyaleAPI', 'replay_tag': d['battle_tag'], 'anonymized': False},
            'battle': {'battle_type': d['battle_type'], 'game_mode': d.get('game_mode'),
                       'result': 'victory' if tc > oc else 'defeat' if tc < oc else 'draw',
                       'team': {'crowns': tc, 'players': [player('team')]},
                       'opponent': {'crowns': oc, 'players': [player('opponent')]}},
            'replay': {'duration': {'timeline_seconds': float(d.get('duration_seconds') or 330)}, 'aggregate_stats': stats},
            'events': events, 'battle_time': d['battle_time_utc']}


def classify(d):
    if (d.get('game_mode'), d.get('battle_type')) not in KEEP_MODES:
        return f'mode {d.get("game_mode")}/{d.get("battle_type")}'
    when = d.get('battle_time_utc') or ''
    if data_at(when) is None:
        return 'before July 2026 (no game data of that time)'
    if '2026-09-03' <= when[:10] <= '2026-09-04' and any(k.startswith('tombstone-hero')
                                                         for k in d.get('team_deck', []) + d.get('opponent_deck', [])):
        return 'Tombstone hero on 2026-09-03/04 (a bug made it free)'
    return None


def ingest_part(path, out):
    import pyarrow.parquet as pq
    kept, rows = [], []
    for raw in pq.read_table(path, columns=['payload_json']).column('payload_json').to_pylist():
        d = json.loads(raw)
        row = {'tag': d.get('battle_tag'), 'battle_time': d.get('battle_time_utc'), 'mode': d.get('game_mode')}
        reason = classify(d)
        if reason is None:
            try:
                kept.append(json.dumps({**row, 'payload': payload(d)}))
            except (KeyError, ValueError, TypeError) as error:
                reason = f'payload: {type(error).__name__}: {str(error)[:80]}'
        rows.append(json.dumps({**row, 'status': 'kept' if reason is None else 'left out', 'reason': reason}))
    (out / 'payloads' / (path.stem + '.jsonl')).write_text('\n'.join(kept) + ('\n' if kept else ''))
    (out / 'ingest' / (path.stem + '.jsonl')).write_text('\n'.join(rows) + '\n')
    return len(rows), len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path)
    ap.add_argument('--parts', default='0-773', help='e.g. 0-773 or 0,5,12')
    ap.add_argument('--token-file', type=Path, required=True)
    args = ap.parse_args()
    token = args.token_file.expanduser().read_text().strip()
    out = args.out.expanduser()
    for sub in ('parquet', 'payloads', 'ingest'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if not (out / 'SHA256SUMS').exists():
        get('SHA256SUMS', out / 'SHA256SUMS', token)
    sums = {line.split()[-1].lstrip('*').lstrip('./'): line.split()[0].lower()
            for line in (out / 'SHA256SUMS').read_text().splitlines() if len(line.split()) >= 2}
    parts = []
    for chunk in args.parts.split(','):
        a, _, b = chunk.partition('-')
        parts += range(int(a), int(b or a) + 1)
    for i in parts:
        name = f'part-{i:06d}'
        if (out / 'ingest' / f'{name}.jsonl').exists():
            continue
        dest = out / 'parquet' / f'{name}.parquet'
        if not dest.exists():
            get(f'replays/{name}.parquet', dest, token)
        want, got = sums.get(f'replays/{name}.parquet'), hashlib.sha256(dest.read_bytes()).hexdigest()
        if want and got != want:
            raise SystemExit(f'{name}: SHA-256 {got} is not the dataset\'s {want}')
        n, k = ingest_part(dest, out)
        print(f'{name}: {n} battles, {k} kept', flush=True)


if __name__ == '__main__':
    main()
