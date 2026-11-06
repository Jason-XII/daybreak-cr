import hashlib
import json
from pathlib import Path

import numpy as np

from cr import observation as C
from cr.inputs_v2.v2_contract import numeric_frame, pending_inputs, player_frame, read_episode
from cr.inputs_v2.v2_numeric import NumericLayout


def load_layout(root, manifest, settings):
    encoded = (Path(root) / 'v2-layout.json').read_bytes()
    layout = json.loads(encoded)
    if hashlib.sha256(encoded).hexdigest() != settings['v2_layout_sha256']:
        raise ValueError('V2 numeric layout hash mismatch')
    if manifest.get('input_profile') != 'inputs-v2-history31-own2':
        raise ValueError('Expected verified V2 history training manifest')
    if (layout.get('scope') != 'full_verified_dataset' or
            layout.get('canonicalization') != 'known-coordinates-own-view-v1' or
            layout.get('encoding') != 'named-signed-log-valid-v1'):
        raise ValueError('Only a complete canonical verified layout can be used for training')
    if settings['inputs'] != 2 or settings['own_flight_slots'] != 2 or settings['ability_labels']:
        raise ValueError('V2 requires inputs2, own2 and frozen labels without relabeling')
    for name, key in (('globals', 'v2_global_dim'), ('entities', 'v2_entity_dim')):
        if layout[name]['width'] != settings[key] or settings[key] != 2 * len(layout[name]['paths']):
            raise ValueError('V2 layout dimensions mismatch')
    sources = {row['manifest_sha256'] for row in layout['source_shards']}
    if sources != {row['manifest_sha256'] for row in manifest.get('lineage', [])}:
        raise ValueError('Layout and training manifest lineage differ')
    return layout


def validate_clock_and_arrays(data, frames):
    ticks = np.asarray([frame['game_tick'] for frame in frames], np.int64)
    if not len(ticks) or np.any(np.diff(ticks) <= 0):
        raise ValueError('V2 decision clock must be strictly increasing')
    if np.any(ticks[:-1] % 5):
        raise ValueError('Nonterminal decision is outside five-tick sampling')
    offsets = data['offsets']
    if offsets.shape != (len(ticks)+1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError('Invalid frozen object offsets')
    required = ('what', 'which', 'tile', 'ok', 'which_ok', 'value', 'next_when', 'next_which', 'next_which_ok')
    for name in required:
        if 't_'+name not in data or data['t_'+name].shape != ticks.shape:
            raise ValueError('Frozen labels differ in decision count: '+name)
    if 't_tick' in data and not np.array_equal(data['t_tick'], ticks):
        raise ValueError('Explicit label clock differs from captured frame clock')
    for key in C.OBJECT_KEYS:
        if len(data['x_'+key]) != offsets[-1]:
            raise ValueError('Frozen object array length differs from offsets: '+key)
    return ticks


def update_form_and_carriers(data, frame, owner, index, entities, begin, end, vocab):
    player = next(p for p in frame['players'] if p['side'] == owner)
    slots = player['hand_deck_indices']
    if len(slots) != 4:
        raise ValueError('Expected four hand slots')
    for hand, slot in enumerate(slots):
        if slot < 0:
            continue
        if slot >= len(player['deck_card_ids']) or slot >= len(player['deck_form_flags']):
            raise ValueError('Hand slot exceeds captured deck')
        observed = data['x_hand_card'][index, hand]
        if observed > 1 and observed != vocab(player['deck_card_ids'][slot]):
            raise ValueError('V2 hand ordering differs from frozen base archive')
        form = player['deck_form_flags'][slot]
        data['x_hand_float'][index, hand, 4:6] = (form == 2, form == 1)
    buttons = {}
    for ability in player.get('abilities', []):
        slot = ability['controller_slot']
        if slot not in (1, 2):
            raise ValueError('Unsupported ability controller slot')
        for entity_id in ability['units']:
            if entity_id in buttons and buttons[entity_id] != slot:
                raise ValueError('Entity belongs to multiple ability controllers')
            buttons[entity_id] = slot
    column = C.OBJ_FLOAT_NAMES.index('ability_button')
    data['x_obj_float'][begin:end, column] = [buttons.get(e['category'], 0) / 2
        if e['side'] == owner else 0 for e in entities]


def encode_episode(episode, archive_path, owner, vocab, global_layout, entity_layout):
    archive_path = Path(archive_path)
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != episode['labels']['sha256']:
        raise ValueError('Frozen base archive hash mismatch')
    with np.load(archive_path, allow_pickle=False) as archive:
        prefix = f'p{owner}_'
        data = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
    frames = episode['frames']
    offsets = data['offsets']
    validate_clock_and_arrays(data, frames)
    auxiliary = np.asarray([row[0] for row in episode['ability'][str(owner)]], np.float32)
    if auxiliary.shape != (len(frames), 2, 17):
        raise ValueError('V2 ability dimension mismatch')
    if not np.array_equal(auxiliary[..., :4], data['x_ability'][..., :4]):
        raise ValueError('V2 base ability first-four mismatch')
    data['x_ability'] = auxiliary
    data['x_ability_card'] = np.asarray([[vocab(value) for value in row[1]]
                                       for row in episode['ability'][str(owner)]], np.int32)
    global_rows, entity_rows = [], []
    pending = {key: [] for key in ('own_flight_card', 'own_flight_float', 'opp_flight_card', 'opp_flight_float')}
    for index, world in enumerate(frames):
        frame = player_frame(world, owner)
        entities = sorted(frame['entities'], key=lambda e: (e['category'] not in C.TOWER_IDS, e['category']))
        begin, end = int(offsets[index]), int(offsets[index+1])
        if len(entities) != end-begin:
            raise ValueError('V2 entity coverage differs from base archive; full rematerialization required')
        for key, values in (('card', [vocab(e['card_id']) for e in entities]),
                            ('data', [vocab(e['data_id']) for e in entities]),
                            ('side', [1 if e['side'] == owner else 2 for e in entities])):
            if not np.array_equal(data['x_obj_'+key][begin:end], values):
                raise ValueError('V2 entity order/identity differs from frozen base: '+key)
        canonical = numeric_frame(frame, owner)
        canonical_entities = sorted(canonical.pop('entities'), key=lambda e: (e['category'] not in C.TOWER_IDS, e['category']))
        for entity in canonical_entities:
            encoded = np.zeros((2, entity_layout.width), np.float32)
            encoded[0] = entity_layout.encode(entity)
            entity_rows.append(encoded)
        update_form_and_carriers(data, frame, owner, index, entities, begin, end, vocab)
        global_rows.append(global_layout.encode(canonical))
        inputs = pending_inputs(world, owner, vocab)
        for key in pending:
            pending[key].append(inputs[key])
    data['x_v2_global'] = np.stack(global_rows)
    data['x_v2_entity'] = np.stack(entity_rows) if entity_rows else np.zeros((0, 2, entity_layout.width), np.float32)
    data.update({'x_'+key: np.stack(rows) for key, rows in pending.items()})
    return data


def read(path, owner, vocab, layouts, archive_path=None, expected_sha256=None):
    episode = read_episode(path, expected_sha256)
    return encode_episode(episode, archive_path or episode['labels']['source_file'], owner, vocab,
                          NumericLayout(layouts['globals']['paths']), NumericLayout(layouts['entities']['paths']))
