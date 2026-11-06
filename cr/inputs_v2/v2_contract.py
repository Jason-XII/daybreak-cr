from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import pyarrow as pa

from cr import observation as C
from cr.inputs_v2.input_spec import IN, OUT, rule


def read_episode(path, expected_sha256=None):
    data = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('V2 episode hash mismatch')
    if len(data) < 16 or data[:8] != b'DBCRINP2':
        raise ValueError('Invalid v2 episode framing')
    size, = struct.unpack('<Q', data[8:16])
    if not 0 < size <= 512 * 1024 * 1024:
        raise ValueError('Invalid v2 decode bound')
    episode = json.loads(pa.Codec('zstd').decompress(data[16:], size).to_pybytes())
    if episode['schema'] != 'dbcr-inputs-v2-episode.v1' or episode['sampling_ticks'] != 5:
        raise ValueError('Unsupported episode schema or decision frequency')
    return episode


def sanitize_frame(world):
    def select(value, prefix=''):
        selected = {}
        for key, child in value.items():
            path = prefix+'.'+key if prefix else key
            match = rule(path)
            if match and match[0] == OUT:
                continue
            if isinstance(child, list) and child and isinstance(child[0], dict) and (not match or path == 'queue'):
                selected[key] = [select(item, path+'[]') for item in child]
            elif match and match[0] == IN:
                selected[key] = deepcopy(child)
            elif isinstance(child, dict):
                selected[key] = select(child, path)
            elif child == [] and path in ('players', 'entities'):
                selected[key] = []
            else:
                raise ValueError('Unclassified v2 input field: '+path)
        return selected
    return select(world)


def player_frame(world, owner):
    if owner not in (0, 1):
        raise ValueError('Invalid player side')
    frame = sanitize_frame(world)
    opponent = next(player for player in frame['players'] if player['side'] != owner)
    for key in ('deck_card_ids', 'deck_form_flags', 'deck_level_index', 'battle_deck',
                'evo_progress', 'hand_deck_indices', 'cycle_deck_indices', 'slot_words'):
        opponent[key] = []
    opponent['next_deck_index'] = -1
    opponent['player_raw'] = [None] * len(opponent['player_raw'])
    for ability in opponent.get('abilities', []):
        carriers = [entity for entity in frame['entities']
                    if entity['side'] != owner and entity['category'] in ability['units']]
        if not carriers:
            ability['character_id'] = 0
    frame['observer_side'] = owner
    return frame


def pending_inputs(frame, owner, vocab):
    tick = frame['game_tick']
    groups = {'own': [], 'opponent': []}
    for command in frame['queue']:
        if not 0 <= command['t2'] - tick <= 6 or command['t'] > tick:
            raise ValueError('Command is outside the v2 causal visibility window')
        if command['side'] not in (0, 1):
            raise ValueError('Invalid queued command side')
        if command['kind'] != 1 or command['t2'] <= tick:
            continue
        side = 'own' if command['side'] == owner else 'opponent'
        groups[side].append((command['t2'], dict(card_id=command['card'],
                                                x=command['a'], y=command['b'])))

    def encode(landing, command):
        x, y = C.rotate(command['x'], command['y'], owner)
        return [x / 18000, y / 32000, (landing-tick) / C.DEPLOY_DELAY,
                C.cost(command['card_id']) / 10]

    result = {}
    for side, prefix in (('own', 'own'), ('opponent', 'opp')):
        result[prefix+'_flight_card'], result[prefix+'_flight_float'] = C.pack_pending_flights(
            groups[side], vocab, encode, side)
    return result


def numeric_frame(frame, owner):
    """Canonicalize only fields with established coordinate/vector semantics."""
    result = deepcopy(frame)
    result['players'].sort(key=lambda player: player['side'] != owner)
    for player in result['players']:
        player['side'] = 1 if player['side'] == owner else 2
    for entity in result['entities']:
        entity['side'] = 1 if entity['side'] == owner else 2
        if owner == 1:
            for x, y in (('x', 'y'), ('tx', 'ty')):
                if x in entity and y in entity and entity[x] is not None and entity[y] is not None:
                    entity[x], entity[y] = C.rotate(entity[x], entity[y], owner)
            state = entity.get('us', {})
            for key in ('aim', 'start', 'surface'):
                value = state.get(key)
                if isinstance(value, list) and len(value) == 2 and all(v is not None for v in value):
                    if key != 'surface' or value != [-1, -1]:
                        state[key] = list(C.rotate(*value, owner))
            for key in ('dx', 'dy'):
                if state.get(key) is not None:
                    state[key] = -state[key]
            if 'facing' in state:
                state['facing'] = [-v if v is not None else None for v in state['facing']]
            # Waypoint can be an absolute node or a push vector; raw-word arrays remain opaque.
    for command in result.get('queue', []):
        command['side'] = 1 if command['side'] == owner else 2
        if owner == 1 and command['kind'] == 1:
            command['a'], command['b'] = C.rotate(command['a'], command['b'], owner)
    result['observer_side'] = 1
    return result
