"""Versioned, compressed world traces independent of any model input schema."""
import hashlib
import json
import os
from pathlib import Path
import struct

import pyarrow as pa

MAGIC = b'DBCRRAW1'
SCHEMA = 'daybreak-cr-world-trace.v1'
MAX_DOCUMENT_BYTES = 512 * 1024 * 1024


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def trace_path(root, tag):
    name = hashlib.sha256(tag.encode()).hexdigest()
    return Path(root) / name[:2] / (name + '.raw.zst')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as sink:
        json.dump(value, sink, separators=(',', ':'), allow_nan=False)
        sink.write('\n')
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def write_trace(path, document, level=3):
    path = Path(path)
    if document.get('schema') != SCHEMA:
        raise ValueError('invalid raw trace schema')
    encoded = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ValueError('raw document exceeds bounded decoder limit')
    packed = pa.Codec('zstd', compression_level=level).compress(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as sink:
        sink.write(MAGIC + struct.pack('<Q', len(encoded)))
        sink.write(packed)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    return {'raw_sha256': digest_file(path), 'raw_bytes': path.stat().st_size,
            'decoded_bytes': len(encoded), 'decisions': len(document['game']['decisions'])}


def read_trace(path):
    if Path(path).stat().st_size > MAX_DOCUMENT_BYTES + 1024 * 1024:
        raise ValueError('compressed raw trace exceeds decoder limit')
    with Path(path).open('rb') as source:
        prefix = source.read(16)
        if len(prefix) != 16 or prefix[:8] != MAGIC:
            raise ValueError('invalid raw trace framing')
        length, = struct.unpack('<Q', prefix[8:])
        if not 0 < length <= MAX_DOCUMENT_BYTES:
            raise ValueError('invalid raw trace document length')
        decoded = pa.Codec('zstd').decompress(source.read(), length).to_pybytes()
    if len(decoded) != length:
        raise ValueError('raw trace length mismatch')
    document = json.loads(decoded)
    if document.get('schema') != SCHEMA:
        raise ValueError('unsupported raw schema')
    return document


def check_trace(document):
    meta = document['metadata']
    fidelity = document['fidelity']
    if (meta.get('cut_at') is not None or fidelity['tower_hp_error'] != 0 or
            fidelity['winner_match'] is not True or fidelity['crowns_match'] is not True):
        raise ValueError('raw trace is not full exact')
    game = document['game']
    if document.get('failures'):
        raise ValueError('raw trace contains native command failures')
    if game['header']['tag'] != meta['tag'] or document['prepared_game']['tag'] != meta['tag']:
        raise ValueError('raw trace identity mismatch')
    if meta['actual_seed'] != document['prepared_game']['match']['rndSeed']:
        raise ValueError('actual seed differs from the reconstruction recipe')
    if type(meta['actual_seed']) is not int or not 0 <= meta['actual_seed'] < 2 ** 31:
        raise ValueError('invalid actual seed')
    ticks = [decision['tick'] for decision in game['decisions']]
    if not ticks or any(a >= b for a, b in zip(ticks, ticks[1:])):
        raise ValueError('nonmonotonic raw trace clock')
    if len(ticks) != meta['decisions'] or ticks[0] != 5:
        raise ValueError('raw trace sampling coverage mismatch')
    terminal_tick = (game['header'].get('terminal') or {}).get('native_tick')
    if terminal_tick is not None:
        last = game['decisions'][-1]
        if ticks[-1] > terminal_tick or (last.get('engineEnded') is not True and ticks[-1] != terminal_tick // 5 * 5):
            raise ValueError('raw trace is missing its terminal tail')
    for previous, decision in zip(ticks, game['decisions'][1:]):
        gap = decision['tick'] - previous
        if not 0 < gap <= 5 or (gap != 5 and decision.get('engineEnded') is not True):
            raise ValueError('missing intermediate raw decisions or unmarked terminal frame')
    for decision in game['decisions']:
        if decision['lean']['tick'] != decision['tick']:
            raise ValueError('raw reader clock mismatch')
        objects = decision['objects']
        if any('addr' in obj for obj in objects):
            raise ValueError('raw trace contains native addresses')
        ids = [o['nativeObjectId'] for o in objects]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate object identity')
        raw_ids = [o['nativeObjectId'] for o in decision['rawNative']['objects']]
        unit_ids = [o['nativeObjectId'] for o in decision['units']]
        if len(raw_ids) != len(set(raw_ids)) or len(unit_ids) != len(set(unit_ids)) or set(unit_ids) != set(ids):
            raise ValueError('duplicate or incomplete native/unit field identity')
        if set(ids) != {o['nativeObjectId'] for o in decision['rawNative']['objects']}:
            raise ValueError('native field coverage mismatch')
        for owner in (0, 1):
            raw = next(p for p in decision['rawNative']['players'] if p['owner'] == owner)
            player = next(p for p in decision['lean']['state']['players'] if p['owner'] == owner)
            if raw['player.elixir_raw'] != player['elixirRaw']:
                raise ValueError('raw elixir mismatch')
        raw_objects = {o['nativeObjectId']: o for o in decision['rawNative']['objects']}
        units = {o['nativeObjectId']: o for o in decision['units']}
        for obj in objects:
            raw = raw_objects[obj['nativeObjectId']]
            if obj['hp'] is not None and (raw.get('health.hp') != obj['hp'] or raw.get('health.max') != obj['maxHp']):
                raise ValueError('raw health mismatch')
            if raw.get('buff.count', 0) < 0:
                raise ValueError('invalid negative buff count')
            if 'buff.count' in raw and raw['buff.count'] != len(units[obj['nativeObjectId']].get('buffs', [])):
                raise ValueError('raw buff entries are incomplete')
    return True
