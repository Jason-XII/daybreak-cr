"""Export ability-name costs from the exact engine CSV/TOML asset tree (Python 3.11+)."""
import csv
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import sys


def decoded(path):
    data = path.read_bytes()
    if data[:1] == b']':  # the game's LZMA-alone header uses a four-byte size
        data = lzma.decompress(data[:9] + b'\0' * 4 + data[9:], format=lzma.FORMAT_ALONE)
    return data.decode('utf-8-sig')


def export(assets, build):
    import tomllib
    root = Path(assets) / 'csv_logic'
    path = root / 'character_abilities.csv'
    rows = list(csv.reader(io.StringIO(decoded(path))))
    definitions = {row[0]: dict(zip(rows[0], row)) for row in rows[2:] if row and row[0]}
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in sorted(root.rglob('*.toml')):
        # Game files can repeat unrelated TOML tables; parse only the scalar
        # cost declarations we need instead of requiring the whole file to be strict TOML.
        abilities, current = {}, None
        for line in decoded(path).splitlines():
            header = re.fullmatch(r'\[ABILITY\.([^\]]+)\]\s*(?:#.*)?', line.strip())
            if header:
                current = header[1].strip('"')
            elif line.lstrip().startswith('['):
                current = None
            elif current and re.match(r'\s*ManaCost\s*=', line):
                abilities[current] = tomllib.loads(line.strip())
        if not abilities:
            continue
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, row in abilities.items():
            definitions.setdefault(name, {}).update(row)
    costs = {}
    for name, row in definitions.items():
        cost = row.get('ManaCost')
        if cost not in (None, ''):
            cost = float(cost)
            if not 0 <= cost <= 10:
                raise ValueError(f'{name}: invalid ability cost {cost}')
            costs[name] = cost
    if costs.get('goblinstein_ability') != 2 or costs.get('Musketeer_hero_Ability') != 3:
        raise ValueError('ability costs do not match the expected engine data')
    return dict(build=str(build), data=Path(assets).name, abilities=costs, sources=sources)


if __name__ == '__main__':
    data = export(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(f"exported {len(data['abilities'])} ability costs from {len(data['sources'])} files")
