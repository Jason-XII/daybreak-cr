import math

from cr.inputs_v2.input_spec import IN, rule
from cr.inputs_v2.v2_contract import numeric_frame, player_frame, read_episode


class PathCollector:
    def __init__(self):
        self.paths = set()
        self.scalar_lengths = {}

    def visit(self, value, path=''):
        if isinstance(value, dict):
            for key, child in value.items():
                self.visit(child, path+'.'+key if path else key)
        elif isinstance(value, list):
            self.paths.add(path+'.length')
            if all(child is None or isinstance(child, (bool, int, float)) for child in value):
                if any(child is not None and not math.isfinite(child) for child in value):
                    raise ValueError('Nonfinite layout input: '+path)
                previous = self.scalar_lengths.get(path, 0)
                self.paths.update(f'{path}[{i}]' for i in range(previous, len(value)))
                self.scalar_lengths[path] = max(previous, len(value))
            else:
                for index, child in enumerate(value):
                    self.visit(child, f'{path}[{index}]')
        elif value is None or isinstance(value, (bool, int, float)):
            if value is not None and not math.isfinite(value):
                raise ValueError('Nonfinite layout input: '+path)
            self.paths.add(path)
        else:
            raise ValueError('Unsupported layout value: '+path)


def inspect_episode(item):
    path, sha256 = item
    episode = read_episode(path, sha256)
    globals_paths, entity_paths = PathCollector(), PathCollector()
    maximum_entities = maximum_pending = 0
    for world in episode['frames']:
        maximum_entities = max(maximum_entities, len(world['entities']))
        for entity in world['entities']:
            for key in entity:
                if key != 'us' and (rule('entities[].'+key) or (None,))[0] != IN:
                    raise ValueError('Unclassified entity layout field: '+key)
            for key in entity.get('us', {}):
                if (rule('entities[].us.'+key) or (None,))[0] != IN:
                    raise ValueError('Unclassified entity state field: '+key)
            # Canonicalization changes values, not entity keys or list dimensions.
            entity_paths.visit(entity)
        global_world = dict(world, entities=[])
        for owner in (0, 1):
            frame = numeric_frame(player_frame(global_world, owner), owner)
            frame.pop('entities')
            globals_paths.visit(frame)
            maximum_pending = max(maximum_pending, sum(command['kind'] == 1 and command['side'] == owner
                and command['t2'] > world['game_tick'] for command in world['queue']))
    return dict(globals=globals_paths.paths, entities=entity_paths.paths, frames=len(episode['frames']),
                max_entities=maximum_entities, max_pending=maximum_pending)
