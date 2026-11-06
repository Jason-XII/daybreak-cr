import math

import numpy as np


def flatten(value, path=''):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten(value[key], path+'.'+key if path else key)
    elif isinstance(value, (list, tuple)):
        yield path+'.length', len(value)
        for index, child in enumerate(value):
            yield from flatten(child, f'{path}[{index}]')
    elif value is None or isinstance(value, (bool, int, float)):
        if value is not None and not math.isfinite(value):
            raise ValueError('Nonfinite v2 field: '+path)
        yield path, value
    else:
        raise ValueError('Unsupported v2 value type: '+path)


class NumericLayout:
    """Frozen named scalar channels, each paired with an independent validity bit."""

    def __init__(self, paths):
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate numeric layout paths')
        self.paths = tuple(paths)
        self.index = {path: index for index, path in enumerate(paths)}

    @classmethod
    def discover(cls, records):
        return cls(sorted({path for record in records for path, value in flatten(record)}))

    @property
    def width(self):
        return 2 * len(self.paths)

    def encode(self, record):
        output = np.zeros((len(self.paths), 2), np.float32)
        for path, value in flatten(record):
            if path not in self.index:
                raise ValueError('V2 field exceeds frozen layout: '+path)
            if value is not None:
                index = self.index[path]
                output[index] = (math.copysign(math.log1p(abs(value)), value), 1)
        return output.reshape(-1)

    def metadata(self):
        return dict(version='named-signed-log-valid-v1', paths=list(self.paths), width=self.width)
