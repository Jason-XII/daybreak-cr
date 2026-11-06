from pathlib import Path
import sys
import unittest

import numpy as np
from cr import observation as C
from cr.inputs_v2.v2_dataset import validate_clock_and_arrays


class ClockTests(unittest.TestCase):
    def data(self):
        values = dict(offsets=np.array([0, 1, 2]))
        for name in ('what', 'which', 'tile', 'ok', 'which_ok', 'value', 'next_when', 'next_which', 'next_which_ok'):
            values['t_'+name] = np.zeros(2)
        for name in C.OBJECT_KEYS:
            values['x_'+name] = np.zeros((2, 1))
        return values

    def test_implicit_clock_and_terminal_nonaligned_tick(self):
        self.assertEqual(validate_clock_and_arrays(self.data(), [dict(game_tick=5), dict(game_tick=9)]).tolist(), [5, 9])

    def test_explicit_clock_mismatch_rejected(self):
        data = self.data()
        data['t_tick'] = np.array([5, 11])
        with self.assertRaises(ValueError):
            validate_clock_and_arrays(data, [dict(game_tick=5), dict(game_tick=10)])

    def test_bad_offsets_or_labels_rejected(self):
        for key, value in [('offsets', np.array([0, 2, 1])), ('t_tile', np.zeros(1)), ('x_obj_side', np.zeros(1))]:
            data = self.data()
            data[key] = value
            with self.assertRaises(ValueError):
                validate_clock_and_arrays(data, [dict(game_tick=5), dict(game_tick=10)])

    def test_duplicate_or_nonaligned_nonterminal_rejected(self):
        for ticks in ([5, 5], [6, 10]):
            with self.assertRaises(ValueError):
                validate_clock_and_arrays(self.data(), [dict(game_tick=t) for t in ticks])


if __name__ == '__main__':
    unittest.main()
