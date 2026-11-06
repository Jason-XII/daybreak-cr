from pathlib import Path
import sys
import unittest

import numpy as np
from cr.inputs_v2.v2_numeric import NumericLayout


class NumericTests(unittest.TestCase):
    def test_null_not_measured_zero(self):
        layout = NumericLayout.discover([dict(raw=[None, 0, -25])])
        a = layout.encode(dict(raw=[None, 0, -25])).reshape(-1, 2)
        self.assertEqual(a[layout.index['raw[0]'], 1], 0)
        self.assertEqual(a[layout.index['raw[1]'], 1], 1)
        self.assertLess(a[layout.index['raw[2]'], 0], 0)

    def test_dynamic_list_all_entries_and_keys(self):
        records = [dict(actions=[[123, [0, 1]], [456, [3, None]]], ctx={'17': 3}),
                   dict(actions=[], ctx={'19': 4})]
        layout = NumericLayout.discover(records)
        original = layout.encode(records[0])
        changed = dict(actions=[[123, [0, 1]], [456, [7, None]]], ctx={'17': 3})
        self.assertFalse(np.array_equal(original, layout.encode(changed)))
        self.assertIn('ctx.17', layout.index)
        self.assertIn('ctx.19', layout.index)
        self.assertIn('actions[1][1][1]', layout.index)

    def test_no_silent_truncation(self):
        layout = NumericLayout.discover([dict(actions=[1])])
        with self.assertRaises(ValueError):
            layout.encode(dict(actions=[1, 2]))
        with self.assertRaises(ValueError):
            layout.encode(dict(actions=[1], new_field=2))

    def test_serialized_layout_stable(self):
        record = dict(a=[1, None], b=dict(c=-5))
        layout = NumericLayout.discover([record])
        restored = NumericLayout(layout.metadata()['paths'])
        np.testing.assert_array_equal(layout.encode(record), restored.encode(record))


if __name__ == '__main__':
    unittest.main()
