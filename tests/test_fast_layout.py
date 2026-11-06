from pathlib import Path
import sys
import unittest

from cr.inputs_v2.fast_layout import PathCollector
from cr.inputs_v2.v2_numeric import flatten


class FastLayoutTests(unittest.TestCase):
    def test_matches_nested_scalar_mixed_and_growing_lists(self):
        records = [dict(a=[1, None], b=[[2, [3]], [4, []]], c={'17': 2}),
                   dict(a=[1, 2, 3], b=[[2, [3, 4]]], c={'19': 5}),
                   dict(a=[], b=[1, dict(x=[None])], c={})]
        collector = PathCollector()
        for record in records:
            collector.visit(record)
        expected = {path for record in records for path, value in flatten(record)}
        self.assertEqual(collector.paths, expected)

    def test_nonfinite_and_strings_rejected(self):
        for record in (dict(a=[float('inf')]), dict(a='identity')):
            with self.assertRaises(ValueError):
                PathCollector().visit(record)


if __name__ == '__main__':
    unittest.main()
