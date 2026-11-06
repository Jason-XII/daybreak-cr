from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from cr import observation as C
from cr.compression import compress_chunk, compress_frame, grouped_v2
from cr.history_imitation import collate


def entities(count):
    x = dict(obj_side=np.ones(count, np.int64), obj_card=np.full(count, 2, np.int64),
             obj_data=np.full(count, 3, np.int64), obj_kind=np.ones(count, np.int64),
             obj_float=np.zeros((count, C.OBJ_FLOATS), np.float32),
             eff_id=np.zeros((count, C.MAX_EFFECTS), np.int64),
             eff_left=np.zeros((count, C.MAX_EFFECTS), np.float32),
             v2_entity=np.zeros((count, 2, 6), np.float32))
    x['v2_entity'][:, 0, 0] = np.arange(count) + 1
    x['v2_entity'][:, 0, 1] = 1
    return x


class CompressionTests(unittest.TestCase):
    def test_pairs_retain_both_original_rows(self):
        x = entities(42)
        result, telemetry = compress_frame(x, 999)
        self.assertEqual(telemetry['pairs'], 21)
        self.assertEqual(result['v2_entity'].shape, (21, 2, 6))
        np.testing.assert_array_equal(np.sort(result['v2_entity'][..., 0].reshape(-1)), np.arange(42)+1)
        np.testing.assert_array_equal(result['obj_count'], np.full(21, 2))

    def test_singletons_keep_empty_second_slot(self):
        x = entities(3)
        result, _ = compress_frame(x, 999)
        np.testing.assert_array_equal(result['v2_entity'], x['v2_entity'])
        self.assertFalse(result['v2_entity'][:, 1].any())

    def test_recursive_pairs_rejected(self):
        x = entities(2)['v2_entity']
        x[0, 1, 0] = 5
        with self.assertRaises(ValueError):
            grouped_v2(x, [[0, 1]])

    def test_chunk_collate_tracks_object_padding(self):
        items = []
        for count in (42, 3):
            x = {k: v[None] for k, v in entities(count).items()}
            x['scalars'] = np.zeros((1, C.SCALARS), np.float32)
            x['v2_global'] = np.ones((1, 4), np.float32)
            compressed, _ = compress_chunk(x, 999)
            items.append((compressed, {'what': np.zeros(1, np.int64)}))
        xs, ts = collate(items)
        self.assertEqual(xs['v2_entity'].shape, (2, 128, 24, 2, 6))
        self.assertEqual(xs['obj_side'].shape, (2, 128, 24))
        self.assertFalse(xs['v2_entity'][1, 0, 3:].any())
        self.assertTrue(torch.isfinite(xs['v2_entity']).all())


if __name__ == '__main__':
    unittest.main()
