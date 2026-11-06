from copy import deepcopy
import unittest

from tools.v2_merge_layout import merge


class MergeTests(unittest.TestCase):
    def shard(self, name, paths):
        return dict(scope='full_verified_dataset_shard', manifest_sha256=name,
                    verification_sha256=name+'verify', episodes=2, frames=30,
                    max_entities=42, max_pending=2, canonicalization='known-coordinates-own-view-v1',
                    encoding='named-signed-log-valid-v1', globals=dict(paths=paths, width=2*len(paths)),
                    entities=dict(paths=['x', 'y'], width=4))

    def test_union_is_deterministic_and_keeps_lineage(self):
        result = merge([self.shard('a', ['a']), self.shard('b', ['b', 'a'])])
        self.assertEqual(result['globals'], dict(paths=['a', 'b'], width=4))
        self.assertEqual(result['episodes'], 4)
        self.assertEqual(len(result['source_shards']), 2)

    def test_duplicate_and_sample_rejected(self):
        shard = self.shard('a', ['a'])
        with self.assertRaises(ValueError):
            merge([shard, deepcopy(shard)])
        shard['scope'] = 'sample'
        with self.assertRaises(ValueError):
            merge([shard])


if __name__ == '__main__':
    unittest.main()
