import unittest

from tools.v2_build_training_manifest import assemble


class ManifestTests(unittest.TestCase):
    def fixture(self):
        episode = dict(tag='a', file='episodes/00/a.inp.zst', frames=10, validation=True, sha256='episode', bytes=20)
        base = dict(schema=dict(vocab=[0, 1, 2]), entries=[dict(tag='a', owner=o, file='a.npz', decisions=10, validation=True) for o in (0, 1)])
        shard = dict(manifest=dict(entries=[episode], failed=[]), manifest_sha256='manifest',
                     base_manifest=base, base_files={'a.npz':dict(sha256='base', bytes=30)})
        layout = dict(scope='full_verified_dataset', episodes=1, source_shards=[dict(manifest_sha256='manifest')])
        return {'ZW1':shard}, layout

    def test_relocatable_files_and_two_views(self):
        shards, layout = self.fixture()
        manifest, delivery = assemble(shards, layout, ['ZW1'])
        self.assertEqual(len(manifest['entries']), 2)
        self.assertEqual(manifest['entries'][0]['base_archive'], 'base/ZW1/a.npz')
        self.assertEqual(len(delivery['files']), 2)

    def test_missing_host_split_and_paths_rejected(self):
        shards, layout = self.fixture()
        with self.assertRaises(ValueError):
            assemble(shards, layout, ['ZW1', 'QY3'])
        shards['ZW1']['base_manifest']['entries'][0]['validation'] = False
        with self.assertRaises(ValueError):
            assemble(shards, layout, ['ZW1'])
        shards, layout = self.fixture()
        shards['ZW1']['manifest']['entries'][0]['file'] = '../secret'
        with self.assertRaises(ValueError):
            assemble(shards, layout, ['ZW1'])


if __name__ == '__main__':
    unittest.main()
