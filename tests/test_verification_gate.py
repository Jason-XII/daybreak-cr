import hashlib
import json
import unittest

from cr.inputs_v2.verification_gate import verified_manifest


class GateTests(unittest.TestCase):
    def fixture(self):
        manifest = dict(entries=[dict(tag='good')], failed=[dict(tag='bad', status='verification_failed', reason='wrong queue')], missing=[])
        encoded = json.dumps(manifest).encode()
        report = dict(ok=True, training_ready=True, verified_games=1, verifier_sha256='strict',
                      selected_games=2, missing_games=0, failed_games=1, processed_games=2, scope_games=2,
                      manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                      errors=[dict(tag='bad', reason='wrong queue')], training_scope='retained manifest.entries only',
                      quarantine=dict(processed_games=2, scope_games=2, retained_games=1, episode_files_modified=False))
        return encoded, report

    def test_verified_retained_scope_with_exclusions(self):
        encoded, report = self.fixture()
        self.assertEqual(verified_manifest(encoded, report, 'strict')['entries'], [dict(tag='good')])

    def test_unquarantined_errors_or_stale_verifier_rejected(self):
        encoded, report = self.fixture()
        with self.assertRaises(ValueError):
            verified_manifest(encoded, report, 'old')
        report['errors'][0]['tag'] = 'good'
        with self.assertRaises(ValueError):
            verified_manifest(encoded, report, 'strict')

    def test_incomplete_scan_rejected(self):
        encoded, report = self.fixture()
        report['quarantine']['processed_games'] = 1
        with self.assertRaises(ValueError):
            verified_manifest(encoded, report, 'strict')

    def test_selected_partition_and_failed_count_rejected(self):
        for key, value in (('selected_games', 3), ('failed_games', 0), ('missing_games', 1), ('processed_games', 1)):
            encoded, report = self.fixture()
            report[key] = value
            with self.assertRaises(ValueError):
                verified_manifest(encoded, report, 'strict')


if __name__ == '__main__':
    unittest.main()
