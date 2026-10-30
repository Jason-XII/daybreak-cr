"""tools/check_inputs.py (31 Oct 2026): an id seen in battles but missing from the vocabulary, or a card / unit without
stats, fails; projectiles and areas need no stats; a manifest's unknown ids fail."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_inputs', ROOT / 'tools/check_inputs.py')
check_inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_inputs)


def _assets(tmp):
    (tmp / 'vocab.json').write_text(json.dumps([26000000, 34000000, 9000001]))
    (tmp / 'card_stats.json').write_text(json.dumps({'fields': ['x'], 'stats': {'26000000': [1.0]}}))
    (tmp / 'effect_ids.json').write_text(json.dumps({'effects': [[9000001, 'Zap']]}))
    (tmp / 'effect_vectors.json').write_text(json.dumps({'fields': ['a'], 'effects': {'Zap': [1.0]}}))
    return tmp


def test_seen_ids(tmp_path):
    assets = _assets(tmp_path)
    seen = {'ids': {'26000000': {'games': 5, 'fields': {'cardId': 9}, 'names': {}},
                    '34000000': {'games': 4, 'fields': {'dataGlobalId': 9}, 'names': {'Knight|unit': 9}},
                    '1892649387': {'games': 3, 'fields': {'dataGlobalId': 3}, 'names': {'BarrelGoblin|unit': 3}},
                    '777': {'games': 2, 'fields': {'dataGlobalId': 2}, 'names': {'LogProjectile|projectile': 2}},
                    '9000001': {'games': 1, 'fields': {'buff': 1}, 'names': {}}}}
    (tmp_path / 'seen.json').write_text(json.dumps(seen))
    fails = {(i, what) for _, i, what, _, _ in check_inputs.check([tmp_path / 'seen.json'], assets)}
    assert fails == {(34000000, 'no card stats'), (1892649387, 'not in the vocabulary'), (1892649387, 'no card stats'),
                     (777, 'not in the vocabulary')}


def test_manifest_unknown_ids(tmp_path):
    assets = _assets(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps({'unknown_ids': {'13000075': {'lookups': 7, 'games': 2}}}))
    assert [f[1:3] for f in check_inputs.check([tmp_path / 'manifest.json'], assets)] == [(13000075, 'not in the vocabulary')]
