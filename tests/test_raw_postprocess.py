"""Synthetic fault coverage for quarantine evidence and independent raw verification."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import quarantine_raw_failures as Q
import verify_raw as V
from cr.raw_trace import SCHEMA, atomic_json, digest_file, trace_path, write_trace


def fixture(tmp_path):
    assignment = tmp_path / 'assignment.jsonl'
    raw = tmp_path / 'raw'
    raw.mkdir()
    tag = 'synthetic-raw-accepted'
    failed_tag = 'synthetic-no-controller'
    rows = [dict(tag=tag, part='unused.jsonl', row=0, engine='d22'),
            dict(tag=failed_tag, part='unused.jsonl', row=1, engine='d22')]
    assignment.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    source = tmp_path / 'full-exact.json'
    entries = [dict(tag=name, owner=owner, decisions=2, archive_sha256='source-sha', file='unused.npz')
               for name in (tag, failed_tag) for owner in (0, 1)]
    atomic_json(source, {'selection': 'full_exact_all_formats', 'entries': entries})
    fingerprints = {'d22': {'source_manifest_sha256': digest_file(source),
                             'assignment_sha256': digest_file(assignment),
                             'effective_asset_root': '/tmp/DBCR/conversion-kit/DATA/d22'}}
    decisions = []
    for tick in (5, 10):
        decisions.append({'tick': tick, 'lean': {'tick': tick, 'state': {'players': [
            {'owner': 0, 'elixirRaw': 50000}, {'owner': 1, 'elixirRaw': 40000}]}},
            'objects': [{'nativeObjectId': 5000000, 'hp': 1000, 'maxHp': 2000}],
            'units': [{'nativeObjectId': 5000000, 'buffs': []}],
            'rawNative': {'objects': [{'nativeObjectId': 5000000, 'health.hp': 1000, 'health.max': 2000}],
                          'players': [{'owner': 0, 'player.elixir_raw': 50000},
                                      {'owner': 1, 'player.elixir_raw': 40000}]}})
    document = {'schema': SCHEMA, 'metadata': {'tag': tag, 'decisions': 2, 'actual_seed': 1,
                'source_archive_sha256': 'source-sha', 'seed_search_attempts': 0,
                'calibration_seed_search_attempts': 0}, 'failures': [],
                'fidelity': {'tower_hp_error': 0, 'winner_match': True, 'crowns_match': True},
                'prepared_game': {'tag': tag, 'match': {'rndSeed': 1}},
                'provenance': fingerprints['d22'], 'game': {'header': {'tag': tag}, 'decisions': decisions}}
    target = trace_path(raw / 'traces', tag)
    sizes = write_trace(target, document)
    validation = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.02
    accepted = {'tag': tag, 'engine': 'd22', 'status': 'accepted', 'file': str(target.relative_to(raw)),
                'actual_seed': 1, 'source_archive_sha256': 'source-sha', 'validation': validation, **sizes}
    failed = {'tag': failed_tag, 'engine': 'd22', 'status': 'failed', 'source_archive_sha256': 'source-sha',
              'reason': "ValueError: native command execution failures: [(10, 'ability', 'unit 5000010 has no ability controller')]"}
    manifest = {'schema': SCHEMA, 'entries': [accepted], 'skipped': [failed], 'provenance': fingerprints,
                'assignment_sha256': digest_file(assignment), 'source_manifest_sha256': digest_file(source)}
    atomic_json(raw / 'manifest.json', manifest)
    atomic_json(raw / 'run.json', {'provenance': fingerprints})
    atomic_json(raw / 'export.done.json', {'manifest_sha256': digest_file(raw / 'manifest.json')})
    return raw, assignment, source, document


def test_unknown_failure_remains_failed_and_excluded(tmp_path, monkeypatch):
    raw, assignment, source, document = fixture(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['quarantine', '--data', str(raw), '--assignment', str(assignment)])
    Q.main()
    after = json.loads((raw / 'manifest.json').read_text())
    assert after['skipped'][0]['status'] == 'failed'
    monkeypatch.setattr(sys, 'argv', ['verify', '--data', str(raw), '--source-manifest', str(source),
                                    '--assignment', str(assignment), '--allow-excluded-failures', '--workers', '1'])
    with pytest.raises(SystemExit) as result:
        V.main()
    assert result.value.code == 0
    report = json.loads((raw / 'verify.done.json').read_text())
    assert report['status'] == 'complete_with_excluded_failures'
    assert report['raw_games'] == 1 and report['failed_games'] == 1 and report['skipped_games'] == 0


def test_missing_tail_refused_against_old_source(tmp_path):
    raw, assignment, source, document = fixture(tmp_path)
    document['game']['decisions'].pop()
    document['metadata']['decisions'] = 1
    manifest = json.loads((raw / 'manifest.json').read_text())
    entry = manifest['entries'][0]
    entry.update(write_trace(raw / entry['file'], document))
    result = V.verify_entry((raw, entry, manifest['provenance'], 2))
    assert result['ok'] is False and result['error'] == 'decision count mismatch'


def test_forged_skip_reason_is_rejected(tmp_path, monkeypatch):
    raw, assignment, source, document = fixture(tmp_path)
    before = raw / 'manifest.before-data-quarantine.json'
    before.write_bytes((raw / 'manifest.json').read_bytes())
    manifest = json.loads(before.read_text())
    forged = manifest['skipped'][0]
    forged.update(status='skipped', skip_code='cached_ability_source_unresolvable',
                  quarantined_previous_status='failed', payload_row_sha256='bogus',
                  provenance=manifest['provenance']['d22'], quarantine_policy_sha256=digest_file(Path(Q.__file__)))
    atomic_json(raw / 'manifest.json', manifest)
    atomic_json(raw / 'quarantine.done.json', {'original_manifest_sha256': digest_file(before),
                'manifest_sha256': digest_file(raw / 'manifest.json'), 'policy_sha256': digest_file(Path(Q.__file__))})
    monkeypatch.setattr(sys, 'argv', ['verify', '--data', str(raw), '--source-manifest', str(source),
                                    '--allow-excluded-failures', '--workers', '1'])
    with pytest.raises(SystemExit) as result:
        V.main()
    assert result.value.code == 1
    report = json.loads((raw / 'verify.done.json').read_text())
    assert any('invalid classified failure evidence' in error for error in report['errors'])


@pytest.mark.parametrize('allow_search', (False, True))
def test_search_attempt_validation_uses_bound_policy(tmp_path, allow_search):
    raw, assignment, source, document = fixture(tmp_path)
    manifest = json.loads((raw / 'manifest.json').read_text())
    policy = {'reuse_cache': True, 'allow_seed_search': allow_search, 'skip_failed': True,
              'seeds': 16, 'calibration_seeds': 64}
    manifest['provenance']['d22']['replay_policy'] = policy
    document['provenance']['replay_policy'] = policy
    document['metadata']['replay_policy'] = policy
    document['metadata']['seed_search_attempts'] = 2
    document['metadata']['calibration_seed_search_attempts'] = 1
    entry = manifest['entries'][0]
    entry.update(write_trace(raw / entry['file'], document))
    result = V.verify_entry((raw, entry, manifest['provenance'], 2))
    assert result['ok'] is allow_search
    if not allow_search:
        assert result['error'] == 'unexpected seed search'


def test_missing_tail_is_refused_even_without_old_decision_count(tmp_path):
    from cr.raw_trace import check_trace
    _, _, _, document = fixture(tmp_path)
    document['game']['header']['terminal'] = {'native_tick': 15}
    with pytest.raises(ValueError, match='terminal tail'):
        check_trace(document)


def test_raw_native_addresses_are_refused(tmp_path):
    from cr.raw_trace import check_trace
    _, _, _, document = fixture(tmp_path)
    document['game']['decisions'][0]['objects'][0]['addr'] = 1234
    with pytest.raises(ValueError, match='native addresses'):
        check_trace(document)


def test_materialization_preserves_current_features_and_annotations(tmp_path, monkeypatch):
    import materialize_raw as M
    import convert
    import numpy as np
    from synthetic import arrays
    raw, _, _, document = fixture(tmp_path)
    document['metadata'].update(kept=True, expert={'team': True, 'opponent': False}, battle_time='2026-11-05')
    document['game']['header'].update(episode={'source_team_owner': 1}, terminal={'winner': 1})
    target = trace_path(raw / 'traces', document['metadata']['tag'])
    sizes = write_trace(target, document)
    entry = {'tag': document['metadata']['tag'], 'file': str(target.relative_to(raw)), **sizes}
    out = tmp_path / 'features'
    out.mkdir()
    monkeypatch.setattr(convert, 'game_features', lambda *args: arrays(2, 0))
    monkeypatch.setattr(M.O, 'cached_legal', lambda: None)
    recipe = {'converter_sha256': 'recipe'}
    entries = M.build_one((raw, out, entry, recipe))
    assert [e['expert'] for e in entries] == [False, True]
    assert all(e['battle_time'] == '2026-11-05' for e in entries)
    metadata = convert.read_metadata(convert.archive_path(out, entry['tag']))
    assert metadata['feature_recipe'] == recipe and metadata['raw_sha256'] == sizes['raw_sha256']
    with np.load(convert.archive_path(out, entry['tag']), allow_pickle=False) as values:
        assert values['p0_x_ability'].shape[-1] == M.O.ABILITY_FLOATS
    assert M.build_one((raw, out, entry, recipe)) == entries
    with pytest.raises(ValueError, match='different recipe'):
        M.build_one((raw, out, entry, {'converter_sha256': 'changed'}))
