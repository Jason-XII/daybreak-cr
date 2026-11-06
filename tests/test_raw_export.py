import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from itertools import product

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import export_raw as E
import quarantine_raw_failures as Q
from cr.raw_trace import MAGIC, SCHEMA, read_trace, write_trace


def accepted(**search):
    return dict(kept=True, hp_error=0, winner=True, crowns=True, search=search)


def test_cached_seed_options():
    options = E.cached_options(accepted(seed=7919 * 128, king={'1': 'cards'}, presses=[1, 0]))
    assert options == dict(seed=7919 * 128, kings={1: 'cards'}, press_choices=[1, 0], calibration_seeds=0)
    assert E.cached_options(accepted(runs=1))['calibration_seeds'] == 0


@pytest.mark.parametrize('metadata', [dict(accepted(), kept=False), dict(accepted(), cut_at=17),
                                    dict(accepted(), hp_error=1), accepted(seed=7), accepted(seed=True),
                                    accepted(seed_dependent=True)])
def test_failed_and_invalid_seed_never_replayed(metadata):
    with pytest.raises(ValueError):
        E.cached_options(metadata)


def test_raw_unknown_values_roundtrip(tmp_path):
    document = {'schema': SCHEMA, 'game': {'decisions': [
        {'state': 999, 'unknown': None, 'negative': -300, 'uint64': 2 ** 64 - 1, 'objects': []}
    ]}}
    target = tmp_path / 'raw.zst'
    original = copy.deepcopy(document)
    result = write_trace(target, document)
    assert document == original == read_trace(target)
    assert result['decisions'] == 1
    assert target.read_bytes().startswith(MAGIC)


def test_corrupt_trace_refused(tmp_path):
    target = tmp_path / 'bad.raw'
    target.write_bytes(b'not-a-trace')
    with pytest.raises(ValueError):
        read_trace(target)


def test_seed_search_never_called_on_cached_path(monkeypatch):
    import convert as C
    assert E.cached_options(accepted(seed=7919))['seed'] == 7919
    source = Path(C.__file__).read_text()
    assert 'calibration_seeds=64, prepared_game=None' in source
    assert 'calibrate_deal(game, native, seeds=calibration_seeds' in source
    def replay(*args, **kwargs):
        assert kwargs['calibration_seeds'] == 0
        return {'fidelity': {'tower_hp_error': 1, 'winner_match': False, 'crowns_match': True}}
    monkeypatch.setattr(C, 'play_replay', replay)
    monkeypatch.setattr(C, 'play_exact', lambda *a, **kw: pytest.fail('disabled search executed'))
    result, searched = E.replay_with_policy(C.Recorder(None), {}, {'calibration_seeds': 0},
        SimpleNamespace(allow_seed_search=False))
    assert searched is False


def export_fixture(monkeypatch, replay):
    meta = dict(accepted(runs=1), tag='test', engine='d22')
    monkeypatch.setattr(E, 'digest_file', lambda _: 'sha')
    monkeypatch.setattr(E.C, 'read_metadata', lambda _: meta)
    monkeypatch.setattr(E, 'payload_at', lambda *_: ({}, 'payload-sha'))
    monkeypatch.setattr(E.C, 'play_replay', replay)
    args = SimpleNamespace(source=ROOT, recipe_cache=None, reuse_cache=True, allow_seed_search=False,
                           skip_failed=True, seeds=16, calibration_seeds=64)
    job = dict(file='unused.npz', tag='test', engine='d22', source_archive_sha256='sha')
    return job, args


def test_command_failure_is_not_accepted(monkeypatch):
    job, args = export_fixture(monkeypatch, lambda *a, **kw: {'failures': [(10, 'card', 'queue failed')]})
    with pytest.raises(ValueError, match='native command execution failures'):
        E.export_one(job, None, args, {}, {})


@pytest.mark.parametrize('message,allowed', [
    ('no compatible native omitted-card layout', True),
    ('native opening hand and queue must cover eight distinct deck slots', False),
    ('native deal calibration repeated a deck configuration', False),
])
def test_only_known_missing_calibration_is_skipped(monkeypatch, message, allowed):
    def replay(*args, **kwargs):
        raise ValueError(message)
    job, args = export_fixture(monkeypatch, replay)
    if allowed:
        result = E.export_one(job, None, args, {}, {})
        assert result['status'] == 'skipped'
        assert result['skip_code'] == 'missing_implicit_calibration_seed'
    else:
        with pytest.raises(ValueError, match=message):
            E.export_one(job, None, args, {}, {})


def test_quarantine_never_promotes_unknown_command_errors():
    assert Q.classify("ValueError: native command execution failures: [(10, 'ability', 'no unit to press')]") == \
        'cached_ability_source_unresolvable'
    assert Q.classify("ValueError: native command execution failures: [(10, 'card', 'queue failed')]") is None
    assert Q.classify('RuntimeError: native worker crashed') is None
    assert Q.classify("ValueError: native command execution failures: [(10, 'ability', 'unit 5000010 has no ability controller')]") is None
    assert Q.classify("ValueError: native command execution failures: [(10, 'ability', 'no unit to press'), (11, 'ability', 'no unit to press'), (12, 'ability', 'no unit to press')]") is None
    assert Q.classify('ValueError: native deal calibration failed after ten configurations') == \
        'cached_deal_configuration_unresolvable'
    assert Q.classify('ValueError: native opening hand and queue must cover eight distinct deck slots') is None


@pytest.mark.parametrize('reuse,search,skip', list(product((False, True), repeat=3)))
def test_policy_flags_are_independent(reuse, search, skip):
    flags = ['--assignment', '/tmp/DBCR/unused', '--source', '/tmp/DBCR/unused',
             '--manifest', '/tmp/DBCR/unused', '--out', '/tmp/DBCR/unused',
             '--reuse-cache' if reuse else '--no-reuse-cache',
             '--allow-seed-search' if search else '--no-allow-seed-search',
             '--skip-failed' if skip else '--no-skip-failed']
    args = E.argument_parser().parse_args(flags)
    assert E.replay_policy(args) == {'reuse_cache': reuse, 'allow_seed_search': search, 'skip_failed': skip,
                                    'seeds': 16, 'calibration_seeds': 64}


def test_cache_disabled_does_not_use_saved_seed(monkeypatch):
    captured = []
    def replay(*a, **kwargs):
        captured.append(kwargs)
        return {'failures': [(10, 'card', 'queue failed')]}
    job, args = export_fixture(monkeypatch, replay)
    args.reuse_cache = False
    monkeypatch.setattr(E.C, 'read_metadata', lambda _: dict(accepted(seed=7919), tag='test', engine='d22'))
    with pytest.raises(ValueError, match='native command execution failures'):
        E.export_one(job, None, args, {}, {})
    assert captured[0]['seed'] is None and captured[0]['kings'] is None
    assert captured[0]['press_choices'] is None and captured[0]['calibration_seeds'] == 0


@pytest.mark.parametrize('initial_exact', (False, True))
def test_search_only_after_allowed_miss(monkeypatch, initial_exact):
    attempts = []
    missed = {'fidelity': {'tower_hp_error': 0 if initial_exact else 1, 'winner_match': True, 'crowns_match': True}}
    found = {'fidelity': {'tower_hp_error': 0, 'winner_match': True, 'crowns_match': True}}
    monkeypatch.setattr(E.C, 'play_replay', lambda *a, **kw: missed)
    def search(recorder, payload, **kwargs):
        attempts.append(kwargs)
        return found
    monkeypatch.setattr(E.C, 'play_exact', search)
    result, searched = E.replay_with_policy(E.C.Recorder(None), {}, {'calibration_seeds': 0},
        SimpleNamespace(allow_seed_search=True, seeds=7, calibration_seeds=11))
    assert searched is not initial_exact
    assert len(attempts) == int(not initial_exact)
    if attempts:
        assert attempts[0] == {'seeds': 7, 'calibration_seeds': 11}
        assert result is found


def test_disabled_search_never_enumerates_calibration(monkeypatch):
    calls = []
    def calibrate(*a, **kw):
        calls.append(kw)
        raise ValueError('native deal calibration failed after ten configurations')
    monkeypatch.setattr(E.C, 'calibrate', calibrate)
    recorder = E.C.Recorder(None)
    game = {'omitted': [[1], []], 'match': {'rndSeed': 1}}
    with pytest.raises(ValueError):
        E.C.calibrate_deal(game, recorder, seeds=0)
    assert len(calls) == 1 and recorder.search_attempts['calibration_seed'] == 0
    with pytest.raises(ValueError):
        E.C.calibrate_deal(game, recorder, seeds=3)
    assert recorder.search_attempts['calibration_seed'] == 3


def test_exact_search_propagates_calibration_limit(monkeypatch):
    calls = []
    result = {'fidelity': {'tower_hp_error': 0, 'winner_match': True, 'crowns_match': True}}
    def replay(*a, **kw):
        calls.append(kw)
        return result
    monkeypatch.setattr(E.C, 'play_replay', replay)
    E.C.play_exact(E.C.Recorder(None), {}, seeds=16, calibration_seeds=0)
    assert calls == [{'record': True, 'calibration_seeds': 0}]


def test_dice_attempts_are_counted_and_calibration_is_not_implicitly_enabled(monkeypatch):
    calls = []
    def replay(recorder, payload, **kwargs):
        calls.append(kwargs)
        seed = kwargs.get('seed')
        return {'game': {'king_alternatives': []}, 'presses': [],
                'fidelity': {'tower_hp_error': 0 if seed == 7919 else 1,
                             'winner_match': True, 'crowns_match': True}}
    monkeypatch.setattr(E.C, 'play_replay', replay)
    recorder = E.C.Recorder(None)
    result = E.C.play_exact(recorder, {}, seeds=2, calibration_seeds=0)
    assert result['search']['seed'] == 7919
    assert recorder.search_attempts == {'seed': 1, 'calibration_seed': 0}
    assert all(call['calibration_seeds'] == 0 for call in calls)


@pytest.mark.parametrize('skip_failed,expected', [(True, {'accepted'}), (False, {'accepted', 'rejected'})])
def test_previous_reject_selection_is_independent(tmp_path, monkeypatch, skip_failed, expected):
    import numpy as np
    monkeypatch.setattr(E, 'ROOT', tmp_path)
    source = tmp_path / 'source'
    payloads = tmp_path / 'hf-ingest/payloads'
    source.mkdir()
    payloads.mkdir(parents=True)
    assignment = tmp_path / 'assignment.jsonl'
    assignment.write_text(''.join(json.dumps({'tag': tag, 'part': 'part.jsonl', 'row': i, 'engine': 'd22'}) + '\n'
                                  for i, tag in enumerate(('accepted', 'rejected', 'partial'))))
    (payloads / 'part.jsonl').write_text('{}\n{}\n{}\n')
    archive = E.C.archive_path(source, 'rejected')
    metadata = {'tag': 'rejected', 'kept': False, 'engine': 'd22'}
    np.savez(archive, _metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    previous = source / 'manifest.json'
    previous.write_text(json.dumps({'entries': [], 'rejected': [{'tag': 'rejected'}]}))
    manifest = source / 'full-exact.json'
    manifest.write_text(json.dumps({'selection': 'full_exact_all_formats',
        'source_manifest_sha256': E.digest_file(previous),
        'entries': [{'tag': 'accepted', 'owner': owner, 'file': 'accepted.npz', 'archive_sha256': 'sha'}
                    for owner in (0, 1)]}))
    args = SimpleNamespace(assignment=assignment, manifest=manifest, source=source, limit=0, skip_failed=skip_failed)
    _, jobs = E.make_jobs(args)
    assert {job['tag'] for job in jobs} == expected
    assert 'partial' not in {job['tag'] for job in jobs}


def test_no_skip_failed_does_not_silently_use_cached_failure(monkeypatch, tmp_path):
    captured = []
    job, args = export_fixture(monkeypatch, lambda *a, **kw: captured.append(kw) or
                                {'failures': [(10, 'card', 'queue failed')]})
    args.skip_failed = False
    args.reuse_cache = False
    args.recipe_cache = tmp_path
    outcome = E.terminal_path(tmp_path, 'test')
    outcome.parent.mkdir(parents=True)
    outcome.write_text(json.dumps({'status': 'failed', 'source_archive_sha256': 'sha', 'reason': 'old failure'}))
    with pytest.raises(ValueError, match='native command execution failures'):
        E.export_one(job, None, args, {}, {})
    assert len(captured) == 1


def test_payload_paths_and_hashes_are_portable(tmp_path):
    import hashlib
    encoded = (json.dumps({'tag': 'test', 'payload': {'source': {'replay_tag': 'test'}, 'name': '\u00e9'}},
                          ensure_ascii=False) + '\n').encode()
    (tmp_path / 'part.jsonl').write_bytes(b'{}\n' + encoded)
    streams = {}
    payload, digest = E.payload_at(dict(part='part.jsonl', offset=3, tag='test'), streams, tmp_path)
    assert payload['name'] == '\u00e9' and digest == hashlib.sha256(encoded).hexdigest()
    for stream in streams.values():
        stream.close()


def test_recipe_cache_reuses_its_own_press_choices(monkeypatch, tmp_path):
    captured = []
    job, args = export_fixture(monkeypatch, lambda *a, **kw: captured.append(kw) or
                                {'failures': [(10, 'card', 'queue failed')]})
    args.recipe_cache = tmp_path
    cached_path = E.trace_path(tmp_path / 'traces', 'test')
    cached_path.parent.mkdir(parents=True)
    cached_path.touch()
    fingerprints = dict.fromkeys(('engine_sha256', 'data_fingerprint_sha256', 'data_tree_sha256', 'bindings_sha256'), 'sha')
    prepared = {'match': {'rndSeed': 123}, 'commands': []}
    monkeypatch.setattr(E, 'read_trace', lambda path: {'prepared_game': prepared, 'provenance': fingerprints,
        'metadata': {'source_archive_sha256': 'sha', 'payload_row_sha256': 'payload-sha', 'search': {'presses': [1, 0]}}})
    monkeypatch.setattr(E, 'check_trace', lambda document: True)
    with pytest.raises(ValueError, match='native command execution failures'):
        E.export_one(job, None, args, fingerprints, {})
    assert captured[0]['prepared_game'] is prepared
    assert captured[0]['press_choices'] == [1, 0] and captured[0]['seed'] is None
