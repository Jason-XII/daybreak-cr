#!/usr/bin/env python3
"""Quarantine a narrowly proven replay-data failure without retrying or accepting its trajectory."""
import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr.raw_trace import atomic_json, digest_file, trace_path


def classify(reason):
    if reason in ('ValueError: native deal calibration failed after ten configurations',
                  'ValueError: native deal calibration repeated a deck configuration'):
        return 'cached_deal_configuration_unresolvable'
    prefix = 'ValueError: native command execution failures: '
    if not reason.startswith(prefix):
        return None
    try:
        failures = ast.literal_eval(reason[len(prefix):])
    except (ValueError, SyntaxError):
        return None
    # The fixed exporter stores at most three failures; three visible rows may hide other errors.
    if not failures or len(failures) >= 3 or not all(
            isinstance(row, tuple) and len(row) == 3 and type(row[0]) is int and
            row[1:] == ('ability', 'no unit to press') for row in failures):
        return None
    return 'cached_ability_source_unresolvable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--assignment', type=Path, required=True)
    parser.add_argument('--payloads', type=Path, help='canonical JSONL payload directory')
    args = parser.parse_args()
    run = json.loads((args.data / 'run.json').read_text())
    payloads = args.payloads or Path(next(iter(run['provenance'].values())).get(
        'payload_root', '/tmp/DBCR/hf-ingest/payloads'))
    if digest_file(args.assignment) != next(iter(run['provenance'].values()))['assignment_sha256']:
        raise ValueError('assignment provenance mismatch')
    manifest_path = args.data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    backup = args.data / 'manifest.before-data-quarantine.json'
    export_done = json.loads((args.data / 'export.done.json').read_text())
    original_path = backup if backup.exists() else manifest_path
    original_sha = digest_file(original_path)
    if original_sha != export_done['manifest_sha256']:
        raise ValueError('original manifest does not match the exporter completion hash')
    if backup.exists():
        original = json.loads(backup.read_text())
        if (original['source_manifest_sha256'] != manifest['source_manifest_sha256'] or
                original['entries'] != manifest['entries']):
            raise ValueError('quarantine backup/source binding mismatch')
        manifest = original
    eligible = {entry['tag']: entry for entry in manifest['skipped']
                if entry['status'] == 'failed' and classify(entry['reason']) is not None}
    rows = [json.loads(line) for line in args.assignment.open()]
    selected = {row['tag']: row for row in rows if row['tag'] in eligible}
    by_part = {}
    for row in selected.values():
        by_part.setdefault(row['part'], {})[row['row']] = row
    changed = []
    for part, wanted in by_part.items():
        with (payloads / part).open('rb') as source:
            for index, line in enumerate(source):
                if index not in wanted:
                    continue
                row = wanted[index]
                item = json.loads(line)
                if item['tag'] != row['tag'] or item['payload']['source']['replay_tag'] != row['tag']:
                    raise ValueError('quarantine canonical payload identity mismatch')
                entry = eligible[row['tag']]
                import hashlib
                entry.update(status='skipped', skip_code=classify(entry['reason']),
                             payload_row_sha256=hashlib.sha256(line).hexdigest(),
                             provenance=run['provenance'][entry['engine']],
                             quarantined_previous_status='failed',
                             quarantine_policy_sha256=digest_file(Path(__file__)))
                changed.append(entry['tag'])
    if len(changed) != len(eligible):
        raise ValueError('quarantine source rows missing')
    if changed:
        if not backup.exists():
            backup.write_bytes(manifest_path.read_bytes())
        for tag in changed:
            outcome = trace_path(args.data / 'outcomes', tag).with_suffix('.json')
            if outcome.is_file():
                before = outcome.with_name(outcome.name + '.before-data-quarantine')
                if not before.exists():
                    before.write_bytes(outcome.read_bytes())
                old_outcome = json.loads(outcome.read_text())
                if (old_outcome.get('source_archive_sha256') != eligible[tag]['source_archive_sha256'] or
                        old_outcome.get('status') not in ('failed', 'skipped')):
                    raise ValueError('quarantine outcome source/status mismatch')
                atomic_json(outcome, eligible[tag])
        atomic_json(manifest_path, manifest)
    remaining = [entry for entry in manifest['skipped'] if entry['status'] == 'failed']
    report = {'quarantined_games': len(changed), 'quarantined_tags': changed,
              'remaining_failed_games': len(remaining), 'remaining_failures': remaining[:20],
              'engine_replays': 0, 'seed_search_attempts': 0,
              'policy_sha256': digest_file(Path(__file__)), 'manifest_sha256': digest_file(manifest_path),
              'original_manifest_sha256': original_sha,
              'assignment_sha256': digest_file(args.assignment),
              'unresolved_failures_remain_excluded': True}
    atomic_json(args.data / 'quarantine.done.json', report)
    print(json.dumps(report), flush=True)
    # Remaining failures stay failed and excluded, never converted into accepted data or known skips.


if __name__ == '__main__':
    main()
