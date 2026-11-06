import hashlib
import json


def verified_manifest(manifest_bytes, report, required_verifier_sha256):
    manifest = json.loads(manifest_bytes)
    if not required_verifier_sha256 or report.get('verifier_sha256') != required_verifier_sha256:
        raise ValueError('Verification source is not the required strict verifier')
    if not report.get('ok') or not report.get('training_ready'):
        raise ValueError('Conversion is not training-ready')
    if report.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Verification does not bind the current manifest')
    entries = manifest['entries']
    failed = manifest.get('failed', [])
    if manifest.get('missing') or not entries or report.get('verified_games') != len(entries):
        raise ValueError('Incomplete retained dataset verification')
    tags = [entry['tag'] for entry in entries + failed]
    if len(tags) != len(set(tags)):
        raise ValueError('Duplicate or overlapping retained/excluded tags')
    if report.get('selected_games') != len(tags) or report.get('missing_games') != 0:
        raise ValueError('Verification selected scope differs from retained/excluded partition')
    if report.get('failed_games') != len(failed):
        raise ValueError('Verification excluded count differs from manifest')
    if report.get('processed_games') != report.get('scope_games'):
        raise ValueError('Verification has not processed its entire original scope')
    errors = report.get('errors', [])
    if errors:
        if report.get('training_scope') != 'retained manifest.entries only':
            raise ValueError('Error report lacks explicit quarantine scope')
        rejected = {entry['tag']: entry for entry in failed if entry.get('status') == 'verification_failed'}
        if {error.get('tag') for error in errors if isinstance(error, dict)} != set(rejected):
            raise ValueError('Verification errors do not cover all quarantined failures')
        for error in errors:
            if not isinstance(error, dict) or error.get('tag') not in rejected:
                raise ValueError('Unquarantined verification error')
            if error.get('reason') != rejected[error['tag']].get('reason'):
                raise ValueError('Quarantine error provenance mismatch')
        quarantine = report.get('quarantine', {})
        if quarantine.get('processed_games') != quarantine.get('scope_games'):
            raise ValueError('Incomplete quarantine scan')
        if quarantine.get('retained_games') != len(entries):
            raise ValueError('Quarantine retained count mismatch')
        if quarantine.get('episode_files_modified') is not False:
            raise ValueError('Quarantine modified episode files')
    return manifest
