"""Cluster-side staging/replacement of gen-r6c with the ability-budget-corrected IL-r6 run.

Run --prepare first, validate experiments/ability_budget_smoke.py in the staged
source, then --replace. Existing checkpoints/results are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path.home() / 'cr-ppo'
BASE_SOURCE = ROOT / 'src-gen6b-1027'
SOURCE = ROOT / 'src-gen6d-1027'
BASE_RUN = ROOT / 'runs/gen-r6c'
RUN = ROOT / 'runs/gen-r6d'
DIAG = ROOT / 'diag/ability-budget-r6d-1027'
PYTHON = '/public/app/miniconda3/py312_24.4.0-0/bin/python'
NEW_FILES = ('assets/ability_costs.json', 'tools/export_ability_costs.py',
             'tests/test_engine_budget.py', 'experiments/ability_budget_smoke.py')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(archive):
    assert BASE_SOURCE.is_dir() and BASE_RUN.is_dir(), 'run this on the training cluster'
    assert not SOURCE.exists() and not RUN.exists(), 'refusing to replace an existing snapshot/run'
    shutil.copytree(BASE_SOURCE, SOURCE, symlinks=True, ignore=shutil.ignore_patterns(
        'runs', 'data', '.git', '__pycache__', '*.pyc', '.pytest_cache'))
    with tempfile.TemporaryDirectory(prefix='cr-r6d-patch-') as temp:
        with tarfile.open(archive) as tar:
            tar.extractall(temp, filter='data')
        patch = str(Path(temp) / 'fix.patch')
        subprocess.run(['git', 'apply', '--check', patch], cwd=SOURCE, check=True)
        subprocess.run(['git', 'apply', patch], cwd=SOURCE, check=True)
        for name in NEW_FILES:
            dest = SOURCE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(temp) / name, dest)
    manifest = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')}
    dump(SOURCE / 'source-manifest.json', manifest)
    RUN.mkdir()
    (RUN / 'logs').mkdir()
    DIAG.mkdir(exist_ok=True)
    settings = json.loads((BASE_RUN / 'settings.json').read_text())
    assert settings['search_frac'] == settings['search_coef'] == 0
    dump(RUN / 'settings.json', settings)
    config = json.loads((BASE_RUN / 'loop.json').read_text())
    for name, command in config['jobs'].items():
        config['jobs'][name] = command.replace(str(BASE_SOURCE), str(SOURCE)).replace(
            str(BASE_RUN), str(RUN)).replace('r6c-', 'r6d-').replace('/tmp/gen-r6c', '/tmp/gen-r6d')
    assert config['champion']['path'].endswith('/il-runs/r6-1026/r6-final.pt')
    dump(RUN / 'loop.json', config)
    dump(RUN / 'provenance.json', dict(base_run='gen-r6c', init=config['champion']['path'],
         source=str(SOURCE), patch_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
         changes=['ability/card payment reservations', 'matching runtime masks', 'ability diagnostics'],
         settings_changed=False, resume_from_ppo=False))
    (RUN / 'launch-notes.txt').write_text(
        'Authorized 27 Oct 2026: fresh PPO policy from IL r6 after correcting ability reservations.\n'
        'No gen-r6c optimizer/policy checkpoint resumed. Same reward, opponents, exploration, BC, gates and caps.\n'
        'Ability costs exported from exact build 160402020 / d22 data; actor and PPO masks agree.\n'
        'Ability probabilities, advantages, sampled/accepted/confirmed presses and expired cards are logged.\n')
    print(json.dumps(dict(source=str(SOURCE), run=str(RUN), native_proof=str(DIAG / 'native-proof.json'))))


def replace():
    proof = json.loads((DIAG / 'native-proof.json').read_text())
    assert len(proof) >= 15 and all(r['unsafe_card_blocked'] and r['funded_card_completed']
                                  and r['expired_cards'] == 0 and r['ability_stats']['confirmed'] == 1 for r in proof)
    manifest = json.loads((SOURCE / 'source-manifest.json').read_text())
    assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
               for name, digest in manifest.items()), 'source changed since validation'
    assert not (RUN / 'loop.pid').exists() and not (RUN / 'endpoint.json').exists()
    state = json.loads((BASE_RUN / 'loop-state.json').read_text())
    assert all(str(job).isdigit() for job in state['jobs'].values())
    pid = int((BASE_RUN / 'loop.pid').read_text())
    cmdline = Path(f'/proc/{pid}/cmdline')
    if cmdline.exists():
        command = cmdline.read_bytes().replace(b'\0', b' ').decode()
        assert 'loop.py' in command and str(BASE_RUN) in command, command
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not cmdline.exists(): break
            time.sleep(.1)
    sys.path.insert(0, str(SOURCE))
    from loop import Loop
    old = Loop(BASE_RUN)
    shutil.copy2(BASE_RUN / 'report.md', BASE_RUN / 'report-before-gen-r6d.md')
    old.stop('user requested fresh IL-r6 retraining with corrected ability reservations; successor gen-r6d')
    old.save()
    old.report({}, old.metrics())
    with (RUN / 'loop.out').open('ab') as log:
        process = subprocess.Popen([PYTHON, str(SOURCE / 'loop.py'), str(RUN)], cwd=SOURCE,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    (RUN / 'loop.pid').write_text(str(process.pid) + '\n')
    print(json.dumps(dict(old_run_stopped=True, new_run=str(RUN), loop_pid=process.pid,
                          native_cases=len(proof))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--replace', action='store_true')
    parser.add_argument('--archive', default='/tmp/cr-gen6d-fix.tar')
    args = parser.parse_args()
    prepare(args.archive) if args.prepare else replace()
