"""Enter an allocation keeper without replacing an already queued Slurm job."""
import json
import os
from pathlib import Path
import sys
import time


def clean_environment(env):
    rank_keys={'RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE',
               'MASTER_ADDR','MASTER_PORT'}
    return {k:v for k,v in env.items() if k not in rank_keys and not k.startswith('TORCHELASTIC_')}


def enter(out=None):
    if os.environ.get('CR_ALLOCATION_CHILD'):return
    if out is None:
        if len(sys.argv)<2 or sys.argv[1]!='learn' or '--out' not in sys.argv:return
        out=Path(sys.argv[sys.argv.index('--out')+1])
    marker=out/'allocation-bootstrap.json'
    if not marker.exists():return
    if int(os.environ.get('RANK',0)):
        while True:time.sleep(30)
    spec=json.loads(marker.read_text())
    os.execve(sys.executable,[sys.executable,spec['worker'],spec['root'],'learner'],clean_environment(os.environ))
