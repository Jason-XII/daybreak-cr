"""A completed task and an auxiliary task must not terminate the allocation keeper."""
import json
from pathlib import Path
import subprocess
import sys
import time


def test_keeper_survives_completed_tasks_and_stage_changes(tmp_path):
    source=Path(__file__).resolve().parents[1]/'experiments'
    folder=tmp_path/'allocations/actors';folder.mkdir(parents=True)
    def task(name,aux=False):
        target=tmp_path/name
        spec=dict(id=name,argv=[sys.executable,'-c',f'from pathlib import Path;Path({str(target)!r}).write_text("done")'])
        p=folder/('aux.json' if aux else 'task.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(spec));tmp.replace(p)
        return target
    def until(predicate):
        deadline=time.monotonic()+10
        while not predicate():
            assert time.monotonic()<deadline
            assert proc.poll() is None
            time.sleep(.05)
    first=task('train-1')
    code=f'import sys;from pathlib import Path;sys.path.insert(0,{str(source)!r});from allocation_worker import run;run(Path({str(tmp_path)!r}),"actors",.05)'
    proc=subprocess.Popen([sys.executable,'-c',code])
    try:
        until(first.exists);until(lambda:(folder/'result.json').exists())
        assert proc.poll() is None
        extra=task('evaluation-summary',True);until(extra.exists)
        second=task('train-next-stage');until(second.exists)
        assert proc.poll() is None
        until(lambda:json.loads((folder/'heartbeat.json').read_text())['task']=='train-next-stage')
        assert json.loads((folder/'heartbeat.json').read_text())['pid']==proc.pid
    finally:
        proc.terminate();proc.wait(timeout=10)
