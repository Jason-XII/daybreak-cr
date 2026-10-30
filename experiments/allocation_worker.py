"""Keep a Slurm allocation while child commands change between curriculum stages."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)


def terminate(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def launch(spec,folder,name):
    env=dict(os.environ,CR_ALLOCATION_CHILD='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.update(spec.get('env',{}))
    log=(folder/(name+'.log')).open('a')
    proc=subprocess.Popen(spec['argv'],cwd=spec.get('cwd'),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();return proc


def run(root,role,poll=2):
    folder=root/'allocations'/role;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=aux=None;held=extra=None;finished=set();retries={};stopping=False;child_started=None
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping:
            try:task=json.loads((folder/'task.json').read_text())
            except (OSError,ValueError):task=None
            key=task['id'] if task else None
            if key!=held:
                terminate(child);child=None;held=key
                if task and task.get('argv'):
                    child=launch(task,folder,'task-'+key);child_started=time.time()
            if child and child.poll() is not None:
                code=child.returncode
                write(folder/'result.json',dict(id=held,exit_code=code,at=time.time()))
                if code and retries.get(held,0)<5:
                    retries[held]=retries.get(held,0)+1;time.sleep(min(10,poll));child=launch(task,folder,'task-'+key);child_started=time.time()
                else:child=None
            try:spec=json.loads((folder/'aux.json').read_text())
            except (OSError,ValueError):spec=None
            if spec and spec['id'] not in finished and aux is None:
                extra=spec['id'];aux=launch(spec,folder,'aux-'+extra)
            if aux and aux.poll() is not None:
                write(folder/'aux-result.json',dict(id=extra,exit_code=aux.returncode,at=time.time()))
                finished.add(extra);aux=None
            write(folder/'heartbeat.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),task=held,
                                              child=child.pid if child else None,child_started=child_started,at=time.time()))
            time.sleep(poll)
    finally:
        terminate(child);terminate(aux)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('role');a=p.parse_args();run(a.root,a.role)
