"""Deterministic evaluation tasks consumed by existing PPO collectors, without new jobs."""
import json
import os
from pathlib import Path


class AssignedEvaluations:
    def __init__(self,run,index):
        self.run,self.index=Path(run),index
        self.key=None;self.tasks=[];self.done=set();self.stream=None

    def next(self,rng):
        try:
            spec=json.loads((self.run/'paired-eval.json').read_text())
        except (OSError,ValueError):return None
        if not spec.get('active') or rng.random()>=spec.get('fraction',.5):return None
        slot=spec['slots'].get(str(os.environ.get('SLURM_JOB_ID')))
        if slot is None:return None
        key=(spec['plan'],slot['offset']+self.index,spec['collectors'])
        if key!=self.key:
            if self.stream:self.stream.close()
            self.key=key
            plan=json.loads(Path(spec['plan']).read_text())
            self.tasks=[t for t in plan['tasks'] if t['id']%spec['collectors']==key[1]]
            path=Path(spec['results'])/f'games-collector-{key[1]}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            self.done={json.loads(s)['id'] for s in path.read_text().splitlines()} if path.exists() else set()
            self.stream=path.open('a')
        return next((t for t in self.tasks if t['id'] not in self.done),None)

    def finish(self,task,score,state,refused):
        row=dict(**task,score=score,crowns=list(state.crowns),tick=state.tick,refused=refused)
        self.stream.write(json.dumps(row)+'\n');self.stream.flush();self.done.add(task['id'])


def models(task):
    a=('paired-candidate',task['candidate']);b=('paired-opponent',task['opponent'])
    return [a,b] if task['a_owner']==0 else [b,a]
