"""Restartable, bounded curriculum supervisor. Login-node work is file/Slurm orchestration only.

PPO and all model evaluation/aggregation run inside Slurm allocations. Every 20 policy
updates: evaluate without rolling back the candidate. After 100: rerank/advance, recover,
or resume the previously authorized specialists if the target did not improve.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from curriculum_evaluation import make_plan

PYTHON='/public/app/miniconda3/py312_24.4.0-0/bin/python'
TERMINAL={'BOOT_FAIL','CANCELLED','COMPLETED','DEADLINE','FAILED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','TIMEOUT'}


def put(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def decision(mode, summary, general_floor=.47):
    if summary['general_score'] < general_floor:
        return 'recover' if mode=='target' else 'fallback'
    if mode=='target' and summary['target_gain'] <= 0:
        return 'fallback'
    return 'advance'


class Curriculum:
    def __init__(self,root):
        self.root=root;self.cfg=json.loads((root/'curriculum.json').read_text())
        self.source=Path(self.cfg['source']);self.tools=root/'tools'
        self.path=root/'curriculum-state.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(
            started=time.time(),phase='new',stage_index=0,stage=None,jobs={},history=[],alerts=[],errors=0,
            accepted=dict(name='general-v320',policy=str(root/'init/policy-v320.pt'),critic=str(root/'init/critic-v320.pt')),
            ranking=str(root/'data/initial-ranking.json'),stopped=None)

    def save(self):
        put(self.path,self.state)
        lines=['# Automated deck curriculum','',f"Updated: {datetime.now().isoformat(timespec='seconds')}",
               f"Phase: {self.state['phase']}; accepted: {self.state['accepted']['name']}",
               f"Stage: {json.dumps(self.state['stage'])}",'', '## History','',
               *[f'- {json.dumps(x)}' for x in self.state['history']], '', '## Alerts','',
               *[f'- {x}' for x in self.state['alerts']]]
        (self.root/'report.md').write_text('\n'.join(lines)+'\n')

    def event(self,message):
        print(datetime.now().isoformat(timespec='seconds'),message,flush=True)
        with (self.root/'events.log').open('a') as f:f.write(f'{datetime.now().isoformat()} {message}\n')

    def alert(self,message):
        self.state['alerts'].append(message);self.event('ALERT '+message)
        put(self.root/'alerts.json',self.state['alerts'])

    def cancel(self):
        jobs=[x['id'] for x in self.state['jobs'].values()]
        if jobs:subprocess.run(['scancel',*jobs],capture_output=True)
        self.state['jobs']={}

    def submit(self,name,command,resources):
        # QOS wall limits: competition (A100/RTX5090) 4 days, Students 1 day; jobs can't be extended later.
        limit='4-00:00:00' if 'competition' in resources else '1-00:00:00'
        cmd=['sbatch','--parsable',*resources,'-t',limit,'-J',name,
             '-o',str(self.root/'logs'/f'{name}-%j.out'),'--wrap',command]
        result=subprocess.run(cmd,capture_output=True,text=True)
        job=result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():raise RuntimeError('sbatch: '+result.stderr)
        self.event(f'{name}: submitted {job}')
        return dict(id=job,command=command,resources=resources,name=name,restarts=0)

    def ensure(self):
        r=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%i'],capture_output=True,text=True)
        if r.returncode:return
        queued=set(r.stdout.split())
        for key,job in list(self.state['jobs'].items()):
            if self.state['phase'] in ('gate','rank') and key.isdigit():
                arena=self.state['arena'];f=Path(arena['root'])/'results'/f'games-{key}.jsonl'
                expected=(arena['expected']+2-int(key))//3
                if f.exists() and sum(1 for _ in f.open())==expected:
                    continue
            if job['id'] in queued:continue
            s=subprocess.run(['scontrol','show','job',job['id']],capture_output=True,text=True)
            found=re.search(r'JobState=(\w+)',s.stdout)
            ended=(found and found[1] in TERMINAL) or 'Invalid job id specified' in s.stdout+s.stderr
            if not ended:continue
            if job.get('restarts',0)>=5:raise RuntimeError(f'{key} failed five times')
            new=self.submit(job['name'],job['command'],job['resources']);new['restarts']=job.get('restarts',0)+1
            self.state['jobs'][key]=new;self.save()
            if self.cfg.get('resident') and self.state['phase'] in ('gate','rank'):
                run=Path(self.state['stage']['run']);path=run/'paired-eval.json'
                spec=json.loads(path.read_text());slots={};offset=0
                for role,resources,count,servers in self.cfg['actors']:
                    slots[self.state['jobs'][role]['id']]=dict(offset=offset,count=count);offset+=count
                spec['slots']=slots;put(path,spec)

    def env(self,command):
        return 'export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; '+command

    def work(self,role,key,argv,cwd=None,aux=False):
        folder=self.root/'allocations'/role;folder.mkdir(parents=True,exist_ok=True)
        put(folder/('aux.json' if aux else 'task.json'),dict(id=key,argv=argv,cwd=str(cwd or self.source)))

    def disable_inline(self):
        if self.state.get('stage'):
            run=Path(self.state['stage']['run'])
            put(run/'paired-eval.json',dict(active=False));put(run/'eval.json',dict(matches=[]))

    def new_stage(self,mode='target',initial=None):
        if not self.cfg.get('resident'):self.cancel()
        else:self.disable_inline()
        self.state['stage_index']+=1;i=self.state['stage_index']
        initial=initial or copy.deepcopy(self.state['accepted'])
        ranking=json.loads(Path(self.state['ranking']).read_text())
        target=ranking['ranking'][0]['index'] if mode=='target' else None
        run=self.root/'stages'/f'{i:02d}-{mode}'
        for d in ('logs','pool','ckpt','league','init'):(run/d).mkdir(parents=True,exist_ok=True)
        settings=json.loads((self.root/'settings-template.json').read_text())
        meta=json.loads((self.root/'data/decks.json').read_text())
        warmup=3 if i==1 else 0
        settings.update(updates=warmup+20,value_warmup=warmup,
                        deck=meta['decks'][target]['deck'] if target is not None else [],
                        deck_frac=.6 if target is not None else 0.,
                        bc_index=str(self.root/'bc'/f'deck-{target}.json') if target is not None else self.cfg['general_bc'])
        put(run/'settings.json',settings);put(run/'champion.json',dict(name=initial['name'],path=initial['policy']))
        self.state.update(phase='train',stage=dict(index=i,mode=mode,run=str(run),target=target,
                          initial=initial,evaluation_reference=copy.deepcopy(self.state['accepted']),
                          warmup=warmup,policy_updates=20,cap=warmup+20,best_target=None))
        self.event(f'stage {i} {mode}: target {target}; starting {initial["name"]}')
        self.start_training()

    def start_training(self):
        s=self.state['stage'];run=Path(s['run']);settings=json.loads((run/'settings.json').read_text())
        accepted=self.state['accepted']
        settings['updates']=s['warmup']+100 if self.cfg.get('resident') else s['cap']
        if self.cfg.get('resident'):settings['hold_after_updates']=True
        settings['fixed_opponents']={'base-v320':[str(self.root/'init/policy-v320.pt'),.5],
                                     accepted['name']:[accepted['policy'],.5]}
        put(run/'settings.json',settings)
        put(run/'fixed-opponents.json',settings['fixed_opponents'])
        put(run/'champion.json',dict(name=accepted['name'],path=accepted['policy']))
        initial=s['initial'];gpus=self.cfg.get('learner_gpus',2)
        learn=shlex.join([PYTHON,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={gpus}',
                         'ppo.py','learn','--init',initial['policy'],'--critic-weights',initial['critic'],
                         '--out',str(run),'--engine',self.cfg['engine'],'--pool',str(self.root/'data/train.json')])
        command=self.env('cd '+shlex.quote(str(self.source))+' && '+learn)
        if self.cfg.get('resident'):
            learner_args=shlex.split(learn)
            self.work('learner',f"stage{s['index']}-train",learner_args)
            if 'learner' not in self.state['jobs']:
                keeper=self.env(shlex.join([PYTHON,str(self.tools/'allocation_worker.py'),str(self.root),'learner']))
                self.state['jobs']['learner']=self.submit('curriculum-learn',keeper,self.cfg['learner_resources'])
            for role,resources,collectors,servers in self.cfg['actors']:
                self.work(role,f"stage{s['index']}-train",[PYTHON,'ppo.py','act',str(run),'--collectors',str(collectors),'--servers',str(servers)])
                if role not in self.state['jobs']:
                    keeper=self.env(shlex.join([PYTHON,str(self.tools/'allocation_worker.py'),str(self.root),role]))
                    self.state['jobs'][role]=self.submit('curriculum-'+role,keeper,resources)
            self.save();return
        if 'learner' not in self.state['jobs']:
            self.state['jobs']['learner']=self.submit('curriculum-learn',command,self.cfg['learner_resources'])
        for key,resources,collectors,servers in self.cfg['actors']:
            command=self.env('cd '+shlex.quote(str(self.source))+' && '+shlex.join(
                [PYTHON,'ppo.py','act',str(run),'--collectors',str(collectors),'--servers',str(servers)]))
            if key not in self.state['jobs']:
                self.state['jobs'][key]=self.submit('curriculum-'+key,command,resources)
        self.save()

    def start_arena(self,kind):
        if not self.cfg.get('resident'):self.cancel()
        s=self.state['stage'];run=Path(s['run'])
        path=self.root/'evaluations'/f"stage-{s['index']:02d}-u{s['policy_updates']:03d}-{kind}"
        path.mkdir(parents=True,exist_ok=True)
        candidate=run/'pool'/f"v{s['cap']:06d}.pt"
        if kind=='gate':
            meta=json.loads((self.root/'data/decks.json').read_text())
            pool=json.loads((self.root/'data/eval.json').read_text())
            plan=make_plan(meta,pool,s['target'],candidate,s['evaluation_reference']['policy'],self.root/'init/policy-v320.pt')
            put(path/'plan.json',plan);decks=path/'plan.json';expected=len(plan['tasks'])
        else:
            candidate=Path(self.state['accepted']['policy'])
            shutil.copy2(self.root/'data/decks.json',path/'decks.json');decks=path/'decks.json'
            n=len(json.loads(decks.read_text())['decks']);expected=n*(n-1)//2*20
        if self.cfg.get('resident'):
            if kind!='gate':
                from common_deck_arena import tasks,match
                meta=json.loads(decks.read_text())
                plan=dict(tasks=[dict(**t,pair=match(meta,t),candidate=str(candidate),opponent=str(candidate)) for t in tasks(meta)])
                put(path/'plan.json',plan)
            slots={};offset=0
            for role,resources,count,servers in self.cfg['actors']:
                slots[self.state['jobs'][role]['id']]=dict(offset=offset,count=count);offset+=count
            put(run/'paired-eval.json',dict(active=True,plan=str(path/'plan.json'),results=str(path/'results'),
                                          slots=slots,collectors=offset,fraction=.5 if kind=='gate' else 1.))
            # Preserve the frozen policies in each decision server's cache.
            paths={str(candidate),str(self.root/'init/policy-v320.pt'),s['evaluation_reference']['policy']}
            put(run/'eval.json',dict(matches=[dict(id=f'keep-{i}',a=['paired',p],b=['paired',p],games=1) for i,p in enumerate(paths)]))
        self.state.update(phase=kind,arena=dict(root=str(path),kind=kind,expected=expected,candidate=str(candidate)))
        if not self.cfg.get('resident'):self.start_arena_jobs()
        self.save()

    def start_arena_jobs(self):
        arena=self.state['arena'];path=Path(arena['root']);candidate=Path(arena['candidate'])
        decks=path/('plan.json' if arena['kind']=='gate' else 'decks.json')
        for shard,resources in enumerate(self.cfg['evaluation_resources']):
            args=[PYTHON,str(self.tools/'common_deck_arena.py'),'--source',str(self.source),
                  '--decks',str(decks),'--out',str(path/'results'),'--checkpoint',str(candidate),
                  '--engine',self.cfg['engine'],'--engine-lib',self.cfg['engine_lib'],
                  '--engine-assets',self.cfg['engine_assets'],'--shards','3','--shard',str(shard),'--workers','60']
            if str(shard) not in self.state['jobs']:
                self.state['jobs'][str(shard)]=self.submit('curriculum-eval-'+str(shard),self.env(shlex.join(args)),resources)

    def start_summary(self):
        if not self.cfg.get('resident'):self.cancel()
        else:self.disable_inline()
        a=self.state['arena'];p=Path(a['root'])
        script='curriculum_evaluation.py' if a['kind']=='gate' else 'summarize_common_decks.py'
        command=self.env(shlex.join([PYTHON,str(self.tools/script),str(p)]))
        self.state['phase']=a['kind']+'_summary'
        if self.cfg.get('resident'):
            self.work('actors',p.name+'-summary',[PYTHON,str(self.tools/script),str(p)],self.tools,aux=True)
            self.save();return
        self.state['jobs']['summary']=self.submit('curriculum-summary',command,self.cfg['summary_resources']);self.save()

    def accepted(self,summary):
        s=self.state['stage'];run=Path(s['run']);cap=s['cap']
        value=summary.get('target_score')
        if summary['general_score']<self.cfg['general_floor']:return
        if s['mode']=='target' and (summary['target_gain']<=0 or (s['best_target'] is not None and value<s['best_target'])):return
        s['best_target']=value
        self.state['accepted']=dict(name=f"stage{s['index']}-v{cap:06d}",policy=str(run/'pool'/f'v{cap:06d}.pt'),
                                    critic=str(run/'ckpt'/f'v{cap:06d}.pt'))
        self.event('accepted '+self.state['accepted']['name'])

    def fallback(self,reason):
        self.cancel();self.alert(reason+'; preserving candidate and resuming the two previously authorized specialists')
        for name in ('gsram-r6e320-1029','giantgy-r6e320-1029'):
            run=self.root.parent/name
            with (run/'loop.out').open('a') as log:
                p=subprocess.Popen([PYTHON,str(self.cfg['old_source']+'/loop.py'),str(run)],stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            (run/'loop.pid').write_text(str(p.pid)+'\n')
        self.state.update(phase='fallback-specialists',stopped=reason);self.save()

    def handle_gate(self,summary):
        s=self.state['stage']
        if not self.cfg.get('resident'):self.cancel()
        self.state['history'].append(dict(stage=s['index'],policy_updates=s['policy_updates'],**summary))
        self.accepted(summary)
        if s['policy_updates']<100:
            s['policy_updates']+=20;s['cap']=s['warmup']+s['policy_updates'];self.state['phase']='train';self.start_training();return
        action=decision(s['mode'],summary,self.cfg['general_floor'])
        if action=='recover':
            self.alert('General score below47% at100updates; starting100-update broad recovery from the final candidate')
            run=Path(s['run']);cap=s['cap']
            self.new_stage('recovery',dict(name=f"candidate-stage{s['index']}",policy=str(run/'pool'/f'v{cap:06d}.pt'),
                                           critic=str(run/'ckpt'/f'v{cap:06d}.pt')))
        elif action=='fallback':self.fallback('No target improvement after100updates' if s['mode']=='target' else 'Broad recovery still regresses')
        else:self.start_arena('rank')

    def prune(self):
        s=self.state['stage'];run=Path(s['run']);keep={s['warmup']+k for k in (20,40,60,80,100)}
        keep.update(int(p.stem[1:]) for p in sorted((run/'pool').glob('v*.pt'))[-3:])
        for folder in ('pool','ckpt'):
            for p in (run/folder).glob('v*.pt'):
                if int(p.stem[1:]) not in keep:p.unlink()

    def step(self):
        if self.state['stopped']:return False
        if time.time()-self.state['started']>self.cfg['max_hours']*3600:
            self.cancel();self.state.update(stopped='24-hour curriculum limit',phase='stopped');self.save();return False
        phase=self.state['phase']
        if phase=='new':self.new_stage();return True
        s=self.state['stage']
        if phase=='train':
            if len(self.state['jobs'])<1+len(self.cfg['actors']):
                self.start_training();return True
            metrics=Path(s['run'])/'metrics.jsonl'
            if metrics.exists():
                lines=metrics.read_text().splitlines()
                if lines and json.loads(lines[-1])['version']>=s['cap']:
                    self.start_arena('gate');return True
                if time.time()-metrics.stat().st_mtime>3600:
                    if self.cfg.get('resident'):
                        heartbeat=self.root/'allocations/learner/heartbeat.json'
                        if heartbeat.exists():
                            h=json.loads(heartbeat.read_text())
                            since=max(metrics.stat().st_mtime,h.get('child_started') or h['at'])
                            if h.get('child') and time.time()-since>3600:
                                task=self.root/'allocations/learner/task.json';spec=json.loads(task.read_text())
                                spec['id']=spec['id']+'-retry';put(task,spec)
                                self.alert('Learner stalled; restarting child inside retained allocation')
                        # A queued allocation is not a stalled learner. Never scancel it.
                        self.ensure();self.save();return True
                    self.cancel();self.alert('Learner stalled >60 minutes; restarting from latest checkpoint');self.start_training();return True
                self.prune()
        elif phase in ('gate','rank'):
            if not self.cfg.get('resident') and len(self.state['jobs'])<len(self.cfg['evaluation_resources']):
                self.start_arena_jobs();self.save();return True
            p=Path(self.state['arena']['root']);count=sum(sum(1 for _ in f.open()) for f in (p/'results').glob('games-*.jsonl'))
            if count==self.state['arena']['expected']:self.start_summary();return True
        elif phase.endswith('_summary'):
            p=Path(self.state['arena']['root']);summary=p/'summary.json'
            if summary.exists():
                data=json.loads(summary.read_text())
                if phase=='gate_summary':self.handle_gate(data)
                else:
                    self.state['ranking']=str(summary);self.new_stage()
                return True
            if not self.state['jobs']:
                self.start_summary();return True
        self.ensure();self.save();return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    lock=(a.root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=Curriculum(a.root)
    while True:
        try:
            if not c.step():break
            c.state['errors']=0
        except BaseException as e:
            import traceback
            traceback.print_exc();c.state['errors']=c.state.get('errors',0)+1;c.alert(str(e));c.save()
            if c.state['errors']>=3:
                c.fallback('Controller failed three consecutive checks');break
        time.sleep(30)


if __name__=='__main__':main()
