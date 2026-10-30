"""Curriculum evaluation pairing and automatic 100-update handoffs."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from curriculum_evaluation import make_plan
from deck_curriculum import Curriculum, decision


def test_general_panel_reverses_decks_and_target_reference_uses_identical_matches():
    decks=[dict(index=i,deck=[dict(d=i)],usage=.01) for i in range(3)]
    meta=dict(decks=decks,tower=[dict(d=99)])
    pool=[dict(decks=[[dict(d=1)],[dict(d=2)]],sc=[[],[]],kt=[16,16],exp=[16,16])]
    plan=make_plan(meta,pool,1,'candidate','accepted','v320')
    general=[t for t in plan['tasks'] if t['kind']=='general']
    assert len(general)==1000
    for a,b in zip(general[::2],general[1::2]):
        assert a['pair_index']==b['pair_index'] and a['seed']==b['seed']
        assert a['a_owner']==0 and b['a_owner']==1
        assert a['pair']['decks']==b['pair']['decks'][::-1]
    target=[t for t in plan['tasks'] if t['kind']=='target']
    reference=[t for t in plan['tasks'] if t['kind']=='target_reference']
    assert len(target)==len(reference)==40
    for a,b in zip(target,reference):
        assert a['pair']==b['pair'] and a['seed']==b['seed'] and a['a_owner']==b['a_owner']
        assert a['pair']['decks'][a['a_owner']]==[dict(d=1)]
        assert a['candidate']=='candidate' and b['candidate']=='accepted'
        assert a['opponent']==b['opponent']=='v320'
    recovery=make_plan(meta,pool,None,'recovery','last-accepted','v320')
    assert len(recovery['tasks'])==1000
    assert {t['opponent'] for t in recovery['tasks']}=={'last-accepted'}


def test_decision_allows_intermediate_regression_but_routes_final_blocks():
    assert decision('target',dict(general_score=.48,target_gain=.10))=='advance'
    assert decision('target',dict(general_score=.46,target_gain=.10))=='recover'
    assert decision('target',dict(general_score=.55,target_gain=-.01))=='fallback'
    assert decision('recovery',dict(general_score=.50,target_gain=None))=='advance'
    assert decision('recovery',dict(general_score=.46,target_gain=None))=='fallback'


def test_100_update_recovery_starts_from_candidate_but_evaluates_against_accepted(tmp_path,monkeypatch):
    root=tmp_path; (root/'data').mkdir();(root/'bc').mkdir();(root/'stages').mkdir()
    (root/'curriculum.json').write_text(json.dumps(dict(source='/source',general_floor=.47,general_bc='/general')))
    (root/'settings-template.json').write_text('{}')
    (root/'data/decks.json').write_text(json.dumps(dict(decks=[dict(deck=[dict(d=1)])])))
    (root/'data/initial-ranking.json').write_text(json.dumps(dict(ranking=[dict(index=0)])))
    c=Curriculum(root);monkeypatch.setattr(c,'cancel',lambda:None);monkeypatch.setattr(c,'save',lambda:None)
    monkeypatch.setattr(c,'start_training',lambda:None)
    c.state.update(stage_index=1,stage=dict(index=1,mode='target',run=str(root/'old'),cap=103,
                   policy_updates=100,best_target=None),accepted=dict(name='accepted',policy='/accepted',critic='/critic'))
    c.handle_gate(dict(general_score=.40,target_score=.40,target_reference_score=.20,target_gain=.20))
    s=c.state['stage'];assert s['mode']=='recovery' and s['cap']==20
    assert s['initial']['policy'].endswith('old/pool/v000103.pt')
    assert s['evaluation_reference']['policy']=='/accepted'


def test_intermediate_regression_continues_candidate_without_rollback(tmp_path,monkeypatch):
    (tmp_path/'curriculum.json').write_text(json.dumps(dict(source='/source',general_floor=.47)))
    c=Curriculum(tmp_path);monkeypatch.setattr(c,'cancel',lambda:None);monkeypatch.setattr(c,'start_training',lambda:None)
    c.state['stage']=dict(index=1,mode='target',run=str(tmp_path),cap=23,warmup=3,policy_updates=20,best_target=None)
    c.handle_gate(dict(general_score=.40,target_score=.40,target_reference_score=.20,target_gain=.20))
    assert c.state['stage']['cap']==43 and c.state['phase']=='train'
    assert c.state['accepted']['name']=='general-v320'


def test_inline_gate_does_not_release_or_replace_training_jobs(tmp_path,monkeypatch):
    (tmp_path/'data').mkdir();(tmp_path/'evaluations').mkdir()
    (tmp_path/'curriculum.json').write_text(json.dumps(dict(source='/source',resident=True,
          actors=[['actors',[],2,1],['actors-stu',[],2,1]])))
    meta=dict(decks=[dict(index=i,deck=[dict(d=i)],usage=.01) for i in range(3)],tower=[])
    (tmp_path/'data/decks.json').write_text(json.dumps(meta))
    pool=[dict(decks=[[dict(d=1)],[dict(d=2)]],sc=[[],[]],kt=[16,16],exp=[16,16])]
    (tmp_path/'data/eval.json').write_text(json.dumps(pool))
    c=Curriculum(tmp_path);run=tmp_path/'run';run.mkdir()
    c.state['stage']=dict(index=1,mode='target',run=str(run),target=1,cap=23,policy_updates=20,
                         evaluation_reference=dict(policy='/accepted'))
    jobs={'learner':dict(id='100'),'actors':dict(id='101'),'actors-stu':dict(id='102')}
    c.state['jobs']=jobs.copy()
    monkeypatch.setattr(c,'cancel',lambda:(_ for _ in ()).throw(AssertionError('released allocations')))
    monkeypatch.setattr(c,'submit',lambda *a:(_ for _ in ()).throw(AssertionError('submitted another job')))
    monkeypatch.setattr(c,'save',lambda:None)
    c.start_arena('gate');assert c.state['jobs']==jobs
    spec=json.loads((run/'paired-eval.json').read_text());assert spec['collectors']==4 and spec['active']
    c.start_summary();assert c.state['jobs']==jobs
    assert not json.loads((run/'paired-eval.json').read_text())['active']
    assert (tmp_path/'allocations/actors/aux.json').exists()


def test_inline_task_assignment_is_complete_disjoint_and_resumes(tmp_path,monkeypatch):
    import random
    from types import SimpleNamespace
    from cr.paired_evaluation import AssignedEvaluations
    tasks=[dict(id=i,candidate='/a',opponent='/b') for i in range(24)]
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(tasks=tasks)))
    (tmp_path/'paired-eval.json').write_text(json.dumps(dict(active=True,fraction=1,plan=str(plan),
           results=str(tmp_path/'results'),slots={'job':dict(offset=0,count=4)},collectors=4)))
    monkeypatch.setenv('SLURM_JOB_ID','job');seen=[]
    for i in range(4):
        player=AssignedEvaluations(tmp_path,i)
        while (task:=player.next(random.Random(1))) is not None:
            seen.append(task['id']);player.finish(task,1.,SimpleNamespace(crowns=[1,0],tick=100),[0,0])
        restarted=AssignedEvaluations(tmp_path,i)
        assert restarted.next(random.Random(1)) is None
    assert sorted(seen)==list(range(24))
