import argparse
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.history_compare import continuation_core as core
from experiments.history_compare.continuation_core import original

ROOT = Path(__file__).resolve().parents[1] / 'tmp'


class IdOriginal(original.Games):
    def _block(self, entries, rng):
        for entry in entries:
            yield entry['id']


class IdOffset(core.OffsetGames):
    def _block(self, entries, rng):
        for entry in entries:
            yield entry['id']


class EntryOriginal(original.Games):
    def _block(self, entries, rng):
        for entry in entries:
            yield (entry['file'], entry['owner'])


class EntryOffset(core.OffsetGames):
    def _block(self, entries, rng):
        for entry in entries:
            yield (entry['file'], entry['owner'])


def unchanged(items):
    return items


def gloo_resume_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(51)
        model=torch.nn.Linear(3,2)
        wrapped=torch.nn.parallel.DistributedDataParallel(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,fused=False)
        x=torch.arange(12,dtype=torch.float32).reshape(4,3)/17+rank
        def update(network,opt):
            opt.zero_grad(set_to_none=True)
            network(x).square().mean().backward()
            opt.step()
        for _ in range(2):
            update(wrapped,optimizer)
        path=Path(directory)/f'gloo-rank{rank}.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),index=2,
                        settings=dict(lr=.001,weight_decay=.01)),path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        restored=torch.nn.Linear(3,2)
        restored.load_state_dict(saved['model'])
        resumed=core.restore_optimizer(restored,saved,fused=False)
        resumed_ddp=torch.nn.parallel.DistributedDataParallel(restored)
        update(wrapped,optimizer)
        update(resumed_ddp,resumed)
        for p,q in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
    finally:
        dist.destroy_process_group()


class ContinuationTests(unittest.TestCase):
    def test_real_cpu_gloo_ddp_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            mp.spawn(gloo_resume_worker,args=(str(Path(directory)/'rendezvous'),directory),nprocs=2,join=True)

    def test_lr_is_absolute_not_rewarmed(self):
        settings = dict(lr=.0003, warmup=2000, final_lr=.1)
        for index in (0, 1, 128, 3199, 3200, 14187):
            expected = .0003 * (1 - .9 * (30000 + index) / 42564)
            self.assertAlmostEqual(core.absolute_lr(settings, index), expected, places=17)
        self.assertAlmostEqual(core.absolute_lr(settings, 0), .00010969833662249787)

    def test_33200_milestone(self):
        self.assertTrue(core.milestone(3200, 14188))
        self.assertTrue(core.milestone(3200, 3200))
        self.assertFalse(core.milestone(3199, 14188))

    def test_even_odd_offset_and_tail(self):
        entries = [dict(id=i, decisions=(i * 73) % 1280) for i in range(16384)]
        for rank in (0, 7, 8, 15):
            batches = original.rank_order(entries, 30000, rank)
            original_loader = torch.utils.data.DataLoader(IdOriginal('.', entries, 1280, 30000, rank=rank, world=16),
                                                          batch_size=4, num_workers=2, multiprocessing_context='spawn')
            expected = [v.tolist() for v in original_loader]
            for offset in (0, 1, 2, 7, 16, 127, len(batches)-1, len(batches)):
                end = min(len(batches), offset + 5)
                loader = core.offset_loader('.', batches, offset, end, rank, dataset_class=IdOffset,
                                             collate_fn=unchanged, pin_memory=False)
                self.assertEqual(list(loader), expected[offset:end])

    def test_resume_cpu_adam_matches_uninterrupted(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, fused=False)
        inputs = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 13
        def update(m, opt):
            opt.zero_grad(set_to_none=True)
            m(inputs).square().mean().backward()
            opt.step()
        for _ in range(3):
            update(model, optimizer)
        saved = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                     settings=dict(lr=.001, weight_decay=.01), index=3)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'toy.pt'
            torch.save(saved, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
        restored = torch.nn.Linear(3, 2)
        restored.load_state_dict(loaded['model'])
        resumed_optimizer = core.restore_optimizer(restored, loaded, fused=False)
        torch.manual_seed(891)
        update(model, optimizer)
        torch.manual_seed(251)
        update(restored, resumed_optimizer)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(optimizer.state.values(), resumed_optimizer.state.values()):
            for name in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_wrong_optimizer_shape_rejected(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), fused=False)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        saved = dict(optimizer=optimizer.state_dict(), settings=dict(lr=.001,weight_decay=.01), index=1)
        saved['optimizer']['state'][0]['exp_avg'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            core.restore_optimizer(model, saved, fused=False)

    def test_matching_33200_contract_and_rejection(self):
        from experiments.history_compare.continuation_compare import compare
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory)
            common=dict(seed=30000,gbs=64,mbs=4,workers=2,real_ddp=True,optimizer_reset=False,
                origin_manifest_sha256='manifest',origin_checkpoint_sha256='best30k',order_hashes=['same']*16,
                settings={},config={},source_sha256={},compile='unchanged',bf16=True,gradient_checkpointing=False,
                validation_views=[dict(id=i) for i in range(32)],core_sha256='core',continuation_sha256='runner',index=3000)
            for arm in ('baseline','split'):
                out=root/arm
                out.mkdir()
                (out/'protocol.json').write_text(json.dumps(dict(**common,arm=arm)))
                (out/'validation-03200.json').write_text(json.dumps(dict(index=3200,step=33200,
                    entries=common['validation_views'],metrics=dict(tile_weighted=2.5),tile_labels=975)))
                (out/'checkpoint-03200.pt').touch()
            self.assertEqual(compare(root/'baseline',root/'split')['step'],33200)
            wrong=dict(**common,arm='split')
            wrong['index']=2999
            (root/'split/protocol.json').write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):
                compare(root/'baseline',root/'split')

    def test_wrong_arm_oldfinal_and_missing_optimizer_rejected(self):
        for saved in (dict(arm='baseline', phase='direct'), dict(arm='split', phase='direct', step=42564),
                      dict(arm='split', phase='direct', real_ddp=True, world_size=16, index=128,step=30128)):
            with self.assertRaises(ValueError):
                core.validate_checkpoint(saved, 'split', {})

    def test_budget_rollback_work_is_charged(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root/'finished.json').write_text(json.dumps(dict(index=3000,seconds=7200)))
            (root/'train.jsonl').write_text(json.dumps(dict(index=128,seconds=120))+'\n')
            previous, _ = core.elapsed_credit(dict(index=128,arm='split'),root/'old.pt',root)
            self.assertEqual(previous,7200)
            with self.assertRaises(ValueError):
                core.elapsed_credit(dict(index=128,arm='split'),root/'old.pt',root, explicit=100)
            self.assertEqual(core.time_status(previous,64800,False)['status'],'bounded_time_limit')
            self.assertFalse(core.time_status(previous,64800,False)['numerical_failure'])

    def test_continuation_budget_does_not_reset(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root/'finished.json').write_text(json.dumps(dict(cumulative_seconds=15000)))
            saved=dict(continuation=dict(cumulative_seconds=14000))
            self.assertEqual(core.elapsed_credit(saved,root/'x.pt',root)[0],15000)

    def test_dropout_is_detected(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout=torch.nn.Dropout(.2)
            def forward(self,x):
                return self.dropout(x)
            def tiles(self,x):
                return x
        self.assertFalse(core.rng_audit(Model())['no_stochastic_training_path_found'])

    def test_no_unsafe_loads_or_gpu_test_calls(self):
        for source in (original.ROOT / 'continuation.py', original.ROOT / 'continuation_core.py', Path(__file__)):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
                    if node.func.value.id=='torch' and node.func.attr=='load':
                        self.assertTrue(any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in node.keywords))

    def test_order_corruption_rejected(self):
        entries=[dict(id=i,decisions=i%1280) for i in range(4096)]
        batches=original.rank_order(entries,30000,0)
        checksum=core.canonical_hash(batches)
        doc=dict(seed=30000,rank=0,sha256=checksum,batches=batches)
        receipt=dict(order_sha256=checksum)
        core.verify_order(doc,entries,0,receipt)
        doc['batches'][0][0]['id']=-1
        with self.assertRaises(ValueError):
            core.verify_order(doc,entries,0,receipt)

    def test_offset_loader_does_not_consume_main_torch_rng(self):
        entries=[dict(id=i,decisions=i%1280) for i in range(4096)]
        batches=original.rank_order(entries,30000,0)
        torch.manual_seed(79)
        before=torch.get_rng_state().clone()
        loader=core.offset_loader('.',batches,1,3,0,dataset_class=IdOffset,collate_fn=unchanged,pin_memory=False)
        list(loader)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


def real_audit(args):
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    protocol = json.loads((args.origin/'protocol.json').read_text())
    core.validate_checkpoint(saved,args.arm,protocol)
    model = original.build(original.Config(**saved['config']),original.C.effect_matrix(original.C.Vocab(saved['vocab'])))
    if args.arm=='split' and not model.config.split_query:
        original.install(model)
    model.load_state_dict(saved['model'],strict=True)
    optimizer=core.restore_optimizer(model,saved,fused=False)
    for identifier,parameter in zip(saved['optimizer']['param_groups'][0]['params'], model.parameters()):
        if identifier in saved['optimizer']['state']:
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[parameter][key],saved['optimizer']['state'][identifier][key],rtol=0,atol=0)
    audit=core.rng_audit(model)
    if not audit['no_stochastic_training_path_found']:
        raise AssertionError(audit)
    manifest=json.loads((args.data/'manifest.json').read_text())
    entries=original.training_entries(manifest,saved['settings'])
    ranks=json.loads((args.origin/'ranks.json').read_text())
    results=[]
    for rank in args.ranks:
        document=json.loads((args.order_dir/f'order-rank-{rank:02d}.json').read_text())
        batches=core.verify_order(document,entries,rank,ranks[rank])
        # Exercise the original DataLoader, not just a second hand-written ordering formula.
        loader=torch.utils.data.DataLoader(EntryOriginal('.',entries,1280,30000,rank=rank,world=16),
                                           batch_size=4,num_workers=2,collate_fn=unchanged,in_order=True)
        checked=0
        for i,micro in enumerate(loader):
            if i>=len(batches):
                break
            if micro != [(e['file'],e['owner']) for e in batches[i]]:
                raise AssertionError(f'Original loader differs at rank{rank} batch{i}')
            checked+=1
        offsets=sorted(set((0,1,127,128,249,3199,3200,3201,len(batches)-1,saved['index'])))
        for offset in offsets:
            end=min(len(batches),offset+3)
            resumed=core.offset_loader('.',batches,offset,end,rank,dataset_class=EntryOffset,collate_fn=unchanged,pin_memory=False)
            if list(resumed) != [[(e['file'],e['owner']) for e in micro] for micro in batches[offset:end]]:
                raise AssertionError(f'Offset loader differs at rank{rank} offset{offset}')
        results.append(dict(rank=rank,all_original_batches_checked=checked,offsets_checked=offsets,order_sha256=document['sha256']))
        print(f'CONTINUATION_CPU_ORDER_OK rank={rank} batches={checked}',flush=True)
    tensor_cases=[]
    for rank,offset in ((0,128),(15,3199)):
        document=json.loads((args.order_dir/f'order-rank-{rank:02d}.json').read_text())
        batches=document['batches']
        cannon=original.C.Vocab(saved['vocab'])(original.CANNON_BARRAGE)
        reference=original.Games(args.data,[],1280,30000,rank=rank,world=16,cannon_index=cannon)
        rng=random.Random(30000*1000+rank*2+offset%2)
        expected=original.collate(list(reference._block(batches[offset],rng)))
        loader=core.offset_loader(args.data,batches,offset,offset+1,rank,cannon,pin_memory=False)
        measured=list(loader)[0]
        for a,b in zip(expected,measured):
            for key in a:
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        tensor_cases.append(dict(rank=rank,offset=offset,all_collated_input_and_label_tensors_equal=True))
        del expected, measured, loader
    # Full-model CPU forward/loss RNG and output check on one complete real validation view.
    selected=protocol['t0_views'][0]
    cannon=original.C.Vocab(saved['vocab'])(original.CANNON_BARRAGE)
    xs,ts=original.collate([original.item(args.data,selected,cannon)])
    model.train()
    torch.manual_seed(9823)
    initial=torch.get_rng_state().clone()
    with torch.no_grad():
        loss1,m1=original.loss_and_metrics(model,xs,ts,saved['settings']['value_coef'],saved['settings']['next_play'])
    if not torch.equal(initial,torch.get_rng_state()):
        raise AssertionError('Real train-mode CPU loss consumed torch RNG')
    torch.manual_seed(1411)
    with torch.no_grad():
        loss2,m2=original.loss_and_metrics(model,xs,ts,saved['settings']['value_coef'],saved['settings']['next_play'])
    torch.testing.assert_close(loss1,loss2,rtol=0,atol=0)
    if m1!=m2:
        raise AssertionError('Real train-mode CPU metrics depend on changed RNG seed')
    return dict(checkpoint=str(args.checkpoint),checkpoint_sha256=original.digest(args.checkpoint),arm=args.arm,index=saved['index'],
                model_tensors_restored=len(saved['model']),optimizer_parameter_states=len(optimizer.state),
                rng_present_in_checkpoint='rng_by_rank' in saved,rng_audit=audit,actual_train_mode_cpu_rng_unchanged=True,
                actual_cpu_loss_bitwise_equal_across_rng_seeds=True,cpu_view=selected,order_results=results,
                actual_collated_tensor_cases=tensor_cases,restored_AdamW_moments_and_steps_exact=True,
                gpu_work_started=False,gpu_resume_equivalence_tested=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--real',action='store_true')
    parser.add_argument('--arm',choices=['baseline','split'],default='split')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--origin',type=Path)
    parser.add_argument('--order-dir',type=Path)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--ranks',type=int,nargs='+',default=list(range(8)))
    parser.add_argument('--out',type=Path,default=ROOT/'continuation-tests.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU tests must not initialize CUDA')
    if args.real:
        result=real_audit(args)
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContinuationTests)
        tests=unittest.TextTestRunner(verbosity=2).run(suite)
        result=dict(tests=tests.testsRun,successful=tests.wasSuccessful(),failures=len(tests.failures),errors=len(tests.errors))
    if torch.cuda.is_initialized():
        raise AssertionError('CPU audit unexpectedly initialized CUDA')
    original.write(args.out,result)
    print(json.dumps(result,indent=2),flush=True)
    if not args.real and not result['successful']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
