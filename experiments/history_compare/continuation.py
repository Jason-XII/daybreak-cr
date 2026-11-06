import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
import traceback

sys.dont_write_bytecode = True
import torch
import torch.distributed as dist
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.history_compare import continuation_core as core
from experiments.history_compare.continuation_core import original


def capture_rng(rank):
    return dict(rank=rank, python=random.getstate(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state())


def restore_rng(saved, rank):
    states = saved.get('rng_by_rank')
    if states is None:
        return False
    if len(states) != 16 or states[rank]['rank'] != rank:
        raise ValueError('RNG rank layout mismatch')
    random.setstate(states[rank]['python'])
    torch.set_rng_state(states[rank]['torch_cpu'])
    torch.cuda.set_rng_state(states[rank]['torch_cuda'])
    return True


def save_boundary(model, optimizer, saved, args, index, rank, previous, started, lineage):
    # Save all rank RNG states after boundary validation, immediately before the next training update.
    rng = [None] * 16
    dist.all_gather_object(rng, capture_rng(rank))
    if rank == 0:
        elapsed = previous + time.monotonic() - started
        path = args.out / f'checkpoint-{index:05d}.pt'
        temporary = path.with_suffix('.partial')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=saved['config'],
                        vocab=saved['vocab'], settings=saved['settings'], index=index, step=30000 + index,
                        arm=args.arm, phase='direct', real_ddp=True, world_size=16, fresh_optimizer=True,
                        parameter_names=[n for n, _ in model.named_parameters()], rng_by_rank=rng,
                        continuation=dict(**lineage, cumulative_seconds=elapsed, optimizer_reset=False,
                                          seed=30000, next_batch_index=index)), temporary)
        temporary.replace(path)
        original.write(args.out / 'latest-checkpoint.json', dict(path=str(path), index=index, step=30000 + index,
                       cumulative_seconds=previous + time.monotonic() - started))
    dist.barrier()


def preflight(args, rank):
    checkpoint = args.checkpoint.resolve()
    parent = args.parent.resolve()
    if checkpoint.parent != parent:
        raise ValueError('Checkpoint must belong to this isolated parent arm directory')
    terminal_path = parent / 'finished.json'
    if not terminal_path.exists():
        raise ValueError('Parent arm has no terminal receipt; do not overlap an active or queued run')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    terminal = json.loads(terminal_path.read_text())
    if terminal.get('status') not in ('complete', 'bounded_time_limit') or terminal.get('index', -1) < saved.get('index', 0):
        raise ValueError('Parent terminal receipt is incompatible with this checkpoint')
    protocol = json.loads((args.origin / 'protocol.json').read_text())
    index = core.validate_checkpoint(saved, args.arm, protocol)
    if index > 3200 and args.require_33200:
        raise ValueError('Cannot recover past33200 from a later state; choose this arm checkpoint at/before3200')
    if not max(index, 3200) <= args.target_index <= protocol['steps']:
        raise ValueError('Target must cover3200 and stay in the recorded original pass')
    ranks = json.loads((args.origin / 'ranks.json').read_text())
    receipt = ranks[rank]
    for name, checksum in receipt['source_sha256'].items():
        if original.digest(original.SOURCE / name) != checksum:
            raise ValueError(f'Frozen source drift: {name}')
    for name, checksum in receipt['implementation_sha256'].items():
        if original.digest(original.SOURCE / name) != checksum:
            raise ValueError(f'Original implementation drift: {name}')
    manifest_path = args.data / 'manifest.json'
    if original.digest(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('Manifest differs from origin')
    manifest = json.loads(manifest_path.read_text())
    entries = original.training_entries(manifest, saved['settings'])
    document = json.loads((args.order_dir / f'order-rank-{rank:02d}.json').read_text())
    batches = core.verify_order(document, entries, rank, receipt)
    if args.target_index > len(batches):
        raise ValueError('Target extends beyond recorded batches')
    previous, timing = core.elapsed_credit(saved, checkpoint, parent, args.charged_seconds)
    torch.manual_seed(saved['settings']['seed'])
    model = original.build(original.Config(**saved['config']), original.C.effect_matrix(original.C.Vocab(saved['vocab'])))
    if args.arm == 'split' and not model.config.split_query:
        original.install(model)
    model.load_state_dict(saved['model'], strict=True)
    audit = core.rng_audit(model)
    if not audit['no_stochastic_training_path_found']:
        raise ValueError('Unexpected stochastic path; old checkpoint has no reliable RNG continuation contract')
    return saved, protocol, index, batches, previous, timing, model, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['baseline', 'split'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--origin', type=Path, required=True)
    parser.add_argument('--order-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--target-index', type=int, default=3200)
    parser.add_argument('--total-seconds', type=float, default=72000)
    parser.add_argument('--charged-seconds', type=float)
    parser.add_argument('--require-33200', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    rank, local, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    if world != 16 or not 0 < args.total_seconds <= 72000:
        raise ValueError('Require real16 ranks and cumulative budget <=20h')
    if args.out.resolve() in (args.parent.resolve(), args.origin.resolve()) or args.out.resolve().exists():
        raise ValueError('Use a new isolated continuation output directory')
    torch.set_num_threads(2)
    started = time.monotonic()
    output_owned = False
    try:
        saved, protocol, index, batches, previous, timing, model, audit = preflight(args, rank)
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True
        dist.init_process_group('nccl', timeout=timedelta(minutes=10), device_id=torch.device('cuda', local))
        creation_error = None
        if local == 0:
            try:
                args.out.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                creation_error = repr(error)
        creation_errors = [None] * world
        dist.all_gather_object(creation_errors, creation_error)
        if any(creation_errors):
            raise ValueError('Refusing existing/unwritable output: ' + repr(creation_errors))
        output_owned = True
        checks = [None] * world
        signature = dict(checkpoint_sha256=original.digest(args.checkpoint), index=index, target=args.target_index,
                         arm=args.arm, previous=previous, total_seconds=args.total_seconds,
                         continuation_sha256=original.digest(__file__), core_sha256=original.digest(ROOT / 'continuation_core.py'),
                         origin_protocol_sha256=original.digest(args.origin/'protocol.json'),
                         origin_ranks_sha256=original.digest(args.origin/'ranks.json'))
        dist.all_gather_object(checks, signature)
        comparable = [{k: v for k, v in row.items() if k != 'previous'} for row in checks]
        if any(row != comparable[0] for row in comparable):
            raise ValueError('Cross-node continuation identity mismatch')
        previous = max(row['previous'] for row in checks)
        if previous + time.monotonic() - started >= args.total_seconds:
            if rank == 0:
                result = core.time_status(previous, time.monotonic()-started, False, args.total_seconds)
                result.update(index=index,step=30000+index,target_index=args.target_index,
                              checkpoint=str(args.checkpoint),optimizer_reset=False,distillation_started=False,
                              reason='Budget exhausted before any optimizer update')
                original.write(args.out / 'finished.json', result)
            return
        model.cuda().train()
        for block in [*model.encoder, *model.tile_blocks, *model.lstm.blocks]:
            block.compile(dynamic=True)
        optimizer = core.restore_optimizer(model, saved, fused=True)
        objective = nn.parallel.DistributedDataParallel(original.Objective(model, saved['settings']['value_coef'], saved['settings']['next_play']),
            device_ids=[local], find_unused_parameters=True)
        lineage = dict(origin=str(args.origin), parent=str(args.parent), checkpoint_sha256=signature['checkpoint_sha256'],
                       resumed_index=index, source='own_new_run_model_and_AdamW', order_hashes=[r['order_sha256'] for r in json.loads((args.origin/'ranks.json').read_text())])
        rng_restored = restore_rng(saved, rank)
        if rank == 0:
            original.write(args.out / 'protocol.json', dict(**signature, seed=30000, gbs=64, mbs=4, workers=2,
                real_ddp=True, optimizer_reset=False, cumulative_limit_seconds=args.total_seconds,
                charged_prior_seconds=previous, timing_evidence=timing, rng_restored=rng_restored, rng_audit=audit,
                exact_gpu_replay_verified=False, required_snapshot_step=33200, validation_views=protocol['t0_views'],
                origin_manifest_sha256=protocol['manifest_sha256'], origin_checkpoint_sha256=protocol['checkpoint_sha256'],
                order_hashes=lineage['order_hashes'], settings=saved['settings'], config=saved['config'],
                source_sha256=protocol['source_sha256'], compile=protocol['compile'], bf16=protocol['bf16'] if 'bf16' in protocol else True,
                gradient_checkpointing=False))
            for name in ('continuation.py', 'continuation_core.py'):
                shutil.copy2(ROOT / name, args.out / name)
        original.write(args.out/f'rank-{rank:02d}.json',dict(rank=rank,local_rank=local,pid=os.getpid(),host=socket.gethostname(),
            start_ticks=Path('/proc/self/stat').read_text().split()[21],gpu_uuid=str(torch.cuda.get_device_properties(local).uuid),**signature))
        selected, settings = protocol['t0_views'], saved['settings']
        cannon = original.C.Vocab(saved['vocab'])(original.CANNON_BARRAGE)
        original.validate(model, selected, args.data, cannon, settings, rank, world, args.out, index, True)
        # Startup evaluation may touch framework RNG; restore saved rank RNG again before creating the isolated loader.
        restore_rng(saved, rank)
        if index == 3200:
            save_boundary(model,optimizer,saved,args,index,rank,previous,started,lineage)
        loader = core.offset_loader(args.data, batches, index, args.target_index, rank, cannon)
        iterator = iter(loader)
        params = dict(model.named_parameters())
        log = (args.out / 'train.jsonl').open('x') if rank == 0 else None
        current = index
        while current < args.target_index:
            stop = torch.tensor(previous + time.monotonic()-started >= args.total_seconds, device='cuda', dtype=torch.int32)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop):
                break
            xs, ts = original.device_batch(next(iterator))
            optimizer.param_groups[0]['lr'] = core.absolute_lr(settings, current)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                loss, metrics = objective(xs, ts)
            finite = torch.isfinite(loss).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError('Nonfinite continuation loss')
            loss.backward()
            detail = (current+1)%50 == 0 or core.milestone(current+1,args.target_index)
            before = {n:p.detach().clone() for n,p in params.items()} if rank==0 and detail else None
            gradients = original.grouped_norms((n,p.grad) for n,p in params.items()) if rank==0 and detail else None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            current += 1
            if current%10 == 0 or detail:
                counts=torch.tensor([metrics['loss_counts'][h] for h in original.HEADS],device='cuda',dtype=torch.float64)
                sums=torch.tensor([metrics[h]*metrics['loss_counts'][h] for h in original.HEADS],device='cuda',dtype=torch.float64)
                dist.all_reduce(counts)
                dist.all_reduce(sums)
                if rank==0:
                    row=dict(index=current,step=30000+current,arm=args.arm,lr=optimizer.param_groups[0]['lr'],
                        preclip_norm=float(norm),counts=counts.tolist(),losses=dict(zip(original.HEADS,(sums/counts.clamp_min(1)).tolist())),
                        cumulative_seconds=previous+time.monotonic()-started,optimizer_reset=False,
                        alpha=[float(b.alpha().detach()) for b in model.tile_blocks] if args.arm=='split' else None)
                    if detail:
                        row.update(gradient_norms=gradients,parameter_norms=original.grouped_norms(params.items()),
                            update_norms=original.grouped_norms((n,p.detach()-before[n]) for n,p in params.items()))
                        row['branch_parameters']={n:dict(parameter_norm=float(p.detach().norm()),
                            gradient_norm=float(p.grad.norm()) if p.grad is not None else None,
                            update_norm=float((p.detach()-before[n]).norm())) for n,p in params.items()
                            if n.startswith('tile_blocks.') and any(k in n for k in ('.q.','.condition_','.norm.','.alpha_'))}
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    log.flush()
                    original.write(args.out/'progress.json',row)
                    print('CONTINUATION_UPDATE '+json.dumps({k:row[k] for k in ('index','step','preclip_norm','cumulative_seconds')}),flush=True)
            del before, xs, ts, loss
            if core.milestone(current,args.target_index):
                original.validate(model,selected,args.data,cannon,settings,rank,world,args.out,current,True)
                save_boundary(model,optimizer,saved,args,current,rank,previous,started,lineage)
        del iterator, loader
        if current != args.target_index:
            original.validate(model,selected,args.data,cannon,settings,rank,world,args.out,current,True)
            save_boundary(model,optimizer,saved,args,current,rank,previous,started,lineage)
        elif current == index:
            save_boundary(model,optimizer,saved,args,current,rank,previous,started,lineage)
        if rank==0:
            result=core.time_status(previous,time.monotonic()-started,current==args.target_index,args.total_seconds)
            result.update(index=current,step=30000+current,target_index=args.target_index,optimizer_reset=False,
                          exact_gpu_replay_verified=False,distillation_started=False)
            original.write(args.out/'finished.json',result)
            log.close()
            print('CONTINUATION_FINISHED '+json.dumps(result),flush=True)
        dist.barrier()
    except BaseException:
        if output_owned:
            original.write(args.out/f'failed-rank-{rank:02d}.json',dict(rank=rank,pid=os.getpid(),traceback=traceback.format_exc(),
                           numerical_failure_not_automatically_established=True))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
