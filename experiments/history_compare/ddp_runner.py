import argparse
from dataclasses import asdict
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]
sys.path.insert(0, str(SOURCE))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from cr.compression import CANNON_BARRAGE, compress_chunk
from cr import observation as C
from cr.policy import TILE_KEYS
from cr.history_imitation import Games, Objective, chunks, collate, loss_and_metrics, read, summarize
from cr.policy import Config, build
from cr.splitquery import SplitQueryBlock, install

HEADS = ('what', 'which', 'tile', 'value', 'next_when', 'next_which')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def training_entries(manifest, settings):
    entries = [e for e in manifest['entries'] if not e['validation'] and
               not (settings.get('val_after') and (e.get('battle_time') or '') >= settings['val_after'])]
    if settings.get('max_train'):
        entries = random.Random(settings['seed']).sample(entries, min(settings['max_train'], len(entries)))
    return entries


def rank_order(entries, seed, rank, world=16, workers=2):
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    lists = []
    for worker in range(workers):
        shard = shuffled[rank * workers + worker::world * workers]
        ordered = [entry for start in range(0, len(shard), 64)
                   for entry in sorted(shard[start:start + 64], key=lambda e: e['decisions'])]
        lists.append([ordered[i:i + 4] for i in range(0, len(ordered), 4) if len(ordered[i:i + 4]) == 4])
    full = []
    for i in range(min(map(len, lists)) * workers):
        full.append(lists[i % workers][i // workers])
    return full


def grouped_norms(items):
    sums = {}
    for name, tensor in items:
        if tensor is None:
            continue
        key = 'tile_blocks.' + name.split('.')[1] if name.startswith('tile_blocks.') else name.split('.')[0]
        sums[key] = sums.get(key, 0) + tensor.detach().float().square().sum()
    return {k: float(v.sqrt()) for k, v in sums.items()}


@contextmanager
def eager_blocks(model):
    blocks = [*model.encoder, *model.tile_blocks, *model.lstm.blocks]
    saved = [block._compiled_call_impl for block in blocks]
    for block in blocks:
        block._compiled_call_impl = None
    try:
        yield
    finally:
        for block, implementation in zip(blocks, saved):
            block._compiled_call_impl = implementation


def rms(tensor):
    values = tensor.detach().float()
    values = values[torch.isfinite(values)]
    return float(values.square().mean().sqrt()) if values.numel() else 0.


def geometry_hook(index, traces):
    def hook(module, args):
        position, memory, mask = [v[:4] for v in args[:3]]
        p = module.q(module.norm(position))
        if isinstance(module, SplitQueryBlock):
            condition = args[3][:4]
            c = module.alpha() * module.condition_q(module.condition_norm(condition))
            q = p + c
            branch = dict(alpha=float(module.alpha()), position_projected_rms=rms(p), condition_projected_rms=rms(c),
                          condition_raw_rms=rms(condition), condition_to_position_rms=rms(c) / max(rms(p), 1e-20))
        else:
            q, branch = p, dict(shared_residual_rms=rms(position.float().mean(1)))
        q = q.reshape(len(position), position.shape[1], module.heads, -1).transpose(1, 2)
        k, _ = module.kv(module.memory_norm(memory)).reshape(len(memory), memory.shape[1], 2, module.heads, -1).unbind(2)
        k = k.transpose(1, 2)
        with torch.autocast('cuda', enabled=False):
            content = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
            bias = mask.to(q.dtype).float()
            scores = content + bias
            probs = scores.softmax(-1)
            entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
            top2 = scores.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
        traces.append(dict(block=index, sampled_play_frames=len(position),
                           scope='FP32 reconstruction of autocast projections, first up to four play frames/view',
                           residual_rms=rms(position), residual_cell_difference_rms=rms(position.float() - position.float().mean(1, keepdim=True)),
                           query_rms=rms(q), key_rms=rms(k), query_cell_difference_rms=rms(q.float() - q.float().mean(2, keepdim=True)),
                           content_logits_rms=rms(content), position_bias_rms=rms(bias), **branch,
                           heads=[dict(entropy=float(entropy[:, h].mean()), margin_mean=float(margin[:, h].mean()),
                                       margin_p95=float(margin[:, h].flatten().quantile(.95)), margin_max=float(margin[:, h].max()),
                                       saturated_fraction=float((probs[:, h].max(-1).values > 1 - 1e-6).float().mean()),
                                       q_norm_mean=float(q[:, h].float().norm(dim=-1).mean()),
                                       k_norm_mean=float(k[:, h].float().norm(dim=-1).mean())) for h in range(module.heads)]))
    return hook


def item(data, entry, cannon):
    xs, ts = next(chunks(read(data / entry['file'], entry['owner']), 1280, True, None))
    xs, _ = compress_chunk(xs, cannon)
    return xs, ts


def device_batch(batch):
    return [{k: v.cuda(non_blocking=True) for k, v in group.items()} for group in batch]


def head_gradients(model, batch, settings):
    xs, ts = device_batch(batch)
    with torch.autocast('cuda', torch.bfloat16):
        out = model(xs)
        zero = out['value'].new_zeros(())
        live = ts['time'] & ts['ok']
        what, target = out['what'][live], ts['what'][live]
        allowed = torch.isfinite(what.gather(1, target[:, None])).squeeze(1)
        play = live.reshape(-1) & (ts['what'].reshape(-1) == C.PLAY)
        chosen = ts['which'].reshape(-1)[play]
        which_ok = ts['which_ok'].reshape(-1)[play]
        which = out['which'].reshape(-1, 4)[play]
        flat = {k: v.reshape(-1, *v.shape[2:])[play] for k, v in xs.items() if k in ('hand_card', 'legal', 'obj_side', 'obj_float')}
        tiles = model.tiles({k: out[k][play] for k in TILE_KEYS if k in out}, flat, chosen) if play.any() else None
        nxt = ts['time'] & ts['next_which_ok']
        losses = [F.cross_entropy(what[allowed], target[allowed]) if allowed.any() else zero,
                  F.cross_entropy(which[which_ok], chosen[which_ok]) if which_ok.any() else zero,
                  F.cross_entropy(tiles, ts['tile'].reshape(-1)[play]) if tiles is not None else zero,
                  settings['value_coef'] * F.mse_loss(out['value'][ts['time']], ts['value'][ts['time']]),
                  settings['next_play'] * F.cross_entropy(out['next_when'][ts['time']], ts['next_when'][ts['time']]),
                  settings['next_play'] * F.cross_entropy(out['next_which'][nxt], ts['next_which'][nxt]) if nxt.any() else zero]
    params = list(model.named_parameters())
    result = {}
    for name, loss in zip(HEADS, losses):
        gradients = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True) if loss.requires_grad else [None] * len(params)
        result[name] = grouped_norms((n, g) for (n, _), g in zip(params, gradients))
    return dict(scope='Single fixed validation view, separate weighted heads, no optimizer update', heads=result)


def validate(model, selected, data, cannon, settings, rank, world, out, index, full_headgrad=False):
    model.eval()
    rows, traces = [], []
    with eager_blocks(model):
        hooks = [block.register_forward_pre_hook(geometry_hook(i, traces)) for i, block in enumerate(model.tile_blocks)]
        try:
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                for selected_index in range(rank, len(selected), world):
                    xs, ts = device_batch(collate([item(data, selected[selected_index], cannon)]))
                    _, metrics = loss_and_metrics(model, xs, ts, settings['value_coef'], settings['next_play'])
                    rows.append((selected_index, metrics))
        finally:
            for hook in hooks:
                hook.remove()
        headgrad = head_gradients(model, collate([item(data, selected[8], cannon)]), settings) if rank == 0 and full_headgrad else None
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(dict(rows=rows, geometry=traces), gathered, dst=0)
    if rank == 0:
        all_rows = [row for _, row in sorted([row for part in gathered for row in part['rows']])]
        value = dict(index=index, step=30000 + index, entries=selected, per_view=all_rows,
                     metrics=summarize(all_rows), tile_labels=sum(r['loss_counts']['tile'] for r in all_rows),
                     geometry=[trace for part in gathered for trace in part['geometry']], headgrad=headgrad,
                     scope='32 matched full validation views, eager BF16, distributed across 16 ranks')
        write(out / f'validation-{index:05d}.json', value)
        print('SPLITQUERY_DDP_VALIDATION ' + json.dumps(dict(index=index, **value['metrics'])), flush=True)
    model.train()
    dist.barrier()


def save(model, optimizer, saved, out, index, arm):
    path = out / f'checkpoint-{index:05d}.pt'
    temp = path.with_suffix('.partial')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=saved['config'],
                    vocab=saved['vocab'], settings=saved['settings'], step=30000 + index, index=index,
                    arm=arm, phase='direct', fresh_optimizer=True, real_ddp=True, world_size=16), temp)
    temp.replace(path)
    write(out / 'latest-checkpoint.json', dict(path=str(path), index=index, step=30000 + index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--arm', choices=['baseline', 'split'], required=True)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 0 or args.max_seconds <= 0:
        raise ValueError('Steps and time bounds must be nonnegative and positive respectively')
    out = Path(args.run).resolve() / args.arm
    out.mkdir(parents=True, exist_ok=True)
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 16:
        raise ValueError('Require real two-node 16-GPU DDP')
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group('nccl', timeout=timedelta(minutes=10), device_id=torch.device('cuda', local))
    try:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved['step'] != 30000 or 'optimizer' in saved:
            raise ValueError('Require model-only best30k, no final optimizer')
        if saved['config'].get('split_query'):
            raise ValueError('Both arms must start from the same unsplit baseline checkpoint')
        settings = saved['settings']
        if settings['batch'] != 64 or settings['micro_batch'] != 4 or settings['workers'] != 2 or settings['ability_labels']:
            raise ValueError('Frozen batch/loader contract differs')
        manifest = json.loads((args.data / 'manifest.json').read_text())
        entries = training_entries(manifest, settings)
        val = sorted((e for e in manifest['entries'] if e['validation']), key=lambda e: e['decisions'])
        if len(val) < 32:
            raise ValueError('Require at least32 distinct validation views')
        selected = [val[len(val) * i // 32] for i in range(32)]
        seed = settings['seed'] * 1_000_003 + 30000
        planned = rank_order(entries, seed, rank)
        lengths = torch.tensor(len(planned), device='cuda')
        dist.all_reduce(lengths, op=dist.ReduceOp.MIN)
        steps = min(int(lengths), args.steps) if args.steps else int(lengths)
        if steps <= 0:
            raise ValueError('No complete global training batch available')
        plan_hash = hashlib.sha256(json.dumps(planned[:steps], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        torch.manual_seed(settings['seed'])
        model = build(Config(**saved['config']), C.effect_matrix(C.Vocab(saved['vocab'])))
        model.load_state_dict(saved['model'], strict=True)
        if args.arm == 'split':
            install(model)
        saved['config'] = asdict(model.config)
        model.cuda().train()
        for block in [*model.encoder, *model.tile_blocks, *model.lstm.blocks]:
            block.compile(dynamic=True)
        objective = nn.parallel.DistributedDataParallel(Objective(model, settings['value_coef'], settings['next_play']),
                                                        device_ids=[local], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'], fused=True)
        evidence = dict(rank=rank, local_rank=local, host=socket.gethostname(), host_alias=os.environ.get('HOST_ALIAS', socket.gethostname()), pid=os.getpid(),
                        start_ticks=Path('/proc/self/stat').read_text().split()[21], gpu_uuid=str(torch.cuda.get_device_properties(local).uuid),
                        interpreter=sys.executable, uv_environment=os.environ.get('UV_PROJECT_ENVIRONMENT'), uv_cache_dir=os.environ.get('UV_CACHE_DIR'),
                        checkpoint_sha256=digest(args.checkpoint), manifest_sha256=digest(args.data / 'manifest.json'),
                        order_sha256=plan_hash, steps=steps, selected_views=steps * 64, train_views=len(entries),
                        source_sha256={name: digest(SOURCE / name) for name in ('cr/history.py', 'cr/history_imitation.py', 'cr/compression.py', 'cr/policy.py', 'cr/observation.py')},
                        implementation_sha256={name: digest(SOURCE / name) for name in ('experiments/history_compare/ddp_runner.py', 'cr/splitquery.py')})
        write(out / f'rank-{rank:02d}.json', evidence)
        ranks = [None] * world if rank == 0 else None
        dist.gather_object(evidence, ranks, dst=0)
        if rank == 0:
            for key in ('checkpoint_sha256', 'manifest_sha256', 'source_sha256', 'implementation_sha256', 'steps'):
                if any(row[key] != ranks[0][key] for row in ranks):
                    raise ValueError(f'Cross-node mismatch: {key}')
            write(out / 'ranks.json', ranks)
            write(out / 'protocol.json', dict(settings=settings, config=saved['config'], real_ddp=True, world_size=16, gbs=64, mbs=4,
                source_step=30000, fresh_optimizer=True, schedule_total=42564, lr_formula='Inherited linear formula, extrapolated beyond42564 if one epoch reaches44188',
                steps=steps, train_views=len(entries), processed_views=steps*64, dataset_pass_fraction=steps*64/len(entries),
                seed=seed, compile='block.compile(dynamic=True), unchanged default', bf16=True, gradient_checkpointing=False,
                arm=args.arm, alpha_initial=.5, alpha_bound=[0,1], target_wall_seconds=3600, hard_bound_seconds=args.max_seconds,
                checkpoint_sha256=ranks[0]['checkpoint_sha256'], manifest_sha256=ranks[0]['manifest_sha256'],
                t0_views=selected, torch=str(torch.__version__), source_sha256=ranks[0]['source_sha256']))
            for name in ('experiments/history_compare/ddp_runner.py', 'cr/splitquery.py'):
                shutil.copy2(SOURCE / name, out / Path(name).name)
        write(out / f'order-rank-{rank:02d}.json', dict(seed=seed, rank=rank, sha256=plan_hash, batches=planned[:steps]))
        del planned, manifest, val
        cannon = C.Vocab(saved['vocab'])(CANNON_BARRAGE)
        validate(model, selected, args.data, cannon, settings, rank, world, out, 0, True)
        loader = torch.utils.data.DataLoader(Games(args.data, entries, 1280, seed, rank=rank, world=world, cannon_index=cannon),
                                             batch_size=4, num_workers=2, collate_fn=collate, pin_memory=True)
        start = window_start = time.monotonic()
        parameters = dict(model.named_parameters())
        log = (out / 'train.jsonl').open('x') if rank == 0 else None
        if rank == 0:
            print(f'SPLITQUERY_DDP_READY world=16 arm={args.arm} steps={steps} end={30000+steps}', flush=True)
        completed, timed_out = 0, False
        iterator = iter(loader)
        for index in range(steps):
            batch = next(iterator)
            xs, ts = device_batch(batch)
            if ts['time'].shape[0] != 4:
                raise ValueError('Partial batch would violate GBS64')
            lr = settings['lr'] * min(1., (30001 + index) / max(1, settings['warmup'])) * (1 - (1 - settings['final_lr']) * (30000 + index) / 42564)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                loss, metrics = objective(xs, ts)
            finite = torch.isfinite(loss).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError('Nonfinite DDP loss')
            loss.backward()
            completed = index + 1
            detail = completed % 50 == 0 or completed in (1, 16, 64, 128, 256, 512, 768, 1024, 3000) or completed == steps
            before = {n: p.detach().clone() for n, p in parameters.items()} if detail and rank == 0 else None
            gradient = grouped_norms((n, p.grad) for n, p in parameters.items()) if detail and rank == 0 else None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if completed % 10 == 0 or detail:
                counts = torch.tensor([metrics['loss_counts'][h] for h in HEADS], device='cuda', dtype=torch.float64)
                numbers = torch.tensor([metrics[h] * metrics['loss_counts'][h] for h in HEADS], device='cuda', dtype=torch.float64)
                dist.all_reduce(counts)
                dist.all_reduce(numbers)
                now = time.monotonic()
                if rank == 0:
                    row = dict(index=completed, step=30000 + completed, lr=lr, arm=args.arm, real_ddp=True,
                        counts=counts.tolist(), losses=dict(zip(HEADS, (numbers/counts.clamp_min(1)).tolist())),
                        preclip_norm=float(norm), clip_scale=min(1.,1./(float(norm)+1e-6)), seconds=now-start,
                        average_steps_per_second=completed/max(now-start,1e-6), eta_seconds=(steps-completed)*(now-start)/completed,
                        alpha=[float(b.alpha().detach()) for b in model.tile_blocks] if args.arm=='split' else None)
                    if detail:
                        row.update(gradient_norms=gradient, parameter_norms=grouped_norms(parameters.items()),
                                   update_norms=grouped_norms((n,p.detach()-before[n]) for n,p in parameters.items()))
                        row['branch_parameters'] = {n:dict(parameter_norm=float(p.detach().norm()), gradient_norm=float(p.grad.norm()) if p.grad is not None else None,
                                                          update_norm=float((p.detach()-before[n]).norm())) for n,p in parameters.items()
                                                    if n.startswith('tile_blocks.') and any(k in n for k in ('.q.', '.condition_', '.norm.', '.alpha_'))}
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    log.flush()
                    write(out/'progress.json',row)
                    print(f'SPLITQUERY_DDP_UPDATE arm={args.arm} step={30000+completed} index={completed} norm={float(norm):.5f} tile={row["losses"]["tile"]:.5f} sps={row["average_steps_per_second"]:.3f} eta_s={row["eta_seconds"]:.0f}',flush=True)
                del before
            validate_now = completed in (1,16,64,128,256,512,768,1024,3000,3200) or completed%1000==0 or completed==steps
            if validate_now:
                if rank==0:
                    save(model,optimizer,saved,out,completed,args.arm)
                validate(model,selected,args.data,cannon,settings,rank,world,out,completed,completed%1000==0 or completed in (256,1024))
            timeout = torch.tensor(time.monotonic()-start >= args.max_seconds,device='cuda',dtype=torch.int32)
            dist.all_reduce(timeout,op=dist.ReduceOp.MAX)
            if bool(timeout):
                timed_out=True
                break
        del iterator, loader
        if timed_out:
            if rank==0:
                save(model,optimizer,saved,out,completed,args.arm)
            validate(model,selected,args.data,cannon,settings,rank,world,out,completed,True)
        result=dict(status='bounded_time_limit' if timed_out else 'complete',index=completed,step=30000+completed,
                    requested_steps=steps,real_ddp=True,world_size=16,seconds=time.monotonic()-start,pid=os.getpid())
        write(out/f'finished-rank-{rank:02d}.json',result)
        dist.barrier()
        if rank==0:
            write(out/'finished.json',result)
            log.close()
            print('SPLITQUERY_DDP_FINISHED '+json.dumps(result),flush=True)
    except BaseException:
        write(out/f'failed-rank-{rank:02d}.json',dict(rank=rank,pid=os.getpid(),traceback=traceback.format_exc()))
        print(f'SPLITQUERY_DDP_FAILED rank={rank}',flush=True)
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
