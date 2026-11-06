"""Imitation learning on a converted dataset: the training loop (the network is cr/policy.py, the loss and data
cr/imitation.py).

    python train.py --data DATASET --out RUN                                  # one GPU (or CPU: --device cpu)
    torchrun --standalone --nproc_per_node=3 train.py --data DATASET --out RUN   # three GPUs, --batch is global

Rerunning the same command resumes from RUN/latest.pt with the settings saved there.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
import torch.distributed as dist

from cr import observation as C
from cr.history_imitation import Games, Objective, collate, loss_and_metrics, summarize
from cr.policy import Config, build
from cr.compression import CANNON_BARRAGE


# ---- training -----------------------------------------------------------------------------------------------------------
DEFAULTS = dict(epochs=4.0, batch=64, chunk=1280, workers=4, lr=3e-4, final_lr=1.0, weight_decay=0.01, warmup=500, value_coef=0.5,
                next_play=0.5, card_dim=32, object_dim=32, attention_layers=1, conv_channels=48, conv_layers=4, core=384, board=0,
                res_blocks=0, flight_grid=0, tile_blocks=0, board_rows=8, board_cols=4, res_norm=1, dist_bias=0,
                card_stats=0, arch=2, width=384, layers=10, heads=6, tile_layers=2, ability_labels=0, max_train=0, max_val=0, val_every=0, val_after='',
                inputs=2, manifest='manifest.json', max_steps=0, max_seconds=0, validation_batch=0, micro_batch=0,
                swanlab_mode='disabled', swanlab_project='DayBCR', swanlab_workspace='Terrasse', swanlab_name='',
                seed=0, log_every=50, checkpoint_every=1000, compile_blocks=1, device='cuda',
                temporal_layers=4, history_length=1280,
                compression_version='pair-unit1-aim1.5-cannon4-cap60-v1',
                own_flight_slots=1, v2_global_dim=0, v2_entity_dim=0, v2_layout_sha256='',
                numbered_checkpoints=1, split_query=0, split_alpha=0.5)
# final_lr: the learning rate falls linearly to lr x final_lr by the last step (1.0 = constant)
# validation (31 Oct): after every pass, and every val_every steps (0 = only after passes; RUN/best.pt keeps the step with
# the lowest 'val' loss); val_after 'YYYY-MM-DD': battles from that day on are left out of training and reported as
# 'val_recent' (the newest game data: how the model does on balance it has not trained on); 'val_expert' is the part of
# 'val' played by roster players. max_val caps each set (use it with val_every: a full set costs minutes).
# medium model (3.4M, 2026-10-16): --card_dim 64 --object_dim 96 --attention_layers 3 --conv_channels 96 --conv_layers 6
# --core 384
# large model (8.0M, 2026-10-18): --card_dim 64 --object_dim 160 --attention_layers 6 --conv_channels 160 --conv_layers 12
# --core 448 --board 32
# token model L (24.5M, 2 Nov): --arch 2 --card_dim 64 --core 384 (width 384, 10 layers, 6 heads, 2 tile blocks)


def lockstep(batches, device, world):
    """`batches` while every rank still has one: the ranks step together, and a pass ends for all when one rank's share
    runs out (the rest of the others' shares is skipped this pass). One tiny all-reduce a step. (DDP's Join let a rank
    that ran out skip optimizer steps: only the weights are re-synced when Join ends, its Adam moments stayed behind.)"""
    for item in batches:
        if world > 1:
            more = torch.ones((), device=device)
            dist.all_reduce(more, op=dist.ReduceOp.MIN)
            if not more:
                return
        yield item
    if world > 1:  # out of batches: tell the others, at the same step
        dist.all_reduce(torch.zeros((), device=device), op=dist.ReduceOp.MIN)


def train(data, out, settings):
    out = Path(out)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = settings['device'] == 'cuda'
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if cuda else torch.device('cpu')
    torch.set_num_threads(2)
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(minutes=15))
    manifest_path = Path(data) / settings.get('manifest', 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    vocab = C.Vocab(manifest['schema']['vocab'])
    v2_layouts = None
    if settings['v2_global_dim'] or settings['v2_entity_dim']:
        from cr.inputs_v2.v2_dataset import load_layout
        v2_layouts = load_layout(data, manifest, settings)
    elif manifest.get('input_profile', '').startswith('inputs-v2-history'):
        raise ValueError('A V2 dataset requires the complete hash-bound input adapter')
    cannon_index = vocab(CANNON_BARRAGE)
    if cannon_index <= 1 or settings['chunk'] != settings['history_length']:
        raise ValueError('Expected known cannon barrage and full-history chunk1280')
    if settings.get('ability_labels') and manifest['schema'].get('presses_at_tap'):
        raise SystemExit('--ability_labels 1 shifts presses to the tap; this dataset has them there already (convert.py since PR #14)')
    after = settings.get('val_after') or ''
    recent = (lambda e: bool(after) and (e.get('battle_time') or '') >= after)
    train_entries = [e for e in manifest['entries'] if not e['validation'] and not recent(e)]
    if max(e['decisions'] for e in manifest['entries']) > settings['history_length']:
        raise ValueError('Dataset includes games exceeding history position capacity')
    val_sets = {'val': [e for e in manifest['entries'] if e['validation'] and not recent(e)]}
    val_sets['val_expert'] = [e for e in val_sets['val'] if e.get('expert')]
    val_sets['val_recent'] = [e for e in manifest['entries'] if recent(e)]
    if after and not val_sets['val_recent']:
        raise SystemExit(f'--val_after {after}: no battle in the manifest is that recent (or it has no battle_time)')
    if settings.get('max_train'):  # ablations: the same random subset for every run with the same seed
        train_entries = random.Random(settings['seed']).sample(train_entries, min(settings['max_train'], len(train_entries)))
    if settings.get('max_val'):
        val_sets = {k: random.Random(settings['seed'] + 1).sample(v, min(settings['max_val'], len(v))) for k, v in val_sets.items()}
    if settings['batch'] < world or settings['batch'] % world:
        raise ValueError('global batch must be positive and divisible by world size')
    batch = settings['batch'] // world
    micro_batch = settings.get('micro_batch') or batch
    if micro_batch <= 0 or micro_batch > batch or batch % micro_batch:
        raise ValueError('per-rank microbatch must divide the per-rank batch')
    accumulation = batch // micro_batch
    # ponytail: an epoch is an estimate (random chunk offsets vary the count a little); exact epoch ends are not kept
    total = int(settings['epochs'] * sum(math.ceil(e['decisions'] / settings['chunk']) for e in train_entries) // settings['batch'])
    if settings.get('max_steps'):
        total = min(total, settings['max_steps'])

    torch.manual_seed(settings['seed'])
    stats = C.card_matrix(vocab) if settings.get('card_stats') else None  # --card_stats 1: the table's fields
    config = Config(vocab_size=len(vocab), next_play=settings['next_play'] > 0, **{k: settings[k] for k in
                    ('card_dim', 'object_dim', 'attention_layers', 'conv_channels', 'conv_layers', 'core')},
                    board=settings.get('board', 0), card_stats=0 if stats is None else stats.shape[1], inputs=settings.get('inputs', 2),
                    **{k: settings.get(k, v) for k, v in dict(res_blocks=0, flight_grid=0, tile_blocks=0, board_rows=8,
                                                              board_cols=4, res_norm=1, dist_bias=0, arch=1, width=384, layers=10,
                                                              heads=6, tile_layers=2).items()},
                    **{k: settings[k] for k in ('temporal_layers', 'history_length', 'compression_version',
                                               'own_flight_slots', 'v2_global_dim', 'v2_entity_dim', 'v2_layout_sha256')})
    config.split_query = bool(settings['split_query'])
    config.split_alpha = settings['split_alpha']
    policy = build(config, C.effect_matrix(vocab), stats).to(device)
    if settings['compile_blocks'] and cuda:
        for block in [*policy.encoder, *policy.tile_blocks, *policy.lstm.blocks]:
            block.compile(dynamic=True)
    rank_evidence = dict(rank=rank, local_rank=int(os.environ.get('LOCAL_RANK', 0)),
                         host=os.environ.get('HOST_ALIAS', os.uname().nodename), pid=os.getpid(),
                         device=str(device), micro_batch=micro_batch, accumulation=accumulation,
                         global_batch=settings['batch'], parameters=sum(p.numel() for p in policy.parameters()))
    if cuda:
        rank_evidence['gpu_uuid'] = str(torch.cuda.get_device_properties(device).uuid)
    if world > 1:
        all_ranks = [None] * world
        dist.all_gather_object(all_ranks, rank_evidence)
    else:
        all_ranks = [rank_evidence]
    if rank == 0:
        (out / 'ranks.json').write_text(json.dumps(all_ranks, indent=2))
        print('RANKS_READY ' + json.dumps(all_ranks), flush=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'], fused=cuda)
    step, decisions_seen = 0, 0
    if (out / 'latest.pt').exists():
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
        if asdict(Config(**saved['config'])) != asdict(config) or saved['vocab'] != list(vocab.index):
            raise ValueError('Checkpoint architecture or vocabulary mismatch')
        policy.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']
        decisions_seen = saved.get('decisions_seen', 0)
    if cuda:  # measured faster on A100s (2026-10-14)
        for conv in getattr(policy, 'convs', ()):  # TokenPolicy has none
            conv.to(memory_format=torch.channels_last)
        if hasattr(policy, 'convs'):
            policy.convs[0].register_forward_pre_hook(lambda m, a: (a[0].contiguous(memory_format=torch.channels_last),))
        if config.arch != 2:  # ponytail: arch 2 runs eager (2 Nov: Inductor ValueRangeError [0:-1] on its dynamic shapes);
            policy.forward = torch.compile(policy.forward, dynamic=True)  # compile it if the eager speed falls short
    objective = Objective(policy, settings['value_coef'], settings['next_play'])
    if world > 1:
        objective = nn.parallel.DistributedDataParallel(objective, device_ids=[device.index] if cuda else None,
                                                        find_unused_parameters=True)
    autocast = (lambda: torch.autocast('cuda', torch.bfloat16)) if cuda else nullcontext
    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None
    tracker = None
    if rank == 0 and settings.get('swanlab_mode', 'disabled') != 'disabled':
        import swanlab
        tracker = swanlab
        prior_run = json.loads((out / 'swanlab.json').read_text()) if (out / 'swanlab.json').exists() else None
        run = tracker.init(project=settings.get('swanlab_project', 'DayBCR'),
                           workspace=settings.get('swanlab_workspace', 'Terrasse'),
                           experiment_name=settings.get('swanlab_name') or out.name,
                           mode=settings['swanlab_mode'], logdir=str(out / 'swanlab'),
                           id=prior_run['id'] if prior_run else None, resume='allow' if prior_run else None,
                           config={**settings, 'model': asdict(config), 'world_size': world,
                                   'parameters': sum(p.numel() for p in policy.parameters()),
                                   'manifest': str(manifest_path.resolve()), 'train_views': len(train_entries),
                                   'val_views': {k: len(v) for k, v in val_sets.items()}})
        (out / 'swanlab.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
        print(f'SWANLAB_READY {run.url}', flush=True)

    def write(row):
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)
        temp = out / 'progress.tmp'
        temp.write_text(json.dumps(row))
        temp.replace(out / 'progress.json')
        if tracker is not None:
            prefix = row['split']
            tracker.log({f'{prefix}/{k}': v for k, v in row.items()
                         if k not in ('split', 'step') and isinstance(v, (int, float))}, step=row['step'])

    def checkpoint():
        if rank == 0:
            tmp = out / 'latest.tmp'
            torch.save(dict(config=asdict(config), model=policy.state_dict(), optimizer=optimizer.state_dict(),
                            step=step, decisions_seen=decisions_seen, settings=settings, vocab=list(vocab.index)), tmp)
            tmp.replace(out / 'latest.pt')
            if settings['numbered_checkpoints'] and (step % settings['checkpoint_every'] == 0 or step == total or timed_out):
                numbered = out / f'checkpoint-{step:08d}.pt'
                if not numbered.exists():
                    os.link(out / 'latest.pt', numbered)

    def gather_window(rows):
        if world == 1:
            return rows
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(rows, gathered, dst=0)
        return [row for shard in gathered for row in shard] if rank == 0 else []

    def loader(entries, seed, validation=False, rank=0, world=1):
        return torch.utils.data.DataLoader(Games(data, entries, settings['chunk'], seed, validation, rank, world,
                                                 bool(settings.get('ability_labels')), cannon_index, v2_layouts, vocab),
                                           batch_size=(settings.get('validation_batch') or micro_batch) if validation else micro_batch,
                                           num_workers=settings['workers'], collate_fn=collate,
                                           pin_memory=cuda)

    best, validated = out / 'best.json', [None]

    def validate():
        if validated[0] == step:  # a pass that ends on a val_every step
            return
        validated[0] = step
        policy.eval()
        for split, entries in val_sets.items():
            if not entries:
                continue
            with torch.no_grad(), autocast():
                rows = [loss_and_metrics(policy, *({k: v.to(device) for k, v in g.items()} for g in b),
                                         settings['value_coef'], settings['next_play'])[1]
                        for b in loader(entries, 0, validation=True, rank=rank, world=world)]
            if world > 1:
                gathered = [None] * world if rank == 0 else None
                dist.gather_object(rows, gathered, dst=0)
                if rank == 0:
                    rows = [row for shard in gathered for row in shard]
            if rank == 0:
                result = summarize(rows)
                write(dict(step=step, split=split, views=len(entries), decisions_seen=decisions_seen, **result))
                if split == 'val' and (not best.exists() or result['loss_weighted'] < json.loads(best.read_text())['loss_weighted']):
                    torch.save(dict(config=asdict(config), model=policy.state_dict(), step=step, settings=settings,
                                    vocab=list(vocab.index)), out / 'best.tmp')
                    (out / 'best.tmp').replace(out / 'best.pt')
                    best.write_text(json.dumps(dict(step=step, loss=result['loss'], loss_weighted=result['loss_weighted'])))
        policy.train()
        if world > 1:
            dist.barrier()

    started, window = time.monotonic(), []
    window_started, window_steps = started, 0
    timed_out = False
    if rank == 0:
        (out / 'running.json').write_text(json.dumps({'pid': os.getpid(), 'step': step, 'total_steps': total,
                                                     'parameters': sum(p.numel() for p in policy.parameters()),
                                                     'world_size': world}))
        print(f'TRAIN_READY step={step} total={total} world={world}', flush=True)
    while step < total:
        # a pass over the data; a resumed run starts a fresh order (seeded by the step it resumes at)
        begun = step
        micro_step, local_observed = 0, 0
        optimizer.zero_grad(set_to_none=True)
        for xs, ts in lockstep(loader(train_entries, settings['seed'] * 1_000_003 + step, rank=rank, world=world),
                               device, world):
            xs, ts = ({k: v.to(device, non_blocking=True) for k, v in g.items()} for g in (xs, ts))
            for group in optimizer.param_groups:
                group['lr'] = settings['lr'] * min(1.0, (step + 1) / max(1, settings['warmup'])) * \
                    (1 - (1 - settings.get('final_lr', 1.0)) * step / total)
            sync_context = objective.no_sync() if world > 1 and micro_step + 1 < accumulation else nullcontext()
            with sync_context:
                with autocast():
                    loss, metrics = objective(xs, ts)
                metrics['batch_views'] = int(ts['time'].any(1).sum())
                metrics['padded_length'] = int(ts['time'].shape[1])
                metrics['padded_objects'] = int(xs['obj_side'].shape[-1])
                finite = torch.tensor(math.isfinite(metrics['loss']), device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite):
                    raise FloatingPointError(f'nonfinite loss at step {step}; latest.pt is unchanged')
                (loss / accumulation).backward()
            micro_step += 1
            local_observed += int(ts['time'].sum())
            window.append(metrics)
            if micro_step < accumulation:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'nonfinite gradient norm at step {step}; latest.pt is unchanged')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            observed = torch.tensor(local_observed, device=device, dtype=torch.int64)
            if world > 1:
                dist.all_reduce(observed)
            decisions_seen += int(observed)
            metrics['observed_decisions'] = int(observed)
            micro_step, local_observed = 0, 0
            window_steps += 1
            if step % settings['log_every'] == 0:
                window = gather_window(window)
                gpu_memory = torch.tensor([torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
                                           torch.cuda.max_memory_allocated(device)] if cuda else [0, 0, 0],
                                          dtype=torch.float64, device=device)
                if world > 1:
                    dist.all_reduce(gpu_memory, op=dist.ReduceOp.MAX)
                now = time.monotonic()
                elapsed = max(now - window_started, 1e-6)
                if rank == 0:
                    extra = dict(lr=optimizer.param_groups[0]['lr'], grad_norm=float(grad_norm),
                                 decisions_seen=decisions_seen, steps_per_second=window_steps / elapsed,
                                 decisions_per_second=sum(m.get('observed_decisions', 0) for m in window) / world / elapsed,
                                 effective_decisions_per_second=sum(m['n'] for m in window) / elapsed,
                                 memory_allocated_gb=float(gpu_memory[0]) / 1e9,
                                 memory_reserved_gb=float(gpu_memory[1]) / 1e9,
                                 max_memory_allocated_gb=float(gpu_memory[2]) / 1e9)
                    extra.update(max_padded_length=max(m['padded_length'] for m in window),
                                 max_padded_objects=max(m['padded_objects'] for m in window),
                                 views_in_window=sum(m['batch_views'] for m in window),
                                 gradient_checkpointing=False)
                    write(dict(step=step, split='train', seconds=round(now - started), **summarize(window), **extra))
                window = []
                window_started, window_steps = now, 0
            if step % settings['checkpoint_every'] == 0:
                checkpoint()
            if settings.get('val_every') and step % settings['val_every'] == 0 and step < total:
                validate()
                window_started, window_steps = time.monotonic(), 0
                window = []
            if settings.get('max_seconds'):
                stop = torch.tensor(time.monotonic() - started >= settings['max_seconds'], device=device)
                if world > 1:
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                timed_out = bool(stop)
            if step >= total or timed_out:
                break
        if step == begun:  # else this loop would never end
            raise RuntimeError('a pass made no step: some rank has no training data')
        checkpoint()
        validate()
        if timed_out:
            break
    if rank == 0:
        (out / 'finished.json').write_text(json.dumps({'step': step, 'total_steps': total, 'timed_out': timed_out,
                                                      'decisions_seen': decisions_seen,
                                                      'seconds': time.monotonic() - started}))
        print(f'TRAIN_FINISHED step={step} timed_out={timed_out}', flush=True)
        if tracker is not None:
            tracker.finish()
        log.close()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='a converted dataset (with manifest.json)')
    parser.add_argument('--out', type=Path, required=True, help='the run directory')
    for key, value in DEFAULTS.items():
        parser.add_argument('--' + key, type=type(value), default=value)
    args = parser.parse_args()
    settings = {k: getattr(args, k) for k in DEFAULTS}
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / 'settings.json'
    if saved.exists():
        settings = json.loads(saved.read_text())  # a resumed run keeps its settings
    elif int(os.environ.get('RANK', 0)) == 0:
        saved.write_text(json.dumps(settings, indent=1))
    train(args.data, args.out, settings)



if __name__ == '__main__':
    main()
