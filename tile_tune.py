"""Tile-head experiments on a frozen IL network (26 Oct, the placement plan's stage 1): several tile heads learn human
placements side by side from one frozen forward pass per batch. Every head sees the same batches, so their
differences are paired, and the trunk runs once (no backward through it).

    torchrun --standalone --nproc_per_node=4 tile_tune.py --base r6-final.pt --data DATASET --out RUN \\
        --variants plain:0,interact:4

A variant NAME:HEADS is the base's tile head (film, tile_blocks, tile) fine-tuned, plus an Interact branch with HEADS
heads (0: none, the control). Only those parts learn. Each variant is saved as a whole loadable checkpoint RUN/NAME.pt
(cr.policy.load). Validation (all ranks, a share each) also scores the base's own head: step 0 must match it for every
variant. Rerunning the same command resumes from RUN/state.pt with the settings saved there.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cr import observation as C
from cr.imitation import Games, collate
from cr.policy import PLAY, Config, Policy, load
from train import lockstep

DEFAULTS = dict(base='', variants='plain:0,interact:4', steps=20000, batch=64, chunk=128, workers=6, lr=3e-4,
                final_lr=0.1, warmup=200, weight_decay=0.01, max_train=0, max_val=2000, val_every=2000, seed=0,
                log_every=50, checkpoint_every=2000, device='cuda')
# batch: chunks per step over all ranks (no backward through the trunk, so twice train.py's 32 fits)
TRAINED = ('film', 'tile', 'tile_blocks', 'interact')  # the tile head's parts (the first name of each parameter)
W = C.W


def plays(out, xs, ts):
    """A frozen forward pass -> the tile head's inputs at the human card plays: (out rows, x rows, slot, target tile)."""
    play = (ts['time'] & ts['ok'] & (ts['what'] == PLAY)).reshape(-1)
    flat = lambda v: v.reshape(-1, *v.shape[2:])[play]  # noqa: E731
    sub = {k: out[k][play].float() for k in ('core', 'grid', 'tokens')}
    sx = {k: flat(xs[k]) for k in ('hand_card', 'legal', 'obj_side', 'obj_float')}
    return sub, sx, flat(ts['which']), flat(ts['tile'])


def scored(scores, target):
    """-> summed loss and metrics of tile scores against the human tiles."""
    loss = F.cross_entropy(scores, target, reduction='sum')
    top1 = scores.argmax(-1)
    dist_ = (((top1 // W - target // W) ** 2 + (top1 % W - target % W) ** 2).float()).sqrt()
    return loss, torch.stack([loss.detach(), (top1 == target).sum().float(),
                              (scores.topk(5, -1).indices == target[:, None]).any(1).sum().float(), dist_.sum(),
                              torch.tensor(float(len(target)), device=scores.device)])


def summary(sums):
    n = max(1.0, float(sums[4]))
    return dict(tile=float(sums[0]) / n, tile_top1=float(sums[1]) / n, tile_top5=float(sums[2]) / n,
                tile_dist=float(sums[3]) / n, plays=int(sums[4]))


def run(data, out, settings):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = settings['device'] == 'cuda'
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if cuda else torch.device('cpu')
    torch.set_num_threads(2)
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(minutes=15))
    manifest = json.loads((Path(data) / 'manifest.json').read_text())
    train_entries = [e for e in manifest['entries'] if not e['validation']]
    val_entries = [e for e in manifest['entries'] if e['validation']]
    if settings['max_train']:
        train_entries = random.Random(settings['seed']).sample(train_entries, min(settings['max_train'], len(train_entries)))
    if settings['max_val']:
        val_entries = random.Random(settings['seed'] + 1).sample(val_entries, min(settings['max_val'], len(val_entries)))
    batch = max(1, settings['batch'] // world)

    base, vocab, saved = load(settings['base'], device)
    base.requires_grad_(False)
    heads, params = {}, {}
    torch.manual_seed(settings['seed'])  # Interact's random init: the same on every rank
    for spec in settings['variants'].split(','):
        name, n_heads = spec.split(':')
        model = Policy(Config(**{**saved['config'], 'interact': int(n_heads)}))
        missing, unexpected = model.load_state_dict(saved['model'], strict=False)
        assert not unexpected and all(k.startswith('interact.') for k in missing), (missing, unexpected)
        model.to(device).eval()
        for k, p in model.named_parameters():
            p.requires_grad_(k.split('.')[0] in TRAINED)
        heads[name] = model
        params[name] = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for ps in params.values() for p in ps], lr=settings['lr'],
                                  weight_decay=settings['weight_decay'])
    step = 0
    out = Path(out)
    if (out / 'state.pt').exists():
        state = torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
        for name, model in heads.items():
            model.load_state_dict(torch.load(out / f'{name}.pt', map_location='cpu', weights_only=False)['model'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
    if cuda:  # as train.py: channels-last convs, the trunk compiled
        for conv in base.convs:
            conv.to(memory_format=torch.channels_last)
        base.convs[0].register_forward_pre_hook(lambda m, a: (a[0].contiguous(memory_format=torch.channels_last),))
        base.forward = torch.compile(base.forward, dynamic=True)
    autocast = (lambda: torch.autocast('cuda', torch.bfloat16)) if cuda else nullcontext
    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None

    def write(row):
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)

    def checkpoint():
        if rank == 0:
            for name, model in heads.items():
                torch.save(dict(config=asdict(model.config), model=model.state_dict(), step=step, settings=settings,
                                vocab=saved['vocab'], base=settings['base']), out / f'{name}.tmp')
                (out / f'{name}.tmp').replace(out / f'{name}.pt')
            torch.save(dict(optimizer=optimizer.state_dict(), step=step), out / 'state.tmp')
            (out / 'state.tmp').replace(out / 'state.pt')
        if world > 1:
            dist.barrier()

    def loader(entries, seed, validation=False):
        return torch.utils.data.DataLoader(Games(data, entries, settings['chunk'], seed, validation, rank, world),
                                           batch_size=batch, num_workers=settings['workers'], collate_fn=collate,
                                           pin_memory=cuda, prefetch_factor=4 if settings['workers'] else None)

    def validate():
        names = ['base', *heads]
        sums = torch.zeros(len(names), 5, device=device)
        with torch.no_grad():
            for xs, ts in loader(val_entries, 0, validation=True):  # each rank its share, no lockstep needed
                xs, ts = ({k: v.to(device, non_blocking=True) for k, v in g.items()} for g in (xs, ts))
                with autocast():
                    o = base(xs)
                sub, sx, slot, target = plays(o, xs, ts)
                if len(target):
                    for i, model in enumerate([base, *heads.values()]):
                        sums[i] += scored(model.tiles(sub, sx, slot), target)[1]
        if world > 1:
            dist.all_reduce(sums)
        if rank == 0:
            for i, name in enumerate(names):
                write(dict(step=step, split='val', variant=name, **summary(sums[i])))

    if step == 0:
        validate()
    started, window, window_plays = time.monotonic(), torch.zeros(len(heads), 5, device=device), 0
    total = settings['steps']
    while step < total:
        begun = step
        for xs, ts in lockstep(loader(train_entries, settings['seed'] * 1_000_003 + step), device, world):
            xs, ts = ({k: v.to(device, non_blocking=True) for k, v in g.items()} for g in (xs, ts))
            for group in optimizer.param_groups:
                group['lr'] = settings['lr'] * min(1.0, (step + 1) / max(1, settings['warmup'])) * \
                    (1 - (1 - settings['final_lr']) * step / total)
            with torch.no_grad(), autocast():
                o = base(xs)
            sub, sx, slot, target = plays(o, xs, ts)
            optimizer.zero_grad(set_to_none=True)
            if len(target):
                losses = []
                for i, model in enumerate(heads.values()):
                    loss, m = scored(model.tiles(sub, sx, slot), target)
                    losses.append(loss / len(target))
                    window[i] += m
                torch.stack(losses).sum().backward()  # the heads share no parameters: each learns from its own loss
            # one all-reduce for every head's gradients (a rank without plays sends zeros)
            flat = [p for ps in params.values() for p in ps]
            grads = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in flat])
            if world > 1:
                dist.all_reduce(grads)
                grads /= world
            offset = 0
            for p in flat:
                p.grad = grads[offset:offset + p.numel()].view_as(p)
                offset += p.numel()
            for ps in params.values():
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
            optimizer.step()
            step += 1
            if step % settings['log_every'] == 0:
                if world > 1:
                    dist.all_reduce(window)
                if rank == 0:
                    seconds = time.monotonic() - started
                    for i, name in enumerate(heads):
                        write(dict(step=step, split='train', variant=name, seconds=round(seconds),
                                   lr=optimizer.param_groups[0]['lr'], **summary(window[i])))
                window.zero_()
            if step % settings['checkpoint_every'] == 0:
                checkpoint()
            if step % settings['val_every'] == 0:
                validate()
            if step >= total:
                break
        if step == begun:
            raise RuntimeError('a pass made no step: some rank has no training data')
    checkpoint()
    if step % settings['val_every']:
        validate()
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
    run(args.data, args.out, settings)


if __name__ == '__main__':
    main()
