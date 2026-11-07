"""Distil placement search into a model's tile head (5 Nov 2026): fine-tune a checkpoint on experiments/search_labels.py's
labels with ppo.search_loss (whole-board cross-entropy toward softmax(score / --tau) over each search's candidates, a
label weighted by its score spread, near-ties under --margin left out), plus --kl x KL(model || the frozen start) on the
play/wait, card and tile heads at every decision, so placement moves and the rest stays (16 Oct: search labels mixed
into PPO at weight 0.5 with no anchor collapsed the policy within ~30 updates).

Data as imitation learning reads games: each searching seat's decisions in chunks of --chunk from a zero LSTM state
(training: a random offset), only chunks holding a label. A fixed share of the label files (by name) is held out.

    python experiments/distill.py START.pt --labels DIR [DIR ...] --out RUN --steps 20000

RUN/metrics.jsonl: train and validation rows; RUN/latest.pt and RUN/step-<n>.pt load like any policy checkpoint
(config, model, vocab), so search_arena.py and the live player take them as they are.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ppo  # noqa: E402  the search loss and the KL
from cr import policy  # noqa: E402


def held_out(path, share):
    return int(hashlib.md5(path.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def chunks(path, chunk, rng):
    """One label file -> (rows {key: [T', ...]}, actions [T'], labels {t, slot, cells, scores}) per chunk with a label."""
    data = np.load(path)
    for seat in sorted({k.split('_')[0] for k in data.files if k.endswith('_t')}):
        p = seat + '_'
        keys = [k[len(p):] for k in data.files if k.startswith(p) and k[len(p):] not in ('action', 't', 'slot', 'cells', 'scores')]
        T, t = len(data[p + 'action']), data[p + 't']
        rows = {k: data[p + k] for k in keys}
        start = rng.randrange(chunk) if rng and T > chunk else 0
        for s in range(start - chunk if start else 0, T, chunk):
            s, e = max(s, 0), min(s + chunk, T)
            inside = (t >= s) & (t < e)
            if e > s and inside.any():
                yield ({k: v[s:e] for k, v in rows.items()}, data[p + 'action'][s:e],
                       dict(t=t[inside] - s, slot=data[p + 'slot'][inside], cells=data[p + 'cells'][inside],
                            scores=data[p + 'scores'][inside]))


class Labels(torch.utils.data.IterableDataset):
    """Shuffled chunks from `files`, forever (training) or once (validation); workers split the files."""

    def __init__(self, files, chunk, seed, validation=False):
        self.files, self.chunk, self.seed, self.validation = files, chunk, seed, validation

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, n = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        files = self.files[wid::n]
        while True:
            order = list(files)
            if not self.validation:
                rng.shuffle(order)
            buffer = []
            for f in order:
                buffer += list(chunks(f, self.chunk, None if self.validation else rng))
                if len(buffer) >= 256 or self.validation:  # ponytail: a 256-chunk shuffle buffer across files
                    if not self.validation:
                        rng.shuffle(buffer)
                    yield from buffer
                    buffer = []
            yield from buffer
            if self.validation:
                return


def collate(items):
    """Chunks -> (model inputs [B, T, ...] on the CPU, actions [B, T], valid [B, T], search_loss's labels as numpy)."""
    T = max(len(a) for _, a, _ in items)
    x = policy.tensors({k: policy.pad_stack([r[k] for r, _, _ in items]) for k in items[0][0]}, 'cpu')
    action = np.zeros((len(items), T), np.int64)
    valid = np.zeros((len(items), T), bool)
    for b, (_, a, _) in enumerate(items):
        action[b, :len(a)], valid[b, :len(a)] = a, True
    K = max(lab['cells'].shape[1] for _, _, lab in items)
    pad = lambda a, v: np.pad(a, ((0, 0), (0, K - a.shape[1])), constant_values=v)  # noqa: E731
    labels = dict(rows=np.concatenate([b * T + lab['t'] for b, (_, _, lab) in enumerate(items)]).astype(np.int64),
                  slot=np.concatenate([lab['slot'] for _, _, lab in items]).astype(np.int64),
                  cells=np.concatenate([pad(lab['cells'], -1) for _, _, lab in items]).astype(np.int64),
                  scores=np.concatenate([pad(lab['scores'], np.nan) for _, _, lab in items]).astype(np.float32))
    return x, action, valid, labels


def heads(model, x, action, valid):
    """Log-probs at the real decisions: play/wait [n, 4], card [n, 4], and the tile of each played card [m, CELLS]."""
    out, fx = policy.flat(model(x), x)
    v = valid.reshape(-1)
    xv = {k: fx[k][v] for k in ('own_flight_card', 'own_flight_float', 'hand_card', 'legal', 'hand_float', 'scalars',
                                'live_playable', 'live_abilities') if k in fx}
    what = policy.what_logp(out['what'][v], xv, 0.0, out['which'][v], True)
    which = policy.which_logp(out['which'][v], xv, True)
    _, slot, _, play = policy.split(action.reshape(-1)[v])
    tile = policy.tile_logp(model, out, fx, slot[play], v.nonzero().squeeze(1)[play])
    return out, fx, dict(what=what, which=which, tile=tile)


def step(model, start, batch, args, device):
    x, action, valid, labels = batch
    x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
    action, valid = torch.from_numpy(action).to(device), torch.from_numpy(valid).to(device)
    labels = {k: torch.from_numpy(v).to(device) for k, v in labels.items()}
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
        out, fx, mine = heads(model, x, action, valid)
        with torch.no_grad():
            _, _, ref = heads(start, x, action, valid)
        found, metrics = ppo.search_loss(model, out, fx, labels, args.tau, args.margin)
    kl = {k: ppo._kl(mine[k].float(), ref[k].float()).mean() for k in mine}
    loss = (found if found is not None else 0.0) + args.kl * sum(kl.values())
    return loss, dict(loss=float(loss), **{f'kl_{k}': float(v) for k, v in kl.items()}, **metrics)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('start', type=Path, help='the model to fine-tune (its frozen copy is the KL anchor)')
    ap.add_argument('--labels', type=Path, nargs='+', required=True, help='search_labels.py output folders')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32, help='chunks per step')
    ap.add_argument('--chunk', type=int, default=128, help='decisions per chunk (as imitation learning)')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--tau', type=float, default=0.1, help="softmax temperature over the candidates' judged logits")
    ap.add_argument('--margin', type=float, default=0.05, help='labels whose scores spread less than this are ties')
    ap.add_argument('--kl', type=float, default=1.0, help='weight of KL(model || start) on the three heads')
    ap.add_argument('--holdout', type=float, default=0.02)
    ap.add_argument('--val-every', type=int, default=1000)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model, _, saved = policy.load(args.start, device)
    start = copy.deepcopy(model).eval()
    for p in start.parameters():
        p.requires_grad_(False)
    model.train()
    model.checkpoint = True  # recompute the encoder in the backward: batch 32 x 128 ran out of an A100's 80 GB (5 Nov)
    files = sorted(f for d in args.labels for f in d.glob('*.npz'))
    val_files = [f for f in files if held_out(f, args.holdout)]
    train_files = [f for f in files if not held_out(f, args.holdout)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'settings.json').write_text(json.dumps(dict(vars(args), start=str(args.start), labels=[str(d) for d in args.labels],
                                                            out=str(args.out), train_files=len(train_files),
                                                            val_files=len(val_files)), indent=1))
    load = lambda ds: torch.utils.data.DataLoader(ds, batch_size=args.batch, collate_fn=collate,  # noqa: E731
                                                  num_workers=args.workers, persistent_workers=False)
    train = iter(load(Labels(train_files, args.chunk, args.seed)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=device.type == 'cuda')
    log = (args.out / 'metrics.jsonl').open('a')

    def write(row):
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)

    def save(name, n):
        torch.save(dict(config=saved['config'], model=model.state_dict(), vocab=saved['vocab'],
                        version=saved.get('version'), distilled=dict(step=n, start=str(args.start))), args.out / f'{name}.tmp')
        (args.out / f'{name}.tmp').replace(args.out / f'{name}.pt')

    def validate(n):
        model.eval()
        rows = []
        with torch.no_grad():
            for batch in load(Labels(val_files, args.chunk, args.seed, validation=True)):
                rows.append(step(model, start, batch, args, device)[1])
        model.train()
        keys = {k for r in rows for k in r}
        write(dict(step=n, split='val', batches=len(rows), **{k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}))

    t0, recent = time.time(), []
    validate(0)
    for n in range(1, args.steps + 1):
        loss, metrics = step(model, start, next(train), args, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent.append(metrics)
        if n % 50 == 0:
            keys = {k for r in recent for k in r}
            write(dict(step=n, split='train', seconds=round(time.time() - t0),
                       **{k: float(np.mean([r[k] for r in recent if k in r])) for k in keys}))
            recent = []
        if n % args.val_every == 0:
            validate(n)
        if n % args.save_every == 0:
            save(f'step-{n}', n)
            save('latest', n)
    save('latest', args.steps)


if __name__ == '__main__':
    main()
