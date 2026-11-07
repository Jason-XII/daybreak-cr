"""Peak GPU memory of one PPO update (critic targets, policy update, critic training) for a checkpoint, on the longest
human games of a dataset (the worst padding a minibatch can get), by minibatch size, with and without activation
checkpointing; also checks that checkpointing leaves the gradient unchanged. One GPU, no actors.

    python experiments/ppo_memory_probe.py --init IL.pt --data ~/il-data/v2 --minibatch 1 2 4
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr import imitation, policy  # noqa: E402
import ppo  # noqa: E402


def trajectories(model, data, n, device):
    entries = sorted((e for e in json.loads((data / 'manifest.json').read_text())['entries'] if not e['validation']),
                     key=lambda e: -e['decisions'])[:n]
    out = []
    for e in entries:
        x, _ = next(imitation.chunks(imitation.read(data / e['file'], e['owner']), 10**6, True, None))
        T = len(x['scalars'])
        rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(T)]
        inputs = {k: policy.pad_stack([r[k] for r in rows]) for k in rows[0]}
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            packed = policy.tensors({k: v[None] for k, v in inputs.items()}, device)
            action, logp = policy.sample(model, model(packed), packed, policy.ABILITY_EXPLORE)
        reward = np.zeros(T, np.float32)
        reward[-1] = 1.0
        out.append(dict(inputs=inputs, opp={k: inputs[k] for k in ppo.PRIVATE}, action=action.cpu().numpy(),
                        logp=logp.float().cpu().numpy(), reward=reward, outcome=1.0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--minibatch', type=int, nargs='+', default=[1, 2, 4])
    args = p.parse_args()
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    model, _, _ = policy.load(args.init, device)
    reference = copy.deepcopy(model).requires_grad_(False)
    critic = ppo.Critic(policy.load(args.init, device)[0]).to(device).train()
    model.train()
    games = trajectories(model, args.data, 2 * max(args.minibatch), device)
    print('decisions per game (longest first):', [len(g['action']) for g in games], flush=True)
    s = argparse.Namespace(gamma=0.999, lam=0.99, epochs=1, critic_epochs=1, clip=0.2, entropy=0.002, kl_coef=0.01,
                           search_coef=0.0, search_tau=0.05, search_margin=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # lr 0: measure without changing the weights
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0)

    # same gradient with and without checkpointing (one minibatch, fp32 to compare closely)
    s.minibatch = 1
    batch = ppo.minibatches([copy.deepcopy(games[-1])], s, device)
    ppo.critic_targets(batch, critic, s, device)
    chunk, x, _, valid = batch[0]
    y = ppo.targets(chunk, ('action',), device)
    grads = []
    for on in (False, True):
        model.checkpoint = on
        model.zero_grad(set_to_none=True)
        ppo.evaluate(model, x, y['action'], valid)['logp'].sum().backward()
        grads.append(torch.cat([q.grad.flatten() for q in model.parameters() if q.grad is not None]))
    print('checkpointing gradient max abs difference', float((grads[0] - grads[1]).abs().max()),
          'of max', float(grads[0].abs().max()), flush=True)
    del grads, batch
    model.zero_grad(set_to_none=True)

    for on in (True, False):
        model.checkpoint = critic.trunk.checkpoint = on
        for mb in args.minibatch:
            s.minibatch = mb
            row = dict(checkpoint=on, minibatch=mb)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                batches = ppo.minibatches(copy.deepcopy(games[:2 * mb]), s, device)
                t = time.monotonic()
                ppo.critic_targets(batches, critic, s, device)
                row['targets_gb'] = torch.cuda.max_memory_allocated(device) / 2**30
                torch.cuda.reset_peak_memory_stats(device)
                ppo.update(model, reference, optimizer, batches, s, device)
                row['policy_gb'] = torch.cuda.max_memory_allocated(device) / 2**30
                torch.cuda.reset_peak_memory_stats(device)
                ppo.train_critic(critic, critic_optimizer, batches, s, device)
                row['critic_gb'] = torch.cuda.max_memory_allocated(device) / 2**30
                torch.cuda.synchronize()
                row['seconds'] = time.monotonic() - t
                row['decisions'] = sum(len(g['action']) for g in games[:2 * mb])
                del batches
            except torch.OutOfMemoryError:
                row['oom'] = True
                torch.cuda.empty_cache()
            print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    print('device total GB', torch.cuda.get_device_properties(device).total_memory / 2**30)


if __name__ == '__main__':
    main()
