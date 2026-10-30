"""Append ids to a model's vocabulary (30 Oct 2026): new rows at the end of the card embedding and the effect table, so
every existing index (and every converted dataset) keeps its meaning.

A new row starts as the model's 'unknown' row, so the model reads the id exactly as it did before it was added, unless
the spec gives a base: then base row + the mean (variant - base) offset over the given pairs (e.g. hero Ice Wizard's unit
= Ice Wizard's row + the mean hero-unit offset of the heroes the model knows).

    python tools/extend_vocab.py SPEC.json --policy IN.pt OUT.pt [--critic CKPT.pt OUT.pt]

SPEC.json: {"ids": [id, ...], "init": {"<id>": {"base": base_id, "pairs": [[variant_id, base_id], ...]}}}
--critic reads a training checkpoint's 'critic' (ppo.py --critic-weights) and writes {'critic': ...} with the same rows.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr import observation as C  # noqa: E402


def rows(weight, index, new, init):
    """New rows of `weight` [n, d] for the ids `new`, by `init` (see the module docstring)."""
    out = []
    for i in new:
        spec = init.get(str(i), {})
        base = index.get(spec.get('base'))
        if base is None:
            out.append(weight[1].clone())  # unknown: read as before
            continue
        pairs = [(index[a], index[b]) for a, b in spec.get('pairs', []) if a in index and b in index]
        offset = torch.stack([weight[a] - weight[b] for a, b in pairs]).mean(0) if pairs else 0
        out.append(weight[base] + offset)
    return torch.stack(out) if out else weight[:0]


def extend_state(state, prefix, index, new, init, effect_rows):
    card = state[prefix + 'card.weight']
    state[prefix + 'card.weight'] = torch.cat([card, rows(card, index, new, init).to(card.dtype)])
    table = state[prefix + 'effect_table']
    state[prefix + 'effect_table'] = torch.cat([table, effect_rows.to(table.dtype)])
    if prefix + 'card.table' in state:  # card stats: the base's facts, else none (as unknown)
        stats = state[prefix + 'card.table']
        extra = [stats[index[init[str(i)]['base']]] if index.get(init.get(str(i), {}).get('base')) else stats[1]
                 for i in new]
        state[prefix + 'card.table'] = torch.cat([stats, torch.stack(extra) if extra else stats[:0]])
    return state


def extend(spec, policy_in, policy_out, critic_in=None, critic_out=None):
    saved = torch.load(policy_in, map_location='cpu', weights_only=False)
    vocab = C.Vocab(saved['vocab'])
    new = [i for i in dict.fromkeys(int(i) for i in spec['ids']) if i not in vocab.index]
    ids = list(vocab.index) + new
    init = {str(k): {**v, 'base': int(v['base']) if 'base' in v else None,
                     'pairs': [[int(a), int(b)] for a, b in v.get('pairs', [])]} for k, v in spec.get('init', {}).items()}
    effect_rows = torch.as_tensor(C.effect_matrix(C.Vocab(ids))[len(vocab):])  # effect ids' fixed property vectors
    index = vocab.index
    extend_state(saved['model'], '', index, new, init, effect_rows)
    saved['config'] = dict(saved['config'], vocab_size=len(ids) + 2)
    saved['vocab'] = ids
    torch.save(saved, policy_out)
    if critic_in:
        critic = torch.load(critic_in, map_location='cpu', weights_only=False)['critic']
        torch.save({'critic': extend_state(critic, 'trunk.', index, new, init, effect_rows)}, critic_out)
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spec', type=Path)
    p.add_argument('--policy', nargs=2, type=Path, required=True, metavar=('IN', 'OUT'))
    p.add_argument('--critic', nargs=2, type=Path, metavar=('CKPT', 'OUT'))
    a = p.parse_args()
    new = extend(json.loads(a.spec.read_text()), *a.policy, *(a.critic or (None, None)))
    print(f'{len(new)} ids appended: {new}')


if __name__ == '__main__':
    main()
