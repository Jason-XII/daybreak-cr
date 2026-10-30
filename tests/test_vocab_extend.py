"""Append-only vocabulary (30 Oct 2026): old indices never move, old models read new ids as unknown, and
tools/extend_vocab.py adds rows without changing what the model already knows."""
import importlib.util
from pathlib import Path

import torch

from cr import observation as C, policy
import synthetic

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('extend_vocab', ROOT / 'tools/extend_vocab.py')
extend_vocab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extend_vocab)


def test_vocab_keeps_list_order_and_old_sorted_indices():
    ids = [26000000, 26000023, 203000000]
    assert C.Vocab(ids).index == {i: n + 2 for n, i in enumerate(sorted(ids))}  # what the sorted vocabularies always gave
    v = C.Vocab(ids + [203000023, 1741956296])
    assert [v(i) for i in ids] == [2, 3, 4] and v(203000023) == 5 and v(1741956296) == 6
    assert list(v.index) == ids + [203000023, 1741956296]  # what checkpoints save
    assert C.Vocab({9, 3}).index == {3: 2, 9: 3}
    v(42), v(42)
    assert v(42) == 1 and v.misses[42] == 3


def inputs(card):
    x, _ = synthetic.arrays(2, 1)
    x['obj_card'][:, 0] = card
    rows = [policy.pack({k: a[i] for k, a in x.items()}) for i in range(2)]
    return policy.tensors({k: policy.pad_stack([r[k] for r in rows])[:, None] for k in rows[0]}, 'cpu')


def outputs(model, card):
    with torch.no_grad():
        out = model(inputs(card))
    return {k: v for k, v in out.items() if torch.is_tensor(v) and v.is_floating_point()}


def same(a, b):
    return all(torch.allclose(a[k], b[k], atol=1e-6, equal_nan=True) for k in a)


def tiny(vocab_size):
    torch.manual_seed(0)
    return policy.Policy(policy.Config(vocab_size=vocab_size, card_dim=4, object_dim=4, core=8, attention_layers=0)).eval()


def test_old_model_reads_newer_ids_as_unknown():
    model = tiny(5)
    assert same(outputs(model, 9), outputs(model, 1)) and not same(outputs(model, 2), outputs(model, 1))


def test_extend_appends_rows_and_keeps_behaviour(tmp_path):
    ids = [26000000, 26000023, 203000000]
    model = tiny(len(C.Vocab(ids)))
    torch.save(dict(config=model.config.__dict__, model=model.state_dict(), vocab=ids, version=7), tmp_path / 'in.pt')
    critic = {'trunk.' + k: v.clone() for k, v in model.state_dict().items()}
    torch.save({'critic': critic}, tmp_path / 'ckpt.pt')
    spec = {'ids': [203000023, 1741956296, 26000000],
            'init': {'203000023': {'base': 26000023, 'pairs': [[203000000, 26000000]]}}}
    new = extend_vocab.extend(spec, tmp_path / 'in.pt', tmp_path / 'out.pt', tmp_path / 'ckpt.pt', tmp_path / 'critic.pt')
    assert new == [203000023, 1741956296]  # known ids are not appended twice
    extended, vocab, saved = policy.load(tmp_path / 'out.pt')
    assert list(vocab.index) == ids + new and saved['config']['vocab_size'] == 7 and saved['version'] == 7
    w0, w = model.card.weight, extended.card.weight
    assert torch.equal(w[:5], w0) and torch.equal(w[6], w0[1])  # old rows as they were; plain new id = unknown
    assert torch.allclose(w[5], w0[3] + (w0[4] - w0[2]))        # hero = base + the known hero offset
    assert same(outputs(extended, 6), outputs(model, 1))         # the model reads the new plain id as before
    assert same(outputs(extended, 3), outputs(model, 3))
    c = torch.load(tmp_path / 'critic.pt', weights_only=False)['critic']
    assert torch.equal(c['trunk.card.weight'], w) and c['trunk.effect_table'].shape[0] == 7
