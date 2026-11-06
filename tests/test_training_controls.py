import json
import math
import sys
from pathlib import Path

import torch

from cr import imitation, observation as C, policy
from synthetic import arrays
import train


def test_old_token_inputs_backward():
    xs, _ = arrays(3, 0)
    xs = {k: torch.as_tensor(v[None]) for k, v in xs.items()}
    xs['ability'] = xs['ability'][..., :4]
    model = policy.build(policy.Config(vocab_size=4, arch=2, inputs=1, width=16, layers=2, heads=2,
                                       tile_layers=1, card_dim=4, core=8))
    out = model(xs)
    scores = model.tiles(out, xs, torch.zeros(3, dtype=torch.long))
    loss = out['value'].square().mean() + scores.square().mean() + out['what'][..., :2].square().mean()
    loss.backward()
    assert math.isfinite(loss.item())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.ability[0].in_features == 8


def test_swanlab_training_metrics(tmp_path, monkeypatch):
    from synthetic import dataset
    root = dataset(tmp_path / 'data', monkeypatch)
    events = []

    class Tracker:
        @staticmethod
        def init(**kwargs):
            events.append(('init', kwargs))
            return type('Run', (), dict(id='test', url='https://example.invalid/run'))()

        @staticmethod
        def log(metrics, step):
            events.append(('log', step, metrics))

        @staticmethod
        def finish():
            events.append(('finish',))

    monkeypatch.setitem(sys.modules, 'swanlab', Tracker)
    settings = dict(train.DEFAULTS, epochs=1, max_steps=2, batch=2, chunk=3, workers=0, card_dim=4,
                    object_dim=4, conv_channels=4, conv_layers=1, core=8, attention_layers=0,
                    device='cpu', log_every=1, swanlab_mode='cloud')
    out = tmp_path / 'run'
    out.mkdir()
    train.train(root, out, settings)
    assert json.loads((out / 'finished.json').read_text())['step'] == 2
    logs = [e[2] for e in events if e[0] == 'log']
    assert any('train/decisions_per_second' in e and 'train/grad_norm' in e for e in logs)
    assert any('val/card_acc' in e and 'val/tile_top1' in e for e in logs)
    assert events[-1] == ('finish',)


def test_accumulation_tracks_global_batch(tmp_path, monkeypatch):
    from synthetic import dataset
    root = dataset(tmp_path / 'data', monkeypatch)
    settings = dict(train.DEFAULTS, epochs=1, max_steps=2, batch=2, micro_batch=1, chunk=3, workers=0,
                    card_dim=4, object_dim=4, conv_channels=4, conv_layers=1, core=8, attention_layers=0,
                    device='cpu', log_every=1)
    out = tmp_path / 'run'
    out.mkdir()
    train.train(root, out, settings)
    saved = torch.load(out / 'latest.pt', weights_only=False)
    assert saved['step'] == 2
    assert 4 <= saved['decisions_seen'] <= 12
