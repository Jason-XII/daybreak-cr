from dataclasses import asdict

import numpy as np
import pytest
import torch

from cr import history_imitation as I, observation as C, policy
from cr.compression import compress_frame
from cr.history import TemporalCore
from synthetic import arrays, dataset
import train_history


def config(**kwargs):
    return policy.Config(vocab_size=4, arch=2, inputs=1, width=16, core=16, heads=2,
                         layers=1, tile_layers=1, card_dim=4, temporal_layers=1,
                         **kwargs)


def test_history_is_causal_and_uses_past_gradients():
    torch.set_num_threads(2)
    core = TemporalCore(16, 2, layers=2, max_length=16)
    x = torch.randn(1, 8, 16, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 4:] += 2
    output = core(x)[0]
    torch.testing.assert_close(output[:, :4], core(changed)[0][:, :4])
    output[:, -1, 0].backward()
    assert x.grad[:, :4].abs().sum() > 0
    with pytest.raises(ValueError):
        core(x, state=(x, x))
    with pytest.raises(ValueError):
        core(torch.randn(1, 17, 16))


def test_history_round_trip_and_backward(tmp_path):
    xs, _ = arrays(3, 0)
    xs['obj_count'] = np.ones_like(xs['obj_side'], dtype=np.float32)
    xs = {k: torch.as_tensor(v[None]) for k, v in xs.items()}
    cfg = config(compression_version='pair-unit1-aim1.5-cannon4-cap60-v1')
    model = policy.build(cfg)
    output = model(xs)
    scores = model.tiles(output, xs, torch.zeros(3, dtype=torch.long))
    loss = output['value'].square().mean() + scores.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    checkpoint = tmp_path / 'model.pt'
    torch.save(dict(config=asdict(cfg), model=model.state_dict(), vocab=[0, 1, 2, 3]), checkpoint)
    loaded = policy.load(checkpoint)[0]
    torch.testing.assert_close(output['value'], loaded(xs)['value'])


def test_compression_retains_real_corners_and_pairs():
    count = 42
    x = dict(obj_side=np.ones(count, np.int64), obj_card=np.full(count, 2, np.int64),
             obj_data=np.full(count, 3, np.int64), obj_kind=np.ones(count, np.int64),
             obj_float=np.zeros((count, C.OBJ_FLOATS), np.float32),
             eff_id=np.zeros((count, C.MAX_EFFECTS), np.int64),
             eff_left=np.zeros((count, C.MAX_EFFECTS), np.float32))
    result, telemetry = compress_frame(x, 999)
    assert telemetry['pairs'] == 21
    assert (result['obj_count'] == 2).all()
    x['obj_data'][:] = 999
    x['obj_kind'][:] = 4
    x['obj_float'][:, :2] = np.array([[i / 18, j / 32] for i in range(6) for j in range(7)])
    result, telemetry = compress_frame(x, 999)
    assert len(result['obj_side']) == 4
    assert (result['obj_count'] == 1).all()
    corners = result['obj_float'][:, :2] * [18, 32]
    assert set(map(tuple, corners.round(4))) == {(0., 0.), (0., 6.), (5., 0.), (5., 6.)}


def test_history_runner_uses_repo_modules(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    settings = dict(train_history.DEFAULTS, epochs=1, max_steps=1, batch=2, micro_batch=1,
                    workers=0, card_dim=4, width=16, core=16, heads=2, layers=1,
                    tile_layers=1, temporal_layers=1, device='cpu', log_every=1,
                    compile_blocks=0, swanlab_mode='disabled')
    from cr.compression import CANNON_BARRAGE
    import json
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['schema']['vocab'].append(CANNON_BARRAGE)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    out = tmp_path / 'run'
    out.mkdir()
    train_history.train(root, out, settings)
    saved = torch.load(out / 'latest.pt', weights_only=True)
    assert saved['step'] == 1
    assert saved['config']['temporal_layers'] == 1
    assert (out / 'ranks.json').exists()
    numbered = out / 'checkpoint-00000001.pt'
    assert numbered.exists()
    before = numbered.read_bytes()
    train_history.train(root, out, settings)
    assert numbered.read_bytes() == before
    initial = tmp_path / 'initial.pt'
    torch.save({key: saved[key] for key in ('config', 'model', 'vocab')}, initial)
    fresh = tmp_path / 'fresh'
    fresh.mkdir()
    train_history.train(root, fresh, settings | {'initial_checkpoint': str(initial)})
    assert torch.load(fresh / 'latest.pt', weights_only=True)['step'] == 1
    bad = tmp_path / 'bad'
    bad.mkdir()
    with pytest.raises(ValueError, match='model-only'):
        train_history.train(root, bad, settings | {'initial_checkpoint': str(out / 'latest.pt')})


def test_full_game_loader_rejects_long_game_and_collate_overflow(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    import json
    entries = json.loads((root / 'manifest.json').read_text())['entries']
    entries[0]['decisions'] = 2000
    with pytest.raises(ValueError, match='must not split'):
        list(I.Games(root, entries[:1], 1280, 0))
    x, t = arrays(3, 0)
    x['obj_side'] = np.ones((3, 61), np.int64)
    with pytest.raises(ValueError, match='silent truncation'):
        I.collate([(x, {'what': np.zeros(3, np.int64)})])
