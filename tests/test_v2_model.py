from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from cr import observation as C, policy
from cr.inputs_v2.v2_dataset import load_layout
from synthetic import arrays


def test_two_pending_slots_sorted_and_overflow_rejected():
    encode = lambda tick, command: [tick, command['card_id'], 0, 0]
    cards, floats = C.pack_pending_flights([(20, {'card_id': 3}), (10, {'card_id': 2})], int, encode, 'own')
    np.testing.assert_array_equal(cards, [2, 3])
    np.testing.assert_array_equal(floats[:, 0], [10, 20])
    with pytest.raises(ValueError, match='capacity'):
        C.pack_pending_flights([(i, {'card_id': i}) for i in range(3)], int, encode, 'own')


def test_two_pending_cards_and_both_costs_reserved():
    x = dict(own_flight_card=torch.tensor([[2, 3]]),
             own_flight_float=torch.tensor([[[0., 0., 0., .3], [0., 0., 0., .4]]]),
             hand_card=torch.tensor([[2, 3, 4, 5]]),
             hand_float=torch.zeros(1, 4, C.HAND_FLOATS), scalars=torch.zeros(1, C.SCALARS),
             legal=torch.ones(1, 4, C.H, C.W, dtype=torch.bool))
    x['scalars'][0, 0] = .8
    x['hand_float'][0, :, 0] = torch.tensor([.3, .4, .2, .1])
    assert policy.playable(x, overlap=True).tolist() == [[False, False, False, True]]
    logp = policy.what_logp(torch.zeros(1, 4), x, overlap=True)
    assert torch.isfinite(logp[0, :2]).all()
    assert torch.isneginf(logp[0, 2:]).all()


def test_v2_projection_backward_and_checkpoint(tmp_path):
    xs, _ = arrays(3, 0)
    count = xs['obj_side'].shape[-1]
    xs.update(own_flight_card=np.full((3, 2), 2, np.int64),
              own_flight_float=np.ones((3, 2, 4), np.float32),
              obj_count=np.ones((3, count), np.float32),
              v2_global=np.ones((3, 8), np.float32),
              v2_entity=np.ones((3, count, 2, 6), np.float32))
    xs = {k: torch.as_tensor(v[None]) for k, v in xs.items()}
    cfg = policy.Config(vocab_size=4, arch=2, inputs=2, width=16, core=16, heads=2,
                        layers=1, tile_layers=1, card_dim=4, temporal_layers=1,
                        compression_version='pair-unit1-aim1.5-cannon4-cap60-v1',
                        own_flight_slots=2, v2_global_dim=8, v2_entity_dim=6, v2_layout_sha256='a'*64)
    model = policy.build(cfg)
    output = model(xs)
    assert output['tokens'].shape[1] == 25 + count
    output['value'].square().sum().backward()
    for projection in (model.v2_global, model.v2_entity):
        assert projection.weight.grad.abs().sum() > 0
        assert torch.isfinite(projection.weight.grad).all()
    path = tmp_path / 'model.pt'
    torch.save(dict(config=asdict(cfg), model=model.state_dict(), vocab=[0, 1, 2, 3]), path)
    loaded = policy.load(path)[0]
    torch.testing.assert_close(output['value'], loaded(xs)['value'])


def test_complete_layout_binding_rejects_sample_and_wrong_hash(tmp_path):
    layout = dict(scope='full_verified_dataset', canonicalization='known-coordinates-own-view-v1',
                  encoding='named-signed-log-valid-v1', globals=dict(paths=['a'], width=2),
                  entities=dict(paths=['b'], width=2), source_shards=[dict(manifest_sha256='source')])
    encoded = json.dumps(layout).encode()
    (tmp_path / 'v2-layout.json').write_bytes(encoded)
    manifest = dict(input_profile='inputs-v2-history31-own2', lineage=[dict(manifest_sha256='source')])
    settings = dict(inputs=2, own_flight_slots=2, ability_labels=0, v2_global_dim=2, v2_entity_dim=2,
                    v2_layout_sha256=hashlib.sha256(encoded).hexdigest())
    assert load_layout(tmp_path, manifest, settings) == layout
    with pytest.raises(ValueError, match='hash'):
        load_layout(tmp_path, manifest, settings | {'v2_layout_sha256': 'wrong'})
    with pytest.raises(ValueError, match='lineage'):
        load_layout(tmp_path, manifest | {'lineage': []}, settings)
    layout['scope'] = 'sample'
    encoded = json.dumps(layout).encode()
    (tmp_path / 'v2-layout.json').write_bytes(encoded)
    settings['v2_layout_sha256'] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match='complete'):
        load_layout(tmp_path, manifest, settings)
