from dataclasses import asdict

import pytest
import torch

from cr import policy
from cr.splitquery import SplitQueryBlock, migrate_checkpoint
from synthetic import arrays


def test_split_condition_has_no_direct_residual_path():
    block = SplitQueryBlock(policy.Block(24, 3, cross=True))
    with torch.no_grad():
        block.o.weight.zero_()
        block.o.bias.zero_()
        for parameter in block.ff.parameters():
            parameter.zero_()
    position, memory = torch.randn(2, 5, 24), torch.randn(2, 7, 24)
    mask = torch.zeros(2, 3, 5, 7)
    for scale in (1, 1e6):
        output = block(position, memory, mask, torch.randn(2, 1, 24) * scale)
        torch.testing.assert_close(output, position, rtol=0, atol=0)


def test_split_query_formula_and_all_branch_gradients():
    old = policy.Block(24, 3, cross=True)
    block = SplitQueryBlock(old)
    for name, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, block.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(block.condition_q.weight, old.q.weight, rtol=0, atol=0)
    assert float(block.alpha().detach()) == .5
    position, memory, condition = torch.randn(2, 5, 24), torch.randn(2, 7, 24), torch.randn(2, 1, 24)
    mask = torch.zeros(2, 3, 5, 7)
    mask[..., -1] = -torch.inf
    output = block(position, memory, mask, condition)
    q = block.q(block.norm(position)) + block.alpha() * block.condition_q(block.condition_norm(condition))
    k, v = block.kv(block.memory_norm(memory)).view(2, 7, 2, 3, 8).unbind(2)
    attended = torch.nn.functional.scaled_dot_product_attention(q.view(2, 5, 3, 8).transpose(1, 2),
        k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    residual = position + block.o(attended.transpose(1, 2).reshape(2, 5, 24))
    torch.testing.assert_close(output, residual + block.ff(block.ff_norm(residual)), rtol=0, atol=0)
    output.square().mean().backward()
    for parameter in (block.q.weight, block.condition_q.weight, block.condition_norm.weight, block.alpha_raw):
        assert parameter.grad.norm() > 0
        assert torch.isfinite(parameter.grad).all()
    for alpha in (0, 1, -1):
        with pytest.raises(ValueError):
            SplitQueryBlock(old, alpha)


def test_split_checkpoint_migration_load_and_optimizer_reset(tmp_path):
    config = policy.Config(vocab_size=4, arch=2, inputs=1, width=16, core=16,
                           heads=2, layers=1, tile_layers=1, card_dim=4)
    base = policy.build(config)
    original = dict(config=asdict(config), model=base.state_dict(), vocab=[0, 1, 2, 3],
                    optimizer={'baseline': 'must not migrate'}, step=30000, settings={})
    saved = migrate_checkpoint(original)
    assert 'optimizer' not in saved
    assert saved['migration']['optimizer_reset']
    assert not original['config']['split_query']
    path = tmp_path / 'split.pt'
    torch.save(saved, path)
    model = policy.load(path)[0]
    assert isinstance(model.tile_blocks[0], SplitQueryBlock)
    xs, _ = arrays(3, 0)
    xs = {k: torch.as_tensor(v[None]) for k, v in xs.items()}
    output = model(xs)
    scores = model.tiles(output, xs, torch.zeros(3, dtype=torch.long))
    scores.square().mean().backward()
    assert model.tile_blocks[0].condition_q.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='already'):
        migrate_checkpoint(saved)
