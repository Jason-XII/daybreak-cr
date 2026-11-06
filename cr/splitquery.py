import copy
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class SplitQueryBlock(nn.Module):
    def __init__(self, old, alpha_initial=0.5):
        super().__init__()
        if not 0 < alpha_initial < 1:
            raise ValueError('alpha_initial must be strictly between zero and one')
        self.heads = old.heads
        for name in ('norm', 'ff_norm', 'memory_norm', 'q', 'kv', 'o', 'ff'):
            setattr(self, name, copy.deepcopy(getattr(old, name)))
        self.condition_norm = copy.deepcopy(old.norm)
        self.condition_q = copy.deepcopy(old.q)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(alpha_initial / (1 - alpha_initial))))

    def alpha(self):
        return self.alpha_raw.sigmoid()

    def forward(self, position, memory, mask, condition):
        n, length, width = position.shape
        query = self.q(self.norm(position)) + self.alpha() * self.condition_q(self.condition_norm(condition))
        query = query.view(n, length, self.heads, -1).transpose(1, 2)
        key, value = self.kv(self.memory_norm(memory)).view(n, memory.shape[1], 2, self.heads, -1).unbind(2)
        attended = F.scaled_dot_product_attention(query, key.transpose(1, 2), value.transpose(1, 2),
                                                  attn_mask=mask.to(query.dtype))
        # The condition has no direct additive path into the residual stream.
        position = position + self.o(attended.transpose(1, 2).reshape(n, length, width))
        return position + self.ff(self.ff_norm(position))


def split_tiles(self, out, x, slot):
    from cr.policy import HAND, CELLS
    tokens = out['tokens']
    n, _, width = tokens.shape
    chosen = tokens[:, HAND].gather(1, slot.view(n, 1, 1).expand(-1, 1, width))
    condition = self.tile_query(out['core']).unsqueeze(1) + chosen
    # Match the dense residual layout of subsequent blocks and the original tile path.
    position = self.cells.unsqueeze(0).expand(n, -1, -1).contiguous()
    everywhere = torch.ones(n, CELLS, dtype=torch.bool, device=position.device)
    mask = self.tile_bias(self.cell_xy.expand(n, -1, -1), out['xy'], everywhere, out['known'], out['present'])
    for block in self.tile_blocks:
        position = block(position, tokens, mask, condition)
    with torch.autocast(position.device.type, enabled=False):
        scores = self.tile(self.tile_norm(position.float())).squeeze(-1)
    if 'legal' in x:
        legal = x['legal'].reshape(n, 4, CELLS).gather(1, slot.view(n, 1, 1).expand(-1, 1, CELLS)).squeeze(1)
        scores = scores.masked_fill(~legal, float('-inf'))
    return scores


def install(model, alpha_initial=0.5):
    if any(isinstance(block, SplitQueryBlock) for block in model.tile_blocks):
        raise ValueError('SplitQuery is already installed')
    model.tile_blocks = nn.ModuleList([SplitQueryBlock(block, alpha_initial) for block in model.tile_blocks])
    model.tiles = MethodType(split_tiles, model)
    model.config.split_query = True
    model.config.split_alpha = alpha_initial
    return model


def migrate_checkpoint(saved, alpha_initial=0.5):
    from dataclasses import asdict
    from cr import policy
    if saved['config'].get('split_query'):
        raise ValueError('Checkpoint already uses SplitQuery')
    config = policy.Config(**saved['config'])
    if config.arch != 2 or config.tile_layers <= 0:
        raise ValueError('Migration requires TokenPolicy tile attention blocks')
    model = policy.build(config)
    model.load_state_dict(saved['model'], strict=True)
    install(model, alpha_initial)
    config.split_query = True
    config.split_alpha = alpha_initial
    # A new branch changes parameter registration; never inherit the baseline optimizer.
    result = {key: copy.deepcopy(saved[key]) for key in ('vocab', 'step', 'settings') if key in saved}
    result.update(config=asdict(config), model=model.state_dict(),
                  migration=dict(source_step=saved.get('step'), optimizer_reset=True, alpha_initial=alpha_initial))
    if 'settings' in result:
        result['settings'].update(split_query=1, split_alpha=alpha_initial)
    return result
