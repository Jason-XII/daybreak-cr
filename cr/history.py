import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(self.norm(x)).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True)
        x = x + self.out(a.transpose(1,2).reshape(b,t,d))
        return x + self.ff(self.ff_norm(x))


class TemporalCore(nn.Module):
    def __init__(self, width, heads, layers=4, max_length=1280):
        super().__init__()
        self.position = nn.Parameter(torch.randn(max_length,width)*0.02)
        self.blocks = nn.ModuleList([TemporalBlock(width,heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x, state=None):
        if state is not None or x.shape[1] > len(self.position):
            raise ValueError('Full-history training expects no cached state and length <=1280')
        x = x + self.position[:x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x), None
