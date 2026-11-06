import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    fixed = torch.load(args.input, map_location='cpu', weights_only=True)
    results = []
    reference = None
    for mode in ('fp64-math','fp32-math','fp32-efficient','fp32-efficient-repeat'):
        dtype = torch.float64 if mode.startswith('fp64') else torch.float32
        q,k,v = [fixed[key].to(device='cuda',dtype=dtype).requires_grad_() for key in ('q','k','v')]
        backend = SDPBackend.MATH if 'math' in mode else SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(q,k,v,attn_mask=fixed['mask'].to(device='cuda',dtype=dtype))
        gradients = torch.autograd.grad(output,(q,k,v),grad_outputs=fixed['upstream'].to(device='cuda',dtype=dtype))
        if reference is None:
            reference = output.detach().double()
        row = dict(mode=mode, output_relative_l2=float((output.double()-reference).norm()/reference.norm()),
                   gradients={key:dict(norm=float(g.double().norm()),finite=bool(torch.isfinite(g).all()))
                              for key,g in zip(('q','k','v'),gradients)})
        results.append(row)
        print('ASTRA_MINIMAL '+json.dumps(row),flush=True)
    args.out.write_text(json.dumps(dict(torch=torch.__version__,pid=os.getpid(),input=str(args.input),
        tf32=False,q_shape=list(fixed['q'].shape),kv_shape=list(fixed['k'].shape),results=results),indent=2))
    print('ASTRA_MINIMAL_FINISHED',flush=True)


if __name__ == '__main__':
    main()
