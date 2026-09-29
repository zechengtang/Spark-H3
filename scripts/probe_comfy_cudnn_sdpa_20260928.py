"""Verify and time Flash/cuDNN SDPA with ComfyUI H3's QKV view and call semantics.

Run in a separate process with CUDA_VISIBLE_DEVICES selecting one GPU. This
does not modify ComfyUI or Spark production code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def kernel_names(fn):
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        with torch.inference_mode():
            out = fn()
            torch.cuda.synchronize()
    names = sorted({event.name for event in prof.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA})
    return names, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=73573)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    n, heads, dim = args.length, 56, 128
    # MiniMax H3 projects QKV together, then splits and transposes its views.
    qkv = torch.randn((n, 3 * heads * dim), device='cuda', dtype=torch.bfloat16)
    q, k, v = (x.view(n, heads, dim).transpose(0, 1).unsqueeze(0)
               for x in qkv.split(heads * dim, dim=-1))
    kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False)
    import comfy.ops

    calls = {
        'comfy_default': lambda: comfy.ops.scaled_dot_product_attention(q, k, v, **kwargs),
        'flash_direct': lambda: direct(SDPBackend.FLASH_ATTENTION, q, k, v, kwargs),
        'cudnn_direct': lambda: direct(SDPBackend.CUDNN_ATTENTION, q, k, v, kwargs),
    }
    records = {}
    for name, fn in calls.items():
        try:
            with torch.inference_mode():
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
                outputs = []
                for start, end in zip(starts, ends):
                    start.record()
                    out = fn()
                    end.record()
                    outputs.append(out)
                torch.cuda.synchronize()
                ms = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            names, prof_out = kernel_names(fn)
            records[name] = dict(status='ok', ms=ms,
                                 cuda_kernels=names,
                                 sample=prof_out.flatten()[:16].float().tolist(),
                                 shape=list(prof_out.shape))
        except Exception as exc:
            records[name] = dict(status='error', error=repr(exc))
    if all(records[k]['status'] == 'ok' for k in ('flash_direct', 'cudnn_direct')):
        with torch.inference_mode():
            a, b = calls['flash_direct'](), calls['cudnn_direct']()
            torch.cuda.synchronize()
            diff = (a.float() - b.float()).abs()
            records['comparison'] = dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                                         same_fraction=float((a == b).float().mean()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(length=n, heads=heads, dim=dim,
                                           dtype=str(q.dtype), qkv_stride=list(q.stride()),
                                           torch_version=torch.__version__,
                                           cudnn_version=torch.backends.cudnn.version(),
                                           gpu=torch.cuda.get_device_name(),
                                           records=records), indent=2))
    print(args.output, flush=True)
    for name, row in records.items():
        print(name, row.get('status'), row.get('ms'), row.get('cuda_kernels'), flush=True)


def direct(backend, q, k, v, kwargs):
    with sdpa_kernel(backend):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)


if __name__ == '__main__':
    main()
