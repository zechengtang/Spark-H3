"""Stage-level ablation of Sol all-exact and ComfyUI Flash SDPA on captured H3 QKV."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck

from profile_diffusers_vs_comfy_spark_20260925 import load_capture

sys.path.insert(0, "/autodl-fs/data/h3_repos/ComfyUI")
import comfy.ops


def bench(fn, n=12):
    for _ in range(3):
        out = fn()
        del out
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del out
    return samples


def profile(fn):
    out = fn()
    del out
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
        del out
    rows = []
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            rows.append({"name": event.name, "duration_ms": event.device_time / 1000})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=0,
                        help="Use a leading slice of the real capture; zero uses all tokens")
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, q, k, v = load_capture(torch)
    if args.tokens:
        if not 0 < args.tokens <= q.shape[1]:
            raise ValueError("tokens must be within the capture length")
        q, k, v = (x[:, :args.tokens].contiguous() for x in (q, k, v))
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))

    def bsa():
        return ck.sol_attn(q, k, v, tau=-1e9, tail=False)

    def sdpa_raw():
        return comfy.ops.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False
        )

    def sdpa_converted():
        return sdpa_raw().transpose(1, 2).contiguous()

    raw_out = sdpa_raw()
    def layout_only():
        return raw_out.transpose(1, 2).contiguous()

    calls = dict(bsa=bsa, sdpa_raw=sdpa_raw,
                 sdpa_converted=sdpa_converted, layout_only=layout_only)
    with torch.inference_mode():
        for fn in calls.values():
            out = fn()
            del out
        torch.cuda.synchronize()
        timings = {}
        for name in ("bsa", "sdpa_raw", "sdpa_converted", "layout_only"):
            timings[name] = bench(calls[name])
            print(name, statistics.median(timings[name]), flush=True)
        profiles = {name: profile(calls[name]) for name in ("bsa", "sdpa_converted")}

    result = {
        "shape": list(q.shape),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "timings_ms": timings,
        "median_ms": {name: statistics.median(vals) for name, vals in timings.items()},
        "profiler_cuda_kernels": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
