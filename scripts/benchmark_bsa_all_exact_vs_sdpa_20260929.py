"""Compare BSA/Sol all-exact attention with ComfyUI's default SDPA on real H3 QKV."""
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


def time_call(fn, n=8):
    values = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
        del out
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))

    def bsa():
        return ck.sol_attn(q, k, v, tau=-1e9, tail=False)

    def sdpa():
        out = comfy.ops.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).contiguous()

    calls = {"bsa_all_exact": bsa, "comfy_default_sdpa": sdpa}
    with torch.inference_mode():
        for _ in range(2):
            for fn in calls.values():
                out = fn()
                del out
        torch.cuda.synchronize()
        samples = {key: [] for key in calls}
        for order in (list(calls), list(reversed(calls)), list(calls)):
            for name in order:
                vals = time_call(calls[name], n=5)
                samples[name].extend(vals)
                print(name, [round(x, 3) for x in vals], flush=True)
        a = bsa()
        b = sdpa()
        delta = (a.float() - b.float()).abs()
        numeric = {
            "mean_abs": delta.mean().item(),
            "max_abs": delta.max().item(),
            "mean_ref_abs": b.float().abs().mean().item(),
        }
        del a, b, delta
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = sdpa()
            torch.cuda.synchronize()
            del out
        kernels = sorted({event.name for event in prof.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})

    medians = {name: statistics.median(vals) for name, vals in samples.items()}
    result = {
        "capture": str(data.get("source", "attention_input_gpu1.pt")),
        "shape_bthd": list(q.shape),
        "dtype": str(q.dtype),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "scope": "attention call on identical post-RoPE QKV; includes SDPA output layout conversion; excludes projection, model loading and full denoising",
        "bsa_configuration": {"tau": -1e9, "tail": False},
        "samples_ms": samples,
        "median_ms": medians,
        "bsa_over_sdpa": medians["bsa_all_exact"] / medians["comfy_default_sdpa"],
        "numeric_difference": numeric,
        "sdpa_cuda_kernels": kernels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("shape_bthd", "median_ms", "bsa_over_sdpa", "numeric_difference", "sdpa_cuda_kernels")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
