"""GPU2 single-chunk A/B for experimental BSA tile BF16 materializer."""

import argparse
import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck

from profile_diffusers_vs_comfy_spark_20260925 import (
    cuda_benchmark, load_capture,
)


def compare(a, b):
    diff = (a.float() - b.float()).abs()
    return {"bitwise": bool(torch.equal(a, b)),
            "equal_fraction": float((a == b).float().mean()),
            "max_abs": float(diff.max()),
            "rmse": float(diff.square().mean().sqrt())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, q0, k0, v0 = load_capture(torch)
    tokens, heads, dim = 16384, q0.shape[2], q0.shape[3]
    q0, k0, v0 = (x[:, :tokens].contiguous() for x in (q0, k0, v0))
    projected = torch.cat([x.reshape(tokens, heads * dim) for x in (q0, k0, v0)], -1)
    del q0, k0, v0
    generator = torch.Generator(device="cuda").manual_seed(20260928)
    theta = torch.randn((tokens, 48), device="cuda", generator=generator) * 0.1
    freqs = torch.zeros((1, tokens, 1, 48, 2, 2), device="cuda", dtype=torch.float32)
    freqs[0, :, 0, :, 0, 0] = theta.cos()
    freqs[0, :, 0, :, 0, 1] = -theta.sin()
    freqs[0, :, 0, :, 1, 0] = theta.sin()
    freqs[0, :, 0, :, 1, 1] = theta.cos()
    qw = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
    kw = torch.ones_like(qw)
    fab = ck._packed_rope_fab(freqs, tokens, 96)
    old = [torch.empty((1, tokens, heads, dim), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    new = [torch.empty_like(old[0]) for _ in range(3)]
    qc, kc, vc = projected.split(heads * dim, dim=-1)
    qc, kc = (x.view(1, tokens, heads, dim) for x in (qc, kc))
    stream = torch.cuda.current_stream().cuda_stream
    wrap = ck._wrap_for_dlpack

    def baseline():
        ck._C.rms_rope(*(wrap(t) for t in (qc, kc, freqs, qw, kw, old[0], old[1])),
                       1e-6, stream, True, 96)
        old[2].copy_(vc.view(1, tokens, heads, dim))

    def candidate():
        ck._C.bsa_materialize_qkv_chunk(
            *(wrap(t) for t in (projected, fab, qw, kw, *new)),
            1e-6, 96, 0, tokens, tokens, heads, stream)

    baseline(); candidate(); torch.cuda.synchronize()
    result = {"shape": [1, tokens, heads, dim], "gpu": torch.cuda.get_device_name(),
              "input": "first 16K rows from captured real post-RoPE QKV, used as materializer input; randomized valid RoPE",
              "parity": {name: compare(a, b) for name, a, b in zip("qkv", old, new)},
              "baseline": cuda_benchmark(torch, baseline, warmup=5, iterations=30),
              "candidate": cuda_benchmark(torch, candidate, warmup=5, iterations=30)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"parity": result["parity"],
                      "baseline_ms": result["baseline"]["median_ms"],
                      "candidate_ms": result["candidate"]["median_ms"]}))


if __name__ == "__main__":
    main()
