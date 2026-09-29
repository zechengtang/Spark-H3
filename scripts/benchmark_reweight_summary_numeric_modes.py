#!/usr/bin/env python3
"""Isolated, allocation-free summary-kernel timing for precision ablations."""
from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.testing

from h3_sparse_attention.sol_numerator_virtual_q import _launch_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.tokens <= 0 or args.heads <= 0 or args.repeats <= 0:
        parser.error("tokens, heads and repeats must be positive")
    torch.manual_seed(20260926)
    device = "cuda"
    t, h, n, d = args.tokens, args.heads, triton.cdiv(args.tokens, 64), 128
    anchor = torch.randn(1, 1, h, d, device=device, dtype=torch.float32)
    k = torch.randn(1, t, h, d, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ak = torch.empty(1, 1, h, n, d, device=device, dtype=torch.bfloat16)
    av = torch.empty_like(ak)
    lm = torch.empty(1, 1, h, n, device=device, dtype=torch.float32)
    results = {}
    for math_mode in ("tensorcore", "comfy_fp32"):
        for logmass_key in ("stored", "pre_round"):
            def call():
                _launch_summaries(
                    anchor, k, v, ak, av, lm, parent_start=0, parent_count=1,
                    parent_stride=1, head_stride=h, head_start=0,
                    summary_math=math_mode, logmass_key=logmass_key)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            results[f"{math_mode}/{logmass_key}"] = triton.testing.do_bench(
                call, warmup=200, rep=args.repeats)
    print(json.dumps({"tokens": t, "heads": h, "blocks": n,
                      "milliseconds": results}, indent=2))


if __name__ == "__main__":
    main()
