#!/usr/bin/env python3
"""A/B the SM90 packed-route KC pipeline change on fixed CUDA inputs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_attention


VARIANTS = (
    ("baseline", False),
    ("reuse_k_for_ak", True),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, choices=(124, 243, 345))
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames is not None and args.tokens is not None:
        parser.error("select either --frames or --tokens")
    if args.frames is not None:
        latent_frames = (args.frames - 5) // 17 * 5 + 2
        tokens = latent_frames * (768 // 16 // 2) * (1344 // 16 // 2)
    else:
        latent_frames = None
        tokens = 72_576 if args.tokens is None else args.tokens
    assert tokens > 0 and args.rounds > 0
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["H3_SPARK_REWEIGHT_FUSED"] = "1"
    b, t, h, d = 1, tokens, args.heads, 128
    n, words = (t + 63) // 64, ((t + 63) // 64 + 31) // 32
    q, k, v = (torch.randn((b, t, h, d), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    kc = torch.randn((b, n, h, d), device="cuda", dtype=torch.bfloat16)
    ak = torch.randn((b, 1, h, n, d), device="cuda", dtype=torch.bfloat16)
    av = torch.randn_like(ak)
    lm = torch.randn((b, 1, h, n), device="cuda", dtype=torch.float32)
    anchor = torch.randn((b, 1, h, d), device="cuda", dtype=torch.bfloat16)
    ranges = torch.tensor([[0, t]], device="cuda", dtype=torch.int64)
    mapping = torch.zeros((n,), device="cuda", dtype=torch.int64)
    bits = torch.rand((b, n, h, words, 32), device="cuda") < 0.10
    if n % 32:
        bits[..., -1, n % 32:] = False
    actual_route_density = float(bits.reshape(b, n, h, words * 32)[..., :n].float().mean().item())
    weights = (torch.ones(32, device="cuda", dtype=torch.int64) << torch.arange(32, device="cuda"))
    route = (bits.to(torch.int64) * weights).sum(-1).to(torch.int32).contiguous()
    del bits
    kwargs = dict(
        virtual_ranges=ranges, leaf_to_virtual=mapping, virtual_anchors=anchor,
        precomputed_summaries=(ak, av, lm), key_centroids=kc, route=route,
        skip_external_route_qk=True, force_local_blocks=1,
    )

    def call(name):
        _, reuse = next(v for v in VARIANTS if v[0] == name)
        os.environ["H3_SM90_REUSE_K_PIPELINE_FOR_AK"] = str(int(reuse))
        return virtual_q_attention(q, k, v, **kwargs)

    reference = call("baseline")
    torch.cuda.synchronize()
    accuracy = {}
    for name, _ in VARIANTS[1:]:
        actual = call(name)
        torch.cuda.synchronize()
        accuracy[name] = {
            "equal": bool(torch.equal(actual, reference)),
            "max_abs_error": float((actual.float() - reference.float()).abs().max().item()),
        }
        if not accuracy[name]["equal"]:
            raise AssertionError(f"{name} differs from baseline: {accuracy[name]}")

    # Both specializations are compiled. Alternate their order each round
    # to reduce clock/temperature drift without timing compilation.
    timings = {name: [] for name, _ in VARIANTS}
    for round_idx in range(args.rounds):
        order = [v[0] for v in VARIANTS]
        if round_idx % 2:
            order.reverse()
        for name in order:
            call(name)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call(name)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            timings[name].append(ms)
            print(f"round={round_idx} {name}={ms:.3f}ms", flush=True)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    report = {
        "gpu": torch.cuda.get_device_name(), "seed": args.seed, "frames": args.frames,
        "latent_frames": latent_frames, "tokens": t, "heads": h,
        "route_density_target": 0.10, "actual_route_density": actual_route_density,
        "rounds": args.rounds,
        "timings_ms": timings, "median_ms": medians,
        "speedup_vs_baseline": {name: medians["baseline"] / value for name, value in medians.items()},
        "accuracy": accuracy,
        "scope": "virtual_q_attention with fixed Q/K/V, route, and precomputed summaries; includes wrapper launches and output allocation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
