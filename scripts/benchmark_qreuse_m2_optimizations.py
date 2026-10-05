#!/usr/bin/env python3
"""Benchmark the source-side M2 optimization against the previous behavior."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import h3_sparse_attention.spark_integration as integration
from h3_sparse_attention.landmark_direction import (
    landmark_direction_factors,
    landmark_single_direction_factor,
)
from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices
from h3_sparse_attention.processor import H3SparseAttentionConfig, PackedLayout, _Controller


def measure(call, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    grid = (21, 54, 64)
    tokens = 1
    for value in grid:
        tokens *= value
    ids = torch.arange(tokens, device=device)
    positions = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, device=device) for size in grid), indexing="ij"
        ), dim=-1,
    ).reshape(tokens, 3)
    layout = PackedLayout(ids, ids.clone(), grid, tokens, tokens, positions)
    torch.manual_seed(20261005)
    query = torch.randn(
        1, tokens, args.heads, 128, device=device, dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    flat_query = query.permute(0, 2, 1, 3).reshape(args.heads, tokens, 128)
    flat_key = key.permute(0, 2, 1, 3).reshape(args.heads, tokens, 128)
    indices = hilbert_midpoint_sample_indices(
        grid, tokens // 64, device=device
    )

    metric_both = measure(
        lambda: landmark_direction_factors(
            flat_query, flat_key, indices, m2_side="both"
        ),
        warmup=3,
        repeats=args.repeats,
    )
    metric_source = measure(
        lambda: landmark_single_direction_factor(
            flat_query, flat_key, indices, "key"
        ),
        warmup=3,
        repeats=args.repeats,
    )

    factor = landmark_single_direction_factor(
        flat_query, flat_key, indices, "key"
    ).to(torch.bfloat16)
    transformed = torch.empty_like(flat_key)
    identity = torch.eye(128, device=device, dtype=torch.bfloat16).expand(
        args.heads, -1, -1
    )
    identity_bmm = measure(
        lambda: torch.bmm(flat_key, identity, out=transformed),
        warmup=3,
        repeats=args.repeats,
    )
    direct_copy = measure(
        lambda: transformed.copy_(flat_key),
        warmup=3,
        repeats=args.repeats,
    )
    frozen_reference = measure(
        lambda: torch.bmm(flat_key, factor, out=transformed),
        warmup=3,
        repeats=args.repeats,
    )

    config = H3SparseAttentionConfig.spark(
        20, landmark_tree_v2_layout_reuse="q_from_k"
    )
    old_controller = _Controller(config)
    new_controller = _Controller(config)
    required = integration._required_reblock_m2_side

    def old_call():
        integration._required_reblock_m2_side = lambda m2_side, layout_reuse: m2_side
        try:
            return integration._landmark_tree_v2_combined_permutations(
                old_controller, query, key, layout, expand_shared=False
            )
        finally:
            integration._required_reblock_m2_side = required

    def new_call():
        return integration._landmark_tree_v2_combined_permutations(
            new_controller, query, key, layout, expand_shared=False
        )

    # Three warmups cover eager execution, capture, and replay for each plan.
    for _ in range(3):
        old_result = old_call()
        new_result = new_call()
    bitwise = (
        torch.equal(old_result[2], new_result[2])
        and torch.equal(old_result[3], new_result[3])
    )
    old_full = measure(old_call, warmup=0, repeats=args.repeats)
    new_full = measure(new_call, warmup=0, repeats=args.repeats)

    result = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name(device),
        "shape": [1, tokens, args.heads, 128],
        "repeats": args.repeats,
        "bitwise_permutation_and_inverse": bitwise,
        "metric_old_both": summary(metric_both),
        "metric_new_source_only": summary(metric_source),
        "metric_speedup": statistics.mean(metric_both) / statistics.mean(metric_source),
        "identity_bmm": summary(identity_bmm),
        "identity_copy": summary(direct_copy),
        "identity_speedup": statistics.mean(identity_bmm) / statistics.mean(direct_copy),
        "zero_tail_reference_transform_avoided": summary(frozen_reference),
        "full_old_emulation": summary(old_full),
        "full_new": summary(new_full),
        "full_speedup": statistics.mean(old_full) / statistics.mean(new_full),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
