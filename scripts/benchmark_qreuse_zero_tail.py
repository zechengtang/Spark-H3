#!/usr/bin/env python3
"""Measure the zero-remainder, unweighted shared-layout reblock path."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from h3_sparse_attention.processor import H3SparseAttentionConfig, PackedLayout, _Controller
from h3_sparse_attention.spark_integration import _landmark_tree_v2_combined_permutations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    grid = (21, 54, 64)
    tokens = 21 * 54 * 64
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
    controller = _Controller(H3SparseAttentionConfig.spark(
        20,
        landmark_tree_v2_layout_reuse="q_from_k",
        landmark_tree_v2_m2_side="none",
    ))

    def call():
        return _landmark_tree_v2_combined_permutations(
            controller, query, key, layout, expand_shared=False
        )

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    plans = [
        value for value in controller.rope_sol_key_clustering_static.values()
        if hasattr(value, "graph_active")
    ]
    print(json.dumps({
        "device": args.device,
        "gpu": torch.cuda.get_device_name(device),
        "shape": [1, tokens, args.heads, 128],
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "graph_active": bool(plans and plans[0].graph_active),
    }, indent=2))


if __name__ == "__main__":
    main()
