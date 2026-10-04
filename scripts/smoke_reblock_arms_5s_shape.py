#!/usr/bin/env python3
"""GPU reblock-only smoke for every registered 5s/768p ablation arm."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("SPARK_PROFILE_REBLOCK", "1")

import torch

from h3_sparse_attention.processor import H3SparseAttentionConfig, PackedLayout, _Controller
from h3_sparse_attention.spark_integration import (
    _landmark_tree_v2_combined_permutations,
    spark_reblock_profile_summary,
)
from scripts.reblock_ablation_5s768p import ARMS


GRID = (37, 24, 42)
TOKENS = 37_296
DIM = 128
REMAINDER = TOKENS % 64


def _layout(device: torch.device) -> PackedLayout:
    identity = torch.arange(TOKENS, device=device)
    t, h, w = torch.meshgrid(
        torch.arange(GRID[0], device=device),
        torch.arange(GRID[1], device=device),
        torch.arange(GRID[2], device=device), indexing="ij",
    )
    positions = torch.stack((t.flatten(), h.flatten(), w.flatten()), dim=1)
    return PackedLayout(identity, identity.clone(), GRID, TOKENS, TOKENS, positions)


def _integrity(permutation: torch.Tensor, inverse: torch.Tensor) -> dict:
    expected = torch.arange(TOKENS, device=permutation.device).expand_as(permutation)
    return {
        "complete_permutation": bool(torch.equal(permutation.sort(1).values, expected)),
        "inverse_roundtrip": bool(torch.equal(inverse.gather(1, permutation), expected)),
    }


def _plan_checks(plan, cfg, combined: torch.Tensor, inverse: torch.Tensor) -> dict:
    checks = _integrity(combined, inverse)
    checks.update({
        "plan_batch": plan.batch == (2 if cfg.landmark_tree_v2_layout_reuse == "independent" else 1),
        "plan_landmark_count": plan.landmark_count == cfg.landmark_tree_v2_landmark_count,
        "plan_landmark_mode": plan.landmark_mode == cfg.landmark_tree_v2_landmark_mode,
        "plan_distance": plan.distance == cfg.landmark_tree_v2_distance,
        "plan_initial_order": plan.initial_order == cfg.landmark_tree_v2_initial_order,
        "plan_proxy_iterations": plan.proxy_iterations == cfg.landmark_tree_v2_proxy_iterations,
        "plan_seed_rule": plan.seed_rule == cfg.landmark_tree_v2_proxy_seed_rule,
        "plan_update_rule": plan.update_rule == cfg.landmark_tree_v2_proxy_update_rule,
    })
    shared = cfg.landmark_tree_v2_layout_reuse != "independent"
    checks["shared_layout_exact"] = (
        bool(torch.equal(combined[0], combined[1]) and torch.equal(inverse[0], inverse[1]))
        if shared else True
    )
    return checks


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    query = torch.randn((1, TOKENS, 1, DIM), device=device, dtype=torch.bfloat16)
    key = torch.randn((1, TOKENS, 1, DIM), device=device, dtype=torch.bfloat16)
    layout = _layout(device)
    base = H3SparseAttentionConfig.spark(20)

    # Compile the production kernels before reporting rough per-arm timings.
    warm = _Controller(base)
    _landmark_tree_v2_combined_permutations(warm, query, key, layout)
    torch.cuda.synchronize(device)
    del warm

    output = {
        "device": torch.cuda.get_device_name(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "shape": [1, TOKENS, 1, DIM],
        "grid": list(GRID),
        "seed": 42,
        "arms": {},
    }
    baseline_tails = None
    for name, spec in ARMS.items():
        if name == "dense":
            output["arms"][name] = {
                "status": "not_applicable", "reason": "dense reference has no reblock plan"
            }
            print(f"{name}: not_applicable", flush=True)
            continue
        record = {"axis": spec.axis, "overrides": spec.overrides}
        try:
            cfg = dataclasses.replace(base, **spec.overrides)
            controller = _Controller(cfg)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            plan, metric_indices, combined, inverse = (
                _landmark_tree_v2_combined_permutations(controller, query, key, layout)
            )
            torch.cuda.synchronize(device)
            record["wall_ms"] = (time.perf_counter() - started) * 1e3
            profile = spark_reblock_profile_summary(controller)
            record["profile"] = profile
            record["checks"] = _plan_checks(plan, cfg, combined, inverse)
            record["metric_index_count"] = int(metric_indices.numel())
            tails = combined[:, -REMAINDER:].sort(1).values
            if name == "baseline":
                baseline_tails = tails.clone()
            if baseline_tails is not None:
                if cfg.landmark_tree_v2_layout_reuse == "q_from_k":
                    expected_tails = baseline_tails[0:1].expand_as(tails)
                elif cfg.landmark_tree_v2_layout_reuse == "k_from_q":
                    expected_tails = baseline_tails[1:2].expand_as(tails)
                else:
                    expected_tails = baseline_tails
                record["checks"]["tail_identity_frozen"] = bool(
                    torch.equal(tails, expected_tails)
                )
            if (cfg.landmark_tree_v2_m2_side != "both" or
                    cfg.landmark_tree_v2_m2_estimator != "hilbert_midpoint"):
                frozen = controller.spark_reblock_frozen_tail_indices
                record["checks"]["explicit_frozen_tail_present"] = frozen is not None
                expected_frozen = frozen
                if expected_frozen is not None and expected_frozen.shape[0] == 1:
                    expected_frozen = expected_frozen.repeat(2, 1)
                record["checks"]["explicit_frozen_tail_honored"] = bool(
                    expected_frozen is not None
                    and torch.equal(tails, expected_frozen.sort(1).values)
                )
            record["status"] = (
                "pass" if all(record["checks"].values()) else "check_failed"
            )
            print(f"{name}: {record['status']} {record['wall_ms']:.1f} ms", flush=True)
        except Exception as error:
            record.update({
                "status": "error", "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            print(f"{name}: ERROR {error!r}", flush=True)
        output["arms"][name] = record
        output_path = Path(os.environ.get(
            "REBLOCK_SMOKE_OUTPUT",
            "/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_20261002/smoke_gpu0.json",
        ))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    failures = [
        name for name, row in output["arms"].items()
        if row["status"] not in ("pass", "not_applicable")
    ]
    output["failures"] = failures
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"wrote {output_path}; failures={failures}", flush=True)


if __name__ == "__main__":
    main()
