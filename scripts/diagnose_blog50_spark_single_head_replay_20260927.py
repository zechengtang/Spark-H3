#!/usr/bin/env python3
"""Replay one captured Spark head with either frozen or current implementation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(os.environ.get(
    "H3_CAPTURE_ROOT",
    "/autodl-fs/data/h3_experiments/blog50_spark_intermediate_20260927",
))
FROZEN = Path("/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920")


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("frozen", "current"):
        raise SystemExit("usage: replay.py frozen|current")
    mode = sys.argv[1]
    head_spec = os.environ.get("H3_REPLAY_HEADS")
    selected_heads = (tuple(int(value) for value in head_spec.split(","))
                      if head_spec else None)
    legacy_directions = os.environ.get("H3_REPLAY_LEGACY_DIRECTIONS") == "1"
    midpoint_mode = os.environ.get("H3_REPLAY_MIDPOINT_MODE")
    if midpoint_mode not in (None, "legacy", "fused"):
        raise ValueError("H3_REPLAY_MIDPOINT_MODE must be legacy or fused")
    repeats = int(os.environ.get("H3_REPLAY_REPEATS", "1"))
    if repeats < 1 or repeats > 4:
        raise ValueError("H3_REPLAY_REPEATS must be in 1..4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("replay requires GPU0 or GPU1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from h3_sparse_attention import H3SparseAttentionConfig
    from h3_sparse_attention import processor, spark_integration, virtual_q_permute
    from h3_sparse_attention import sol_topk_cutoff, sol_numerator_virtual_q
    direction_stats = {"calls": 0, "mismatched_elements": 0, "max_abs": 0.0}
    if legacy_directions:
        if mode != "current":
            raise ValueError("legacy-directions override is only for current implementation")
        from h3_sparse_attention import landmark_v2_fused_node
        from h3_sparse_attention.landmark_tree_v2_triton import indexed_interval_means
        from h3_sparse_attention.landmark_v2_cosine_fast import build_cosine_directions
        fused_directions = landmark_v2_fused_node.fused_midpoint_directions

        def frozen_style_directions(source, global_indices, child_capacities,
                                    *, fp8=False, landmarks=32):
            if fp8:
                raise NotImplementedError("the reproduction uses BF16 reblock features")
            centers, weights = indexed_interval_means(
                source, global_indices, landmarks, midpoint=True)
            baseline = build_cosine_directions(centers, weights, child_capacities)
            if os.environ.get("H3_TRACE_DIRECTION_DIFF") == "1":
                fused = fused_directions(source, global_indices, child_capacities,
                                         fp8=fp8, landmarks=landmarks)
                direction_stats["calls"] += 1
                direction_stats["mismatched_elements"] += int((baseline != fused).sum().item())
                direction_stats["max_abs"] = max(
                    direction_stats["max_abs"],
                    float((baseline - fused).abs().max().item()),
                )
            return baseline

        landmark_v2_fused_node.fused_midpoint_directions = frozen_style_directions

    data = torch.load(ROOT / "post_rope_head0_eval04_layer01.pt", map_location="cpu", weights_only=True)
    q, k, v = (
        (data[name][:, :, selected_heads] if selected_heads is not None else data[name]).cuda()
        for name in ("query", "key", "value")
    )
    old_cfg = json.loads((FROZEN / "protocol.json").read_text())["configs"]["topk10_reblock_global_reweight"]
    extra = dict(sol_tail_granularity="query", sol_global_anchor_dtype="bfloat16",
                 sol_reweight_summary_math="tensorcore", sol_reweight_logmass_key="stored",
                 sol_reweight_components="full", sol_route_topk_execution="threshold",
                 sol_video_tail_mode="dense", sol_legacy_full_query=False)
    if midpoint_mode is not None:
        extra["landmark_tree_v2_midpoint_direction_mode"] = midpoint_mode
    fields = {field.name for field in dataclasses.fields(H3SparseAttentionConfig)}
    cfg = H3SparseAttentionConfig(**{key: value for key, value in {**old_cfg, **extra}.items() if key in fields})
    ctrl = processor._Controller(cfg)
    ctrl.evaluation_index = 4
    n = int(data["sequence_length"])
    ids = torch.arange(n, device="cuda", dtype=torch.int64)
    layout = processor.PackedLayout(
        permutation=ids, inverse_permutation=ids, grid=tuple(data["grid"]),
        video_tokens=int(data["video_tokens"]), sequence_length=n,
        video_positions=ids[: int(data["video_tokens"])],
    )
    stages: dict[str, torch.Tensor] = {}

    iteration = 0
    def record(name: str, tensor: torch.Tensor) -> None:
        stages[f"repeat{iteration}_{name}"] = tensor.detach().contiguous().cpu()

    old_builder = spark_integration._landmark_tree_v2_qk_block_permutations
    def builder(*args, **kwargs):
        result = old_builder(*args, **kwargs)
        record("query_permutation", result[0])
        record("key_permutation", result[2])
        return result
    spark_integration._landmark_tree_v2_qk_block_permutations = builder

    old_anchor = virtual_q_permute.permute_with_virtual_anchors
    def anchors(*args, **kwargs):
        result = old_anchor(*args, **kwargs)
        record("global_anchor", result[1])
        return result
    virtual_q_permute.permute_with_virtual_anchors = anchors

    old_cutoff = sol_topk_cutoff.gemm_radix_topk_cutoff
    def cutoff(*args, **kwargs):
        result = old_cutoff(*args, **kwargs)
        record("threshold", result[0])
        return result
    sol_topk_cutoff.gemm_radix_topk_cutoff = cutoff

    old_attention = sol_numerator_virtual_q.virtual_q_attention
    def attention(q1, k1, v1, **kwargs):
        # This is the exact Q/K/V and anchor payload reaching the reweight
        # kernel. Recompute summaries separately for an inspectable checkpoint.
        a = kwargs.get("virtual_anchors")
        if a is not None:
            summary_kwargs = {}
            if mode == "current":
                summary_kwargs = dict(summary_math="tensorcore", logmass_key="stored",
                                      reweight_components="full")
            ak, av, lm = sol_numerator_virtual_q.virtual_summaries(a, k1, v1, **summary_kwargs)
            record("weighted_key", ak)
            record("weighted_value", av)
            record("logmass", lm)
        output = old_attention(q1, k1, v1, **kwargs)
        record("sparse_output", output)
        return output
    sol_numerator_virtual_q.virtual_q_attention = attention

    with torch.no_grad():
        for iteration in range(repeats):
            result = spark_integration.spark_attention(
                ctrl, q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3), layout, 1, return_bthd=True,
            )
            record("final_output", result)
        torch.cuda.synchronize()

    suffix = ("_legacydirs" if legacy_directions else "")
    if os.environ.get("H3_TRACE_DIRECTION_DIFF") == "1":
        suffix += "_trace"
    if midpoint_mode is not None:
        suffix += f"_midpoint_{midpoint_mode}"
    if selected_heads is not None:
        suffix += "_heads" + "-".join(str(value) for value in selected_heads)
    target = ROOT / (f"{mode}{suffix}_repeat{repeats}" if repeats > 1 else f"{mode}{suffix}")
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    for name, value in stages.items():
        torch.save(value, target / f"{name}.pt")
    (target / "manifest.json").write_text(json.dumps({
        "mode": mode,
        "selected_heads": selected_heads,
        "legacy_directions": legacy_directions,
        "direction_stats": direction_stats,
        "config": dataclasses.asdict(cfg),
        "stages": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": digest(value)} for name, value in stages.items()},
        "attention_summary": ctrl.summary(),
    }, indent=2, default=str) + "\n")
    print("REPLAY", mode, {name: digest(value) for name, value in stages.items()}, flush=True)
    if direction_stats["calls"]:
        print("DIRECTION_DIFF", direction_stats, flush=True)


if __name__ == "__main__":
    main()
