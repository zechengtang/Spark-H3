#!/usr/bin/env python3
"""Fixed-input diagnostics for fused midpoint directions and external routing.

This script is intentionally read-only with respect to historical artifacts.  It
loads two existing post-RoPE captures, replays route construction and the SM120
attention implementations, and writes only compact JSON metrics.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch


ROOT = Path("/autodl-fs/data/h3_experiments/check_fused_extern_20261004")
CAPTURE_10S = Path(
    "/autodl-fs/data/h3_experiments/blog50_spark_intermediate_allheads_20260927/"
    "post_rope_head0_eval04_layer01.pt"
)
STAGES_10S = CAPTURE_10S.parent
CAPTURE_5S = Path(
    "/autodl-fs/data/h3_experiments/reblock_approx_10prompt_5s768p_20261002/"
    "captures/case01_eval04_layer01.pt"
)
HEADS_10S = (13, 51, 54)


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    assert a.shape == b.shape
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = torch.linalg.vector_norm(af)
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "bitwise_equal": bool(torch.equal(a, b)),
        "mismatched_elements": int(torch.count_nonzero(a != b).item()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "relative_l2": float((torch.linalg.vector_norm(af - bf) / denom).item()),
        "finite_a": bool(torch.isfinite(af).all().item()),
        "finite_b": bool(torch.isfinite(bf).all().item()),
    }


def unpack_route(route: torch.Tensor, blocks: int) -> torch.Tensor:
    bits = torch.arange(32, device=route.device, dtype=torch.int32)
    unpacked = ((route[..., None] >> bits) & 1).bool().flatten(-2)
    return unpacked[..., :blocks]


def make_config(direction: str, execution: str):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        20,
        warmup_percent=20.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=0.1,
        sol_log_density=False,
        sol_force_local_blocks=False,
        sol_tail_granularity="query",
        sol_video_tail_mode="dense",
        sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore",
        sol_reweight_logmass_key="stored",
        sol_reweight_components="full",
        landmark_tree_v2_midpoint_direction_mode=direction,
        sol_route_topk_execution=execution,
    )


def layout_for(q: torch.Tensor, video_tokens: int, grid: tuple[int, int, int]):
    from h3_sparse_attention.processor import PackedLayout

    ids = torch.arange(q.shape[1], device=q.device, dtype=torch.int64)
    return PackedLayout(
        permutation=ids,
        inverse_permutation=ids,
        grid=grid,
        video_tokens=video_tokens,
        sequence_length=q.shape[1],
        video_positions=ids[:video_tokens],
    )


def build_permutations(q, k, layout, direction: str):
    from h3_sparse_attention.processor import _Controller
    from h3_sparse_attention.spark_integration import _landmark_tree_v2_qk_block_permutations

    controller = _Controller(make_config(direction, "threshold"))
    controller.evaluation_index = 4
    result = _landmark_tree_v2_qk_block_permutations(controller, q, k, layout)
    return result[:4], controller.landmark_reblock_hierarchy


def perm_metrics(legacy, fused) -> dict[str, Any]:
    names = ("query_permutation", "query_inverse", "key_permutation", "key_inverse")
    result = {}
    for name, a, b in zip(names, legacy, fused, strict=True):
        row = tensor_metrics(a, b)
        # Index differences are more legible than floating relative errors.
        row["different_fraction"] = row["mismatched_elements"] / a.numel()
        result[name] = row
    return result


def permute_qkv(q, k, v, permutation, video_tokens):
    from h3_sparse_attention.spark_integration import _headwise_permute_video_tokens
    from h3_sparse_attention.landmark_tree_v2_triton import headwise_permute_pair_bthd

    qp, _, kp, _ = permutation
    q2 = _headwise_permute_video_tokens(q, qp, video_tokens=video_tokens)
    k2, v2 = headwise_permute_pair_bthd(k, v, kp, video_tokens=video_tokens)
    return q2, k2, v2


@torch.no_grad()
def route_metrics(q, k, *, video_tokens: int) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    from h3_sparse_attention.sol_numerator_virtual_q import reduce_virtual_key_centroids
    from h3_sparse_attention.sol_topk_cutoff import (
        _gemm_score_map_prefix,
        gemm_radix_topk_cutoff,
        gemm_topk_packed_route,
    )

    kc = reduce_virtual_key_centroids(k)
    sink_tokens = q.shape[1] - video_tokens
    query_tokens = (video_tokens // 64) * 64
    query_blocks = query_tokens // 64
    candidate_blocks = video_tokens // 64
    blocks = math.ceil(q.shape[1] / 64)
    threshold, threshold_stats = gemm_radix_topk_cutoff(
        q, kc, video_tokens=video_tokens, sink_tokens=sink_tokens,
        topk_ratio=0.1, _return_first_excluded=True, collect_tie_stats=True,
    )
    packed, packed_stats = gemm_topk_packed_route(
        q, kc, video_tokens=video_tokens, topk_ratio=0.1,
        query_tokens=query_tokens,
    )
    scores = _gemm_score_map_prefix(
        q, kc, query_blocks=query_blocks, candidate_blocks=candidate_blocks,
    )
    threshold_mask = scores > threshold[:, :query_blocks, :, None]
    packed_mask = unpack_route(packed, blocks)[..., :candidate_blocks]
    intersection = (threshold_mask & packed_mask).sum(-1)
    union = (threshold_mask | packed_mask).sum(-1)
    disagreement = threshold_mask ^ packed_mask
    target = packed_stats["target_topk_blocks_per_query"]
    ordered = scores.sort(dim=-1, descending=True).values
    selected_min = ordered[..., target - 1]
    excluded_max = ordered[..., target]
    gaps = selected_min - excluded_max
    diff_pos = disagreement.nonzero(as_tuple=False)
    examples = diff_pos[:32].cpu().tolist()
    counts_threshold = threshold_mask.sum(-1)
    counts_packed = packed_mask.sum(-1)
    metrics = {
        "shape": list(q.shape),
        "video_tokens": video_tokens,
        "context_tokens": sink_tokens,
        "blocks": blocks,
        "candidate_complete_video_blocks": candidate_blocks,
        "query_sparse_blocks": query_blocks,
        "video_tail_tokens": video_tokens % 64,
        "context_tail_tokens": sink_tokens % 64,
        "target_topk": target,
        "threshold_stats": {k: v for k, v in threshold_stats.items() if not k.startswith("_")},
        "packed_stats": packed_stats,
        "threshold_count_min_max": [int(counts_threshold.min()), int(counts_threshold.max())],
        "packed_count_min_max": [int(counts_packed.min()), int(counts_packed.max())],
        "different_bits": int(disagreement.sum().item()),
        "different_rows": int(disagreement.any(-1).sum().item()),
        "rows": int(disagreement.any(-1).numel()),
        "jaccard_mean": float((intersection.float() / union.clamp_min(1)).mean().item()),
        "jaccard_min": float((intersection.float() / union.clamp_min(1)).min().item()),
        "boundary_gap_min": float(gaps.min().item()),
        "boundary_gap_mean": float(gaps.mean().item()),
        "boundary_exact_ties": int((gaps == 0).sum().item()),
        "difference_examples_bqhk": examples,
        "sink_policy": (
            "mask intentionally omits sink/context; main kernel forces blocks "
            f"[{video_tokens // 64},{math.ceil(q.shape[1] / 64)}) exact"
        ),
        "local_policy": "disabled in the historical four-arm configuration",
    }
    return metrics, threshold, packed


@torch.no_grad()
def attention_outputs(q, k, v, layout, permutation, hierarchy):
    from h3_sparse_attention.processor import _Controller
    from h3_sparse_attention import spark_integration

    original = spark_integration._landmark_tree_v2_qk_block_permutations

    def fixed(controller, *_args, **_kwargs):
        controller.landmark_reblock_hierarchy = hierarchy
        empty = {"fixed_capture_permutation": True}
        return (*permutation, empty, empty)

    outputs = {}
    try:
        spark_integration._landmark_tree_v2_qk_block_permutations = fixed
        for execution in ("threshold", "packed_external", "packed_external_no_route_qk"):
            controller = _Controller(make_config("legacy", execution))
            controller.evaluation_index = 4
            outputs[execution] = spark_integration.spark_attention_bthd(
                controller, q.clone(), k.clone(), v.clone(), layout, layer=1,
                return_bthd=True,
            ).cpu()
    finally:
        spark_integration._landmark_tree_v2_qk_block_permutations = original
    return {
        "packed_external_vs_no_route_qk": tensor_metrics(
            outputs["packed_external"], outputs["packed_external_no_route_qk"]
        ),
        "threshold_vs_packed_external": tensor_metrics(
            outputs["threshold"], outputs["packed_external"]
        ),
        "threshold_vs_no_route_qk": tensor_metrics(
            outputs["threshold"], outputs["packed_external_no_route_qk"]
        ),
    }


def load_5s():
    data = torch.load(CAPTURE_5S, map_location="cpu", weights_only=True)
    # The capture stores video Q/K/V and context K/V separately.  Context Q is
    # irrelevant to the sparse prefix; zeros make the dense suffix explicit and
    # deterministic without claiming it reproduces the original context output.
    qv = data["q"].permute(1, 0, 2).unsqueeze(0).contiguous()
    kv = data["k"].permute(1, 0, 2).unsqueeze(0).contiguous()
    vv = data["v"].permute(1, 0, 2).unsqueeze(0).contiguous()
    ck = data["context_k"].permute(1, 0, 2).unsqueeze(0).contiguous()
    cv = data["context_v"].permute(1, 0, 2).unsqueeze(0).contiguous()
    cq = torch.zeros_like(ck)
    return (
        torch.cat((qv, cq), dim=1).cuda(),
        torch.cat((kv, ck), dim=1).cuda(),
        torch.cat((vv, cv), dim=1).cuda(),
        qv.shape[1], tuple(data["grid"]),
        {"case": data["case"], "evaluation": data["evaluation"], "layer": data["layer"],
         "heads": list(data["heads"]), "context_q": "zero fixture; sparse video prefix only"},
    )


def load_10s():
    data = torch.load(CAPTURE_10S, map_location="cpu", weights_only=True)
    index = list(HEADS_10S)
    return (
        data["query"][:, :, index].contiguous().cuda(),
        data["key"][:, :, index].contiguous().cuda(),
        data["value"][:, :, index].contiguous().cuda(),
        int(data["video_tokens"]), tuple(data["grid"]),
        {"case": data["case"], "evaluation": data["evaluation"], "layer": data["layer"],
         "heads": index, "context_q": "captured"},
    )


def saved_10s_permutations(device):
    result = []
    for mode in ("current_legacydirs_heads13-51-54", "current_midpoint_fused_heads13-51-54"):
        d = STAGES_10S / mode
        qp = torch.load(d / "repeat0_query_permutation.pt", map_location=device, weights_only=True)
        kp = torch.load(d / "repeat0_key_permutation.pt", map_location=device, weights_only=True)
        qi = torch.argsort(qp, dim=-1)
        ki = torch.argsort(kp, dim=-1)
        result.append((qp, qi, kp, ki))
    return tuple(result)


def synthetic_route_cases() -> dict[str, Any]:
    from h3_sparse_attention.sol_numerator_virtual_q import reduce_virtual_key_centroids

    generator = torch.Generator(device="cuda").manual_seed(20261004)
    result = {}
    for name, zero in (("normal_tail", False), ("boundary_tie_tail", True)):
        video_tokens, context_tokens, heads = 1025, 67, 2
        total = video_tokens + context_tokens
        q = torch.randn(1, total, heads, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q, generator=generator)
        if zero:
            q.zero_()
            k.zero_()
        metrics, _, _ = route_metrics(q.contiguous(), k.contiguous(), video_tokens=video_tokens)
        result[name] = metrics
    return result


def compare_saved_10s_stages() -> dict[str, Any]:
    a = STAGES_10S / "current_legacydirs_heads13-51-54"
    b = STAGES_10S / "current_midpoint_fused_heads13-51-54"
    result = {}
    for name in (
        "query_permutation", "key_permutation", "global_anchor", "threshold",
        "weighted_key", "weighted_value", "logmass", "sparse_output", "final_output",
    ):
        x = torch.load(a / f"repeat0_{name}.pt", map_location="cpu", weights_only=True)
        y = torch.load(b / f"repeat0_{name}.pt", map_location="cpu", weights_only=True)
        result[name] = tensor_metrics(x, y)
    trace = json.loads((STAGES_10S / "current_legacydirs_trace_heads13-51-54/manifest.json").read_text())
    result["direction_trace_legacy_vs_fused"] = trace["direction_stats"]
    return result


def environment() -> dict[str, Any]:
    import triton
    files = [
        Path("h3_sparse_attention/landmark_tree_v2.py"),
        Path("h3_sparse_attention/landmark_v2_fused_node.py"),
        Path("h3_sparse_attention/spark_integration.py"),
        Path("h3_sparse_attention/sol_topk_cutoff.py"),
        Path("h3_sparse_attention/sol_numerator_virtual_q.py"),
        Path("h3_sparse_attention/spark_reweight_sm120.py"),
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "H3_SM120_RUNTIME_TOPK_RATIO",
            "H3_LMV2_FUSED_NODE", "H3_LMV2_FUSED_NODE_MAX_TOKENS",
        )},
        "source_sha256": {str(p): sha256(p) for p in files},
        "capture_sha256": {str(CAPTURE_5S): sha256(CAPTURE_5S), str(CAPTURE_10S): sha256(CAPTURE_10S)},
    }


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    ROOT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output: dict[str, Any] = {"status": "running", "environment": environment()}
    (ROOT / "results.partial.json").write_text(json.dumps(output, indent=2) + "\n")

    output["synthetic_routes"] = synthetic_route_cases()
    output["saved_10s_stages"] = compare_saved_10s_stages()

    for label, loader in (("5s_case01", load_5s), ("10s_case05", load_10s)):
        q, k, v, video_tokens, grid, metadata = loader()
        layout = layout_for(q, video_tokens, grid)
        if label.startswith("10s"):
            legacy, fused = saved_10s_permutations(q.device)
            # Rebuild once only to recover the topology used by fixed replay.
            _, hierarchy = build_permutations(q, k, layout, "legacy")
        else:
            legacy, hierarchy = build_permutations(q, k, layout, "legacy")
            fused, _ = build_permutations(q, k, layout, "fused")
        ql, kl, vl = permute_qkv(q, k, v, legacy, video_tokens)
        qf, kf, vf = permute_qkv(q, k, v, fused, video_tokens)
        route_legacy, _, _ = route_metrics(ql, kl, video_tokens=video_tokens)
        route_fused, _, _ = route_metrics(qf, kf, video_tokens=video_tokens)
        output[label] = {
            "metadata": metadata,
            "permutation": perm_metrics(legacy, fused),
            "legacy_fixed_permutation_routes": route_legacy,
            "fused_fixed_permutation_routes": route_fused,
            "legacy_fixed_permutation_outputs": attention_outputs(q, k, v, layout, legacy, hierarchy),
            "fused_fixed_permutation_outputs": attention_outputs(q, k, v, layout, fused, hierarchy),
        }
        del q, k, v, ql, kl, vl, qf, kf, vf
        torch.cuda.empty_cache()
        (ROOT / "results.partial.json").write_text(json.dumps(output, indent=2) + "\n")

    output["status"] = "complete"
    (ROOT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "results.partial.json").unlink()
    print(json.dumps({"status": output["status"], "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
