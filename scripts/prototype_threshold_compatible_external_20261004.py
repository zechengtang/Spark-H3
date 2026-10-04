#!/usr/bin/env python3
"""Prototype a threshold-compatible packed route without editing production code.

The prototype is defined entirely in this file and monkey-patched only inside
this process.  It preserves the packed_external_no_route_qk execution kernel,
but packs ``score > radix_cutoff`` instead of ``torch.topk`` indices.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable

import torch
import triton
import triton.language as tl

from diagnose_fused_external_20261004 import (
    build_permutations,
    layout_for,
    load_5s,
    load_10s,
    saved_10s_permutations,
    tensor_metrics,
)


ROOT = Path(
    os.environ.get(
        "H3_PROTO_ROOT",
        "/autodl-fs/data/h3_experiments/threshold_compatible_external_v2_20261004",
    )
)


@triton.jit
def _pack_strict_cutoff_kernel(
    scores,
    threshold,
    packed,
    score_stride_batch,
    score_stride_query,
    score_stride_head,
    score_stride_key,
    threshold_stride_batch,
    threshold_stride_query,
    threshold_stride_head,
    query_blocks: tl.constexpr,
    heads: tl.constexpr,
    candidate_blocks: tl.constexpr,
    words: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // words
    word = program - row * words
    head = row % heads
    query = (row // heads) % query_blocks
    batch = row // (heads * query_blocks)
    lane = tl.arange(0, 32)
    key_block = word * 32 + lane
    score_base = (
        batch * score_stride_batch
        + query * score_stride_query
        + head * score_stride_head
    )
    values = tl.load(
        scores + score_base + key_block * score_stride_key,
        mask=key_block < candidate_blocks,
        other=-float("inf"),
    ).to(tl.float32)
    threshold_offset = (
        batch * threshold_stride_batch
        + query * threshold_stride_query
        + head * threshold_stride_head
    )
    cutoff = tl.load(threshold + threshold_offset)
    selected = (key_block < candidate_blocks) & (values > cutoff)
    bit = (1 << lane).to(tl.int32)
    value = tl.sum(tl.where(selected, bit, 0), axis=0).to(tl.int32)
    tl.store(packed + row * words + word, value)


@torch.no_grad()
def threshold_compatible_packed_route(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    video_tokens: int,
    topk_ratio: float,
    query_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pack the exact strict-threshold mask used by the threshold mainloop."""
    from h3_sparse_attention.sol_topk_cutoff import (
        BLOCK_SIZE,
        _gemm_score_map_prefix,
        _radix_cutoff_from_scores,
        _validate,
    )

    batch, tokens, heads, blocks, candidate_blocks, _, target_topk = _validate(
        q, key_centroids, video_tokens, topk_ratio
    )
    if not 0 < query_tokens <= tokens or query_tokens % BLOCK_SIZE:
        raise ValueError("query_tokens must be a complete block prefix")
    query_blocks = query_tokens // BLOCK_SIZE
    scores = _gemm_score_map_prefix(
        q, key_centroids,
        query_blocks=query_blocks,
        candidate_blocks=candidate_blocks,
    )
    threshold, tie_flags, _ = _radix_cutoff_from_scores(
        scores,
        blocks=query_blocks,
        heads=heads,
        candidate_blocks=candidate_blocks,
        target_topk=target_topk,
    )
    words = math.ceil(blocks / 32)
    packed = torch.empty(
        (batch, query_blocks, heads, words), device=q.device, dtype=torch.int32
    )
    rows = batch * query_blocks * heads
    _pack_strict_cutoff_kernel[(rows * words,)](
        scores, threshold, packed,
        *scores.stride(),
        *threshold.stride(),
        query_blocks=query_blocks,
        heads=heads,
        candidate_blocks=candidate_blocks,
        words=words,
        num_warps=1,
        num_stages=1,
    )
    return packed, {
        "block_size": BLOCK_SIZE,
        "blocks": blocks,
        "candidate_video_blocks": candidate_blocks,
        "query_video_blocks": query_blocks,
        "sink_blocks": max(0, blocks - candidate_blocks),
        "route_threshold_mode": "prototype_strict_radix_cutoff_packed_external",
        "route_topk_ratio": topk_ratio,
        "target_topk_blocks_per_query": target_topk,
        "packed_route_words": words,
        # Keep this as a device tensor during timed calls; reporting converts it
        # only after synchronization outside the benchmark.
        "_tie_flags": tie_flags,
    }


def unpack(route: torch.Tensor, blocks: int) -> torch.Tensor:
    bits = torch.arange(32, device=route.device, dtype=torch.int32)
    return ((route[..., None] >> bits) & 1).bool().flatten(-2)[..., :blocks]


def cuda_benchmark(fn: Callable[[], Any], *, warmups: int, repeats: int) -> dict:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def fixed_attention_call(q, k, v, layout, permutation, hierarchy, execution: str):
    from h3_sparse_attention.processor import _Controller
    from h3_sparse_attention import spark_integration
    from diagnose_fused_external_20261004 import make_config

    controller = _Controller(make_config("legacy", execution))
    controller.evaluation_index = 4
    original_builder = spark_integration._landmark_tree_v2_qk_block_permutations

    def fixed(ctrl, *_args, **_kwargs):
        ctrl.landmark_reblock_hierarchy = hierarchy
        diag = {"prototype_fixed_permutation": True}
        return (*permutation, diag, diag)

    try:
        spark_integration._landmark_tree_v2_qk_block_permutations = fixed
        return spark_integration.spark_attention_bthd(
            controller, q.clone(), k.clone(), v.clone(), layout,
            layer=1, return_bthd=True,
        )
    finally:
        spark_integration._landmark_tree_v2_qk_block_permutations = original_builder


def analyze_case(label: str, loader) -> dict[str, Any]:
    from h3_sparse_attention.sol_numerator_virtual_q import reduce_virtual_key_centroids
    from h3_sparse_attention.sol_topk_cutoff import (
        gemm_radix_topk_cutoff,
        gemm_topk_packed_route,
        _gemm_score_map_prefix,
    )
    from h3_sparse_attention import sol_topk_cutoff

    q, k, v, video_tokens, grid, metadata = loader()
    layout = layout_for(q, video_tokens, grid)
    if label.startswith("10s"):
        legacy, _ = saved_10s_permutations(q.device)
        _, hierarchy = build_permutations(q, k, layout, "legacy")
    else:
        legacy, hierarchy = build_permutations(q, k, layout, "legacy")

    from diagnose_fused_external_20261004 import permute_qkv
    qp, kp, _ = permute_qkv(q, k, v, legacy, video_tokens)
    kc = reduce_virtual_key_centroids(kp)
    query_tokens = (video_tokens // 64) * 64
    query_blocks = query_tokens // 64
    candidate_blocks = video_tokens // 64
    blocks = math.ceil(q.shape[1] / 64)

    threshold, threshold_stats = gemm_radix_topk_cutoff(
        qp, kc,
        video_tokens=video_tokens,
        sink_tokens=q.shape[1] - video_tokens,
        topk_ratio=0.1,
        _return_first_excluded=True,
    )
    original_route, original_stats = gemm_topk_packed_route(
        qp, kc, video_tokens=video_tokens, topk_ratio=0.1,
        query_tokens=query_tokens,
    )
    compat_route, compat_stats = threshold_compatible_packed_route(
        qp, kc, video_tokens=video_tokens, topk_ratio=0.1,
        query_tokens=query_tokens,
    )
    scores = _gemm_score_map_prefix(
        qp, kc, query_blocks=query_blocks, candidate_blocks=candidate_blocks,
    )
    threshold_mask = scores > threshold[:, :query_blocks, :, None]
    original_mask = unpack(original_route, blocks)[..., :candidate_blocks]
    compat_mask = unpack(compat_route, blocks)[..., :candidate_blocks]

    route_bench = {
        "original_topk_packed": cuda_benchmark(
            lambda: gemm_topk_packed_route(
                qp, kc, video_tokens=video_tokens, topk_ratio=0.1,
                query_tokens=query_tokens,
            ), warmups=5, repeats=20,
        ),
        "prototype_strict_cutoff_packed": cuda_benchmark(
            lambda: threshold_compatible_packed_route(
                qp, kc, video_tokens=video_tokens, topk_ratio=0.1,
                query_tokens=query_tokens,
            ), warmups=5, repeats=20,
        ),
    }

    original_function = sol_topk_cutoff.gemm_topk_packed_route
    try:
        threshold_output = fixed_attention_call(
            q, k, v, layout, legacy, hierarchy, "threshold"
        )
        original_external_output = fixed_attention_call(
            q, k, v, layout, legacy, hierarchy,
            "packed_external_no_route_qk",
        )
        sol_topk_cutoff.gemm_topk_packed_route = threshold_compatible_packed_route
        compat_external_output = fixed_attention_call(
            q, k, v, layout, legacy, hierarchy,
            "packed_external_no_route_qk",
        )
        # Whole-path timings include permutation, virtual summaries, sparse
        # attention, dense suffix and inverse permutation; model projections and
        # denoising trajectory are intentionally outside this capture replay.
        whole_bench = {
            "threshold": cuda_benchmark(
                lambda: fixed_attention_call(
                    q, k, v, layout, legacy, hierarchy, "threshold"
                ), warmups=2, repeats=7,
            ),
            "prototype_compatible_external_no_route_qk": cuda_benchmark(
                lambda: fixed_attention_call(
                    q, k, v, layout, legacy, hierarchy,
                    "packed_external_no_route_qk",
                ), warmups=2, repeats=7,
            ),
        }
    finally:
        sol_topk_cutoff.gemm_topk_packed_route = original_function

    result = {
        "metadata": metadata,
        "shape": list(q.shape),
        "candidate_blocks": candidate_blocks,
        "target_topk": original_stats["target_topk_blocks_per_query"],
        "threshold_tie_rows": threshold_stats["cutoff_tie_video_query_rows"],
        "mask_checks": {
            "prototype_equals_threshold": bool(torch.equal(compat_mask, threshold_mask)),
            "prototype_threshold_different_bits": int((compat_mask ^ threshold_mask).sum()),
            "original_topk_threshold_different_bits": int((original_mask ^ threshold_mask).sum()),
            "prototype_count_min_max": [
                int(compat_mask.sum(-1).min()), int(compat_mask.sum(-1).max())
            ],
            "threshold_count_min_max": [
                int(threshold_mask.sum(-1).min()), int(threshold_mask.sum(-1).max())
            ],
        },
        "output_checks": {
            "prototype_external_vs_threshold": tensor_metrics(
                compat_external_output.cpu(), threshold_output.cpu()
            ),
            "original_external_vs_threshold": tensor_metrics(
                original_external_output.cpu(), threshold_output.cpu()
            ),
        },
        "route_builder_benchmark": route_bench,
        "whole_attention_replay_benchmark": whole_bench,
    }
    original_ms = route_bench["original_topk_packed"]["median_ms"]
    compat_ms = route_bench["prototype_strict_cutoff_packed"]["median_ms"]
    result["route_builder_median_delta_ms"] = compat_ms - original_ms
    threshold_ms = whole_bench["threshold"]["median_ms"]
    whole_compat_ms = whole_bench[
        "prototype_compatible_external_no_route_qk"
    ]["median_ms"]
    result["whole_replay_compatible_vs_threshold_percent"] = (
        (whole_compat_ms / threshold_ms - 1.0) * 100.0
    )
    del q, k, v, qp, kp, kc
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    ROOT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "status": "running",
        "prototype": "strict radix cutoff -> packed mask -> no-route-QK",
        "production_source_modified": False,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": {},
    }
    for label, loader in (("5s_case01", load_5s), ("10s_case05", load_10s)):
        result["cases"][label] = analyze_case(label, loader)
        (ROOT / "results.partial.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(f"complete {label}", flush=True)
    result["status"] = "complete"
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "results.partial.json").unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
