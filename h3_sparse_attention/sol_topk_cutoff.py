"""Triton cutoff preprocessors for fixed-ratio stock CuTe SOL routing."""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


BLOCK_SIZE = 64
HEAD_DIM = 128
_LOG2_E = math.log2(math.e)


@triton.jit
def _pack_topk_indices_kernel(
    indices,
    packed_route,
    rows,
    topk: tl.constexpr,
    words: tl.constexpr,
    index_tile: tl.constexpr,
):
    """Pack one exact Top-K row into 32-bit route words."""

    row = tl.program_id(0)
    lanes = tl.arange(0, index_tile)
    key_block = tl.load(
        indices + row * topk + lanes,
        mask=(row < rows) & (lanes < topk),
        other=0,
    ).to(tl.int32)
    word = key_block // 32
    bit = key_block - word * 32
    value = (1 << bit).to(tl.int32)
    tl.atomic_or(
        packed_route + row * words + word,
        value,
        mask=(row < rows) & (lanes < topk),
    )


@triton.jit
def _rms_query_stats_kernel(
    query,
    query_mean,
    query_direction,
    tokens,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    dim_tile: tl.constexpr,
):
    """Reduce one Q block once, emitting its mean and signed squared mean."""

    block_head = tl.program_id(0)
    dim_group = tl.program_id(1)
    block = block_head % blocks
    head = (block_head // blocks) % heads
    batch = block_head // (blocks * heads)
    rows = tl.arange(0, block_size)
    dims = dim_group * dim_tile + tl.arange(0, dim_tile)
    token_ids = block * block_size + rows
    offsets = (
        ((batch * tokens + token_ids[:, None]) * heads + head) * head_dim
        + dims[None, :]
    )
    valid = (token_ids[:, None] < tokens) & (dims[None, :] < head_dim)
    values = tl.load(query + offsets, mask=valid, other=0.0).to(tl.float32)
    count = tl.minimum(block_size, tokens - block * block_size).to(tl.float32)
    mean = tl.sum(values, axis=0) / count
    squared = mean * mean
    output = ((batch * heads + head) * blocks + block) * head_dim + dims
    output_valid = dims < head_dim
    tl.store(query_mean + output, mean, mask=output_valid)
    direction = ((batch * heads + head) * blocks + block) * (2 * head_dim)
    tl.store(
        query_direction + direction + dims,
        tl.where(mean > 0.0, squared, 0.0),
        mask=output_valid,
    )
    tl.store(
        query_direction + direction + head_dim + dims,
        tl.where(mean < 0.0, squared, 0.0),
        mask=output_valid,
    )


@triton.jit
def _rms_key_stats_kernel(
    key,
    key_mean,
    key_moments,
    tokens,
    heads: tl.constexpr,
    candidate_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    dim_tile: tl.constexpr,
):
    """Fused FP32 mean and one-sided second moments for complete K blocks."""

    block_head = tl.program_id(0)
    dim_group = tl.program_id(1)
    block = block_head % candidate_blocks
    head = (block_head // candidate_blocks) % heads
    batch = block_head // (candidate_blocks * heads)
    rows = tl.arange(0, block_size)
    dims = dim_group * dim_tile + tl.arange(0, dim_tile)
    token_ids = block * block_size + rows
    offsets = (
        ((batch * tokens + token_ids[:, None]) * heads + head) * head_dim
        + dims[None, :]
    )
    valid = dims[None, :] < head_dim
    values = tl.load(key + offsets, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / block_size
    residual = values - mean[None, :]
    positive = tl.maximum(residual, 0.0)
    negative = tl.maximum(-residual, 0.0)
    moment_pos = tl.sum(positive * positive, axis=0) / block_size
    moment_neg = tl.sum(negative * negative, axis=0) / block_size
    output = (
        ((batch * heads + head) * candidate_blocks + block) * head_dim + dims
    )
    output_valid = dims < head_dim
    tl.store(key_mean + output, mean, mask=output_valid)
    moments = (
        ((batch * heads + head) * candidate_blocks + block) * (2 * head_dim)
    )
    tl.store(key_moments + moments + dims, moment_pos, mask=output_valid)
    tl.store(
        key_moments + moments + head_dim + dims,
        moment_neg,
        mask=output_valid,
    )


@triton.jit
def _radix_select_descending(values, ordered, eligible, rank):
    active = eligible
    remaining_rank = rank
    bin_ids = tl.arange(0, 16)
    for shift in range(28, -1, -4):
        digits = ((ordered >> shift) & 0xF).to(tl.int32)
        histogram = tl.histogram(digits, 16, mask=active)
        higher_counts = tl.sum(
            tl.where(
                bin_ids[None, :] > bin_ids[:, None],
                histogram[None, :],
                0,
            ),
            axis=1,
        )
        chosen = (remaining_rank >= higher_counts) & (
            remaining_rank < higher_counts + histogram
        )
        chosen_digit = tl.sum(tl.where(chosen, bin_ids, 0), axis=0)
        chosen_higher = tl.sum(tl.where(chosen, higher_counts, 0), axis=0)
        remaining_rank -= chosen_higher
        active &= digits == chosen_digit
    return tl.max(tl.where(active, values, -float("inf")), axis=0)


@triton.jit
def _radix_select_cutoff_kernel(
    scores,
    threshold,
    tie_flags,
    first_excluded,
    score_stride_batch,
    score_stride_query,
    score_stride_head,
    score_stride_key,
    rows: tl.constexpr,
    candidate_blocks: tl.constexpr,
    target_topk: tl.constexpr,
    score_pad: tl.constexpr,
    heads: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // (rows * heads)
    query_block = (row // heads) % rows
    head = row % heads
    lanes = tl.arange(0, score_pad)
    values = tl.load(
        scores
        + batch * score_stride_batch
        + query_block * score_stride_query
        + head * score_stride_head
        + lanes * score_stride_key,
        mask=lanes < candidate_blocks,
        other=0.0,
    )
    eligible = lanes < candidate_blocks
    eligible_count = tl.sum(eligible.to(tl.int32), axis=0)
    all_selected = target_topk >= eligible_count
    bits = values.to(tl.uint32, bitcast=True)
    ordered = tl.where(
        (bits >> 31) != 0,
        bits ^ 0xFFFFFFFF,
        bits ^ 0x80000000,
    )
    selected_min = _radix_select_descending(
        values, ordered, eligible, target_topk - 1
    )
    excluded_max = _radix_select_descending(
        values, ordered, eligible, target_topk
    )
    midpoint = excluded_max + (selected_min - excluded_max) * 0.5
    cutoff = tl.where(all_selected, -float("inf"), midpoint)
    boundary_tie = (~all_selected) & (selected_min <= excluded_max)
    tl.store(threshold + row, cutoff)
    tl.store(tie_flags + row, boundary_tie.to(tl.uint8))
    tl.store(first_excluded + row, excluded_max)


@triton.jit
def _candidate_diag_moments_kernel(
    key_centroids,
    key_mean,
    key_variance,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    candidate_blocks: tl.constexpr,
    stats_tile: tl.constexpr,
    head_dim: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch = batch_head // heads
    head = batch_head % heads
    dims = tl.arange(0, head_dim)
    lanes = tl.arange(0, stats_tile)
    total = tl.zeros((head_dim,), tl.float32)
    total_sq = tl.zeros((head_dim,), tl.float32)
    for key_start in range(0, candidate_blocks, stats_tile):
        key_blocks = key_start + lanes
        offsets = (
            ((batch * blocks + key_blocks[:, None]) * heads + head) * head_dim
            + dims[None, :]
        )
        values = tl.load(
            key_centroids + offsets,
            mask=key_blocks[:, None] < candidate_blocks,
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
    count = candidate_blocks
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    output_offsets = batch_head * head_dim + dims
    tl.store(key_mean + output_offsets, mean)
    tl.store(key_variance + output_offsets, variance)


@triton.jit
def _gaussian_cutoff_kernel(
    q,
    key_mean,
    key_variance,
    threshold,
    tokens,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    candidate_blocks: tl.constexpr,
    target_topk: tl.constexpr,
    scale_log2: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    q_rows = tl.arange(0, block_size)
    dims = tl.arange(0, head_dim)
    q_tokens = query_block * block_size + q_rows
    q_offsets = (
        ((batch * tokens + q_tokens[:, None]) * heads + head) * head_dim
        + dims[None, :]
    )
    q_values = tl.load(
        q + q_offsets,
        mask=q_tokens[:, None] < tokens,
        other=0.0,
    ).to(tl.float32)
    q_len = tl.minimum(block_size, tokens - query_block * block_size).to(
        tl.float32
    )
    q_centroid = tl.sum(q_values, axis=0) / q_len
    stat_offsets = batch_head * head_dim + dims
    mean_k = tl.load(key_mean + stat_offsets)
    var_k = tl.load(key_variance + stat_offsets)
    mean = tl.sum(q_centroid * mean_k, axis=0) * scale_log2
    variance = (
        tl.sum(q_centroid * q_centroid * var_k, axis=0)
        * scale_log2
        * scale_log2
    )

    eligible = candidate_blocks
    keep = tl.minimum(target_topk, eligible)
    keep_probability = keep.to(tl.float32) / eligible.to(tl.float32)
    cdf = tl.minimum(tl.maximum(1.0 - keep_probability, 1.0e-6), 1.0 - 1.0e-6)
    z = 1.4142135623730951 * libdevice.erfinv(2.0 * cdf - 1.0)
    cutoff = mean + z * tl.sqrt(variance + 1.0e-6)
    cutoff = tl.where(keep >= eligible, -float("inf"), cutoff)
    out_offset = (batch * blocks + query_block) * heads + head
    tl.store(threshold + out_offset, cutoff)


def _validate(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    video_tokens: int,
    topk_ratio: float,
) -> tuple[int, int, int, int, int, int, int]:
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError("Q must have shape [B, T, H, 128]")
    if not q.is_cuda or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise TypeError("Q must be a contiguous CUDA BF16 tensor")
    batch, tokens, heads, dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    if key_centroids.shape != (batch, blocks, heads, dim):
        raise ValueError("key_centroids shape does not match Q blocks")
    if (
        key_centroids.device != q.device
        or key_centroids.dtype != torch.bfloat16
        or not key_centroids.is_contiguous()
    ):
        raise TypeError("key_centroids must be contiguous CUDA BF16")
    candidate_blocks = video_tokens // BLOCK_SIZE
    query_blocks = math.ceil(video_tokens / BLOCK_SIZE)
    if candidate_blocks < 1:
        raise ValueError("at least one complete video block is required")
    target_topk = max(1, round(topk_ratio * candidate_blocks))
    return batch, tokens, heads, blocks, candidate_blocks, query_blocks, target_topk


def _gemm_score_map(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    blocks: int,
    candidate_blocks: int,
) -> torch.Tensor:
    """Build native-BF16 GEMM scores, then promote them for FP32 radix."""

    batch, tokens, heads, _ = q.shape
    padding = blocks * BLOCK_SIZE - tokens
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, padding))
    counts = torch.full(
        (blocks,), float(BLOCK_SIZE), device=q.device, dtype=torch.float32
    )
    # Scalar assignment copies a host tensor and synchronizes the CUDA stream.
    # Filling a device view keeps this entirely on the GPU.
    counts[-1:].fill_(tokens - (blocks - 1) * BLOCK_SIZE)
    query_centroids = q_padded.view(
        batch, blocks, BLOCK_SIZE, heads, HEAD_DIM
    ).sum(dim=2, dtype=torch.float32) / counts.view(1, blocks, 1, 1)
    query_centroids = query_centroids.to(torch.bfloat16)

    scores = torch.einsum(
        "bqhd,bkhd->bqhk",
        query_centroids,
        key_centroids[:, :candidate_blocks],
    ).float()
    scores.mul_(HEAD_DIM**-0.5 * _LOG2_E)
    return scores


def _gemm_score_map_prefix(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    query_blocks: int,
    candidate_blocks: int,
) -> torch.Tensor:
    """Build scores only for complete query blocks consumed by Spark."""

    batch, tokens, heads, _ = q.shape
    query_tokens = query_blocks * BLOCK_SIZE
    if not 0 < query_tokens <= tokens:
        raise ValueError("query block prefix is outside Q")
    query_centroids = q[:, :query_tokens].view(
        batch, query_blocks, BLOCK_SIZE, heads, HEAD_DIM
    ).sum(dim=2, dtype=torch.float32).mul_(1.0 / BLOCK_SIZE).to(torch.bfloat16)
    scores = torch.einsum(
        "bqhd,bkhd->bqhk",
        query_centroids,
        key_centroids[:, :candidate_blocks],
    ).float()
    scores.mul_(HEAD_DIM**-0.5 * _LOG2_E)
    return scores


@torch.no_grad()
def gemm_topk_packed_route(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    video_tokens: int,
    topk_ratio: float,
    query_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute exact Top-K once and return a 32-bit packed external route.

    Sink/context blocks are intentionally omitted: the attention mainloop
    forces its configured sink range exact independently of this mask.
    """

    (
        batch,
        tokens,
        heads,
        blocks,
        candidate_blocks,
        _,
        target_topk,
    ) = _validate(q, key_centroids, video_tokens, topk_ratio)
    if not 0 < query_tokens <= tokens or query_tokens % BLOCK_SIZE:
        raise ValueError("query_tokens must be a complete block prefix")
    query_blocks = query_tokens // BLOCK_SIZE
    scores = _gemm_score_map_prefix(
        q,
        key_centroids,
        query_blocks=query_blocks,
        candidate_blocks=candidate_blocks,
    )
    indices = scores.topk(
        min(target_topk, candidate_blocks), dim=-1, largest=True, sorted=False
    ).indices.contiguous()
    selected = indices.shape[-1]
    words = math.ceil(blocks / 32)
    packed = torch.zeros(
        (batch, query_blocks, heads, words),
        device=q.device,
        dtype=torch.int32,
    )
    rows = batch * query_blocks * heads
    _pack_topk_indices_kernel[(rows,)](
        indices,
        packed,
        rows,
        selected,
        words,
        triton.next_power_of_2(selected),
        num_warps=4,
        num_stages=1,
    )
    return packed, {
        "block_size": BLOCK_SIZE,
        "blocks": blocks,
        "candidate_video_blocks": candidate_blocks,
        "query_video_blocks": query_blocks,
        "sink_blocks": max(0, blocks - candidate_blocks),
        "route_threshold_mode": "gemm_exact_topk_packed_external",
        "route_topk_ratio": topk_ratio,
        "target_topk_blocks_per_query": target_topk,
        "packed_route_words": words,
    }


def _radix_cutoff_from_scores(
    scores: torch.Tensor,
    *,
    blocks: int,
    heads: int,
    candidate_blocks: int,
    target_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = scores.shape[0]
    threshold = torch.empty(
        (batch, blocks, heads), device=scores.device, dtype=torch.float32
    )
    tie_flags = torch.empty(
        (batch, blocks, heads), device=scores.device, dtype=torch.uint8
    )
    first_excluded = torch.empty_like(threshold)
    score_pad = triton.next_power_of_2(candidate_blocks)
    _radix_select_cutoff_kernel[(batch * blocks * heads,)](
        scores,
        threshold,
        tie_flags,
        first_excluded,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        blocks,
        candidate_blocks,
        target_topk,
        score_pad,
        heads,
        num_warps=4,
        num_stages=1,
    )
    return threshold, tie_flags, first_excluded


@torch.no_grad()
def gemm_radix_topk_cutoff(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    video_tokens: int,
    sink_tokens: int,
    topk_ratio: float,
    _return_first_excluded: bool = False,
    _key_for_rms_auxiliary: torch.Tensor | None = None,
    collect_tie_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute a BF16-GEMM score map and select its cutoff by FP32 radix."""

    (
        batch,
        tokens,
        heads,
        blocks,
        candidate_blocks,
        query_blocks,
        target_topk,
    ) = _validate(q, key_centroids, video_tokens, topk_ratio)
    scores = _gemm_score_map(
        q,
        key_centroids,
        blocks=blocks,
        candidate_blocks=candidate_blocks,
    )
    threshold, tie_flags, first_excluded = _radix_cutoff_from_scores(
        scores,
        blocks=blocks,
        heads=heads,
        candidate_blocks=candidate_blocks,
        target_topk=target_topk,
    )
    # Reporting these counts requires two device-to-host synchronizations.
    # Threshold-only callers need them only when collecting a route sample.
    sink_first = video_tokens // BLOCK_SIZE
    sink_last = math.ceil((video_tokens + sink_tokens) / BLOCK_SIZE)
    stats = {
        "block_size": BLOCK_SIZE,
        "blocks": blocks,
        "candidate_video_blocks": candidate_blocks,
        "query_video_blocks": query_blocks,
        "sink_blocks": max(0, sink_last - sink_first),
        "route_threshold_mode": "gemm_radix_topk_cutoff",
        "score_gemm_input_dtype": "bfloat16",
        "score_gemm_output_dtype": "bfloat16",
        "radix_score_dtype": "float32",
        "route_topk_ratio": topk_ratio,
        "target_topk_blocks_per_query": target_topk,
    }
    if collect_tie_stats or _key_for_rms_auxiliary is not None:
        tie_mask = tie_flags.bool()
        stats["cutoff_tie_rows"] = int(tie_mask.sum().item())
        stats["cutoff_tie_video_query_rows"] = int(
            tie_mask[:, :query_blocks].sum().item()
        )
    if _return_first_excluded:
        stats["_first_excluded"] = first_excluded
    if _key_for_rms_auxiliary is not None:
        if _return_first_excluded:
            raise ValueError("RMS auxiliary output cannot include first_excluded")
        key_moments = triton_one_sided_key_moments(
            _key_for_rms_auxiliary, video_tokens=video_tokens
        )
        selected = scores > threshold[..., None]
        route = torch.zeros(
            (batch, blocks, heads, blocks), device=q.device, dtype=torch.bool
        )
        route[..., :candidate_blocks] = selected
        if sink_tokens:
            route[..., sink_first:sink_last] = True
        return threshold, stats, key_moments, route.contiguous()
    return threshold, stats


@torch.no_grad()
def triton_one_sided_key_moments(
    k: torch.Tensor, *, video_tokens: int
) -> torch.Tensor:
    """Emit packed positive/negative K moments for complete video blocks."""

    if k.ndim != 4 or k.shape[-1] != HEAD_DIM:
        raise ValueError("K must have shape [B, T, H, 128]")
    if not k.is_cuda or k.dtype != torch.bfloat16 or not k.is_contiguous():
        raise TypeError("K must be a contiguous CUDA BF16 tensor")
    batch, tokens, heads, _ = k.shape
    candidate_blocks = video_tokens // BLOCK_SIZE
    if candidate_blocks < 1 or candidate_blocks * BLOCK_SIZE > tokens:
        raise ValueError("video_tokens must contain at least one complete K block")
    key_mean = torch.empty(
        (batch, heads, candidate_blocks, HEAD_DIM),
        device=k.device,
        dtype=torch.float32,
    )
    key_moments = torch.empty(
        (batch, heads, candidate_blocks, 2 * HEAD_DIM),
        device=k.device,
        dtype=torch.float32,
    )
    dim_tile = 32
    _rms_key_stats_kernel[
        (batch * heads * candidate_blocks, triton.cdiv(HEAD_DIM, dim_tile))
    ](
        k,
        key_mean,
        key_moments,
        tokens,
        heads,
        candidate_blocks,
        BLOCK_SIZE,
        HEAD_DIM,
        dim_tile,
        num_warps=4,
        num_stages=1,
    )
    return key_moments


@torch.no_grad()
def triton_rms_k_radix_topk_cutoff(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    video_tokens: int,
    sink_tokens: int,
    topk_ratio: float,
    _return_auxiliary: bool = False,
):
    """Exact fixed-budget one-sided RMS-K cutoff without an explicit route.

    Q means and the signed squared query directions are emitted by one Triton
    reduction.  A second reduction emits K means and both one-sided moments,
    without materialising token residual tensors.  Two batched FP32 GEMMs then
    form the mean and directional-variance terms before the shared radix
    selector reduces the score map to one cutoff per query block and head.
    """

    if q.shape != k.shape:
        raise ValueError("Q/K must share shape")
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError("Q must have shape [B, T, H, 128]")
    if not q.is_cuda or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise TypeError("Q must be a contiguous CUDA BF16 tensor")
    if not k.is_cuda or k.dtype != torch.bfloat16 or not k.is_contiguous():
        raise TypeError("K must be a contiguous CUDA BF16 tensor")
    batch, tokens, heads, _ = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    candidate_blocks = video_tokens // BLOCK_SIZE
    query_blocks = math.ceil(video_tokens / BLOCK_SIZE)
    if candidate_blocks < 1:
        raise ValueError("at least one complete video block is required")
    target_topk = max(1, round(topk_ratio * candidate_blocks))

    query_mean = torch.empty(
        (batch, heads, blocks, HEAD_DIM), device=q.device, dtype=torch.float32
    )
    query_direction = torch.empty(
        (batch, heads, blocks, 2 * HEAD_DIM),
        device=q.device,
        dtype=torch.float32,
    )
    key_mean = torch.empty(
        (batch, heads, candidate_blocks, HEAD_DIM),
        device=q.device,
        dtype=torch.float32,
    )
    key_moments = torch.empty(
        (batch, heads, candidate_blocks, 2 * HEAD_DIM),
        device=q.device,
        dtype=torch.float32,
    )
    dim_tile = 32
    _rms_query_stats_kernel[
        (batch * heads * blocks, triton.cdiv(HEAD_DIM, dim_tile))
    ](
        q,
        query_mean,
        query_direction,
        tokens,
        heads,
        blocks,
        BLOCK_SIZE,
        HEAD_DIM,
        dim_tile,
        num_warps=4,
        num_stages=1,
    )
    _rms_key_stats_kernel[
        (batch * heads * candidate_blocks, triton.cdiv(HEAD_DIM, dim_tile))
    ](
        k,
        key_mean,
        key_moments,
        tokens,
        heads,
        candidate_blocks,
        BLOCK_SIZE,
        HEAD_DIM,
        dim_tile,
        num_warps=4,
        num_stages=1,
    )

    flat_batch = batch * heads
    mean_scores = torch.bmm(
        query_mean.view(flat_batch, blocks, HEAD_DIM),
        key_mean.view(flat_batch, candidate_blocks, HEAD_DIM).transpose(1, 2),
    )
    variance_scores = torch.bmm(
        query_direction.view(flat_batch, blocks, 2 * HEAD_DIM),
        key_moments.view(flat_batch, candidate_blocks, 2 * HEAD_DIM).transpose(1, 2),
    )
    variance_scores.clamp_min_(0.0).sqrt_()
    mean_scores.add_(variance_scores).mul_(HEAD_DIM**-0.5 * _LOG2_E)
    scores = mean_scores.view(batch, heads, blocks, candidate_blocks).permute(
        0, 2, 1, 3
    )
    threshold, tie_flags, _ = _radix_cutoff_from_scores(
        scores,
        blocks=blocks,
        heads=heads,
        candidate_blocks=candidate_blocks,
        target_topk=target_topk,
    )
    tie_mask = tie_flags.bool()
    sink_first = video_tokens // BLOCK_SIZE
    sink_last = math.ceil((video_tokens + sink_tokens) / BLOCK_SIZE)
    stats = {
        "block_size": BLOCK_SIZE,
        "blocks": blocks,
        "candidate_video_blocks": candidate_blocks,
        "query_video_blocks": query_blocks,
        "sink_blocks": max(0, sink_last - sink_first),
        "route_threshold_mode": "triton_rms_k_fp32_gemm_radix_cutoff",
        "score_gemm_input_dtype": "float32",
        "radix_score_dtype": "float32",
        "route_topk_ratio": topk_ratio,
        "target_topk_blocks_per_query": target_topk,
        "cutoff_tie_rows": int(tie_mask.sum().item()),
        "cutoff_tie_video_query_rows": int(
            tie_mask[:, :query_blocks].sum().item()
        ),
        "rms_lambdas": [1.0, 0.0, 0.0],
    }
    if not _return_auxiliary:
        return threshold, stats

    # The corrected approximate branch currently uses the portable Triton
    # explicit-route mainloop. Materialise the route only for that opt-in path;
    # ordinary RMS routing consumes the compact cutoff directly.
    selected = scores > threshold[..., None]
    route = torch.zeros(
        (batch, blocks, heads, blocks), device=q.device, dtype=torch.bool
    )
    route[..., :candidate_blocks] = selected
    if sink_tokens:
        route[..., sink_first:sink_last] = True
    return threshold, stats, key_moments, route.contiguous()


@torch.no_grad()
def triton_gaussian_moment_cutoff(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    video_tokens: int,
    sink_tokens: int,
    topk_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Infer a fixed-budget cutoff from SOL-style diagonal key moments."""

    (
        batch,
        tokens,
        heads,
        blocks,
        candidate_blocks,
        query_blocks,
        target_topk,
    ) = _validate(q, key_centroids, video_tokens, topk_ratio)
    key_mean = torch.empty(
        (batch, heads, HEAD_DIM), device=q.device, dtype=torch.float32
    )
    key_variance = torch.empty_like(key_mean)
    threshold = torch.empty(
        (batch, blocks, heads), device=q.device, dtype=torch.float32
    )
    _candidate_diag_moments_kernel[(batch * heads,)](
        key_centroids,
        key_mean,
        key_variance,
        heads,
        blocks,
        candidate_blocks,
        64,
        HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )
    _gaussian_cutoff_kernel[(blocks, batch * heads)](
        q,
        key_mean,
        key_variance,
        threshold,
        tokens,
        heads,
        blocks,
        candidate_blocks,
        target_topk,
        HEAD_DIM**-0.5 * _LOG2_E,
        BLOCK_SIZE,
        HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    sink_first = video_tokens // BLOCK_SIZE
    sink_last = math.ceil((video_tokens + sink_tokens) / BLOCK_SIZE)
    stats = {
        "block_size": BLOCK_SIZE,
        "blocks": blocks,
        "candidate_video_blocks": candidate_blocks,
        "query_video_blocks": query_blocks,
        "sink_blocks": max(0, sink_last - sink_first),
        "route_threshold_mode": "triton_gaussian_moment_cutoff",
        "route_topk_ratio": topk_ratio,
        "target_topk_blocks_per_query": target_topk,
    }
    return threshold, stats


__all__ = [
    "gemm_radix_topk_cutoff",
    "triton_one_sided_key_moments",
    "triton_rms_k_radix_topk_cutoff",
    "triton_gaussian_moment_cutoff",
]
