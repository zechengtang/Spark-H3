"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import triton


import triton.language as tl


@triton.jit
def _average_route_kernel(
    q_ptr,
    kc_ptr,
    vc_ptr,
    threshold_ptr,
    route_ptr,
    average_output_ptr,
    average_lse_ptr,
    scale_log2,
    tokens,
    sink_start_block,
    sink_end_block,
    has_sink: tl.constexpr,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    value_tile: tl.constexpr,
    block_size: tl.constexpr,
    route_group_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    value_tile_id, query_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // heads, batch_head % heads
    query_tokens = query_block * block_size + tl.arange(0, block_size)
    query_valid = query_tokens < tokens
    dims = tl.arange(0, head_dim)
    value_dims = value_tile_id * value_tile + tl.arange(0, value_tile)
    query_offsets = (
        ((batch * tokens + query_tokens[:, None]).to(tl.int64) * heads + head)
        * head_dim
        + dims[None, :]
    )
    query = tl.load(q_ptr + query_offsets, mask=query_valid[:, None], other=0.0)
    query_length = tl.minimum(block_size, tokens - query_block * block_size)
    route_threshold = tl.load(
        threshold_ptr + (batch * blocks + query_block) * heads + head
    )

    output = tl.zeros((block_size, value_tile), dtype=tl.float32)
    row_sum = tl.zeros((block_size,), dtype=tl.float32)
    row_max = tl.full((block_size,), -float("inf"), tl.float32)
    group_offsets = tl.max_contiguous(
        tl.arange(0, route_group_size), route_group_size
    )

    for group_start in range(0, blocks, route_group_size):
        block_indices = group_start + group_offsets
        valid_blocks = block_indices < blocks
        key_offsets = (
            ((batch * blocks + block_indices[:, None]).to(tl.int64) * heads + head)
            * head_dim
            + dims[None, :]
        )
        key_centroids = tl.load(
            kc_ptr + key_offsets, mask=valid_blocks[:, None], other=0.0
        )
        scores = tl.dot(query, key_centroids.T).to(tl.float32) * scale_log2
        column_mean = tl.sum(scores, axis=0) / query_length.to(tl.float32)
        exact = (column_mean > route_threshold) | (
            tl.abs(query_block - block_indices) <= 1
        )
        if has_sink:
            exact = exact | (
                (block_indices >= sink_start_block)
                & (block_indices < sink_end_block)
            )
        exact = exact & valid_blocks

        if value_tile_id == 0:
            route_offsets = (
                ((batch * blocks + query_block) * heads + head) * blocks
                + block_indices
            )
            tl.store(
                route_ptr + route_offsets,
                exact.to(tl.uint8),
                mask=valid_blocks,
            )

        approximate = valid_blocks & ~exact
        has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
        candidate_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
        new_max = tl.where(has_approximate, candidate_max, row_max)
        alpha = tl.math.exp2(
            tl.where(has_approximate, row_max - new_max, 0.0)
        )
        probabilities = tl.math.exp2(
            safe_scores - tl.where(has_approximate, new_max, 0.0)[:, None]
        )
        probabilities = tl.where(
            has_approximate & approximate[None, :], probabilities, 0.0
        )
        value_offsets = (
            ((batch * blocks + block_indices[:, None]).to(tl.int64) * heads + head)
            * head_dim
            + value_dims[None, :]
        )
        value_sums = tl.load(
            vc_ptr + value_offsets,
            mask=valid_blocks[:, None] & (value_dims[None, :] < head_dim),
            other=0.0,
        )
        output = output * alpha[:, None] + tl.dot(
            probabilities.to(value_sums.dtype), value_sums
        )
        block_lengths = tl.minimum(
            block_size, tl.maximum(0, tokens - block_indices * block_size)
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            probabilities * block_lengths[None, :], axis=1
        )
        row_max = new_max

    valid_sum = row_sum > 0.0
    normalized = tl.where(valid_sum[:, None], output / row_sum[:, None], 0.0)
    output_offsets = (
        ((batch * tokens + query_tokens[:, None]).to(tl.int64) * heads + head)
        * head_dim
        + value_dims[None, :]
    )
    tl.store(
        average_output_ptr + output_offsets,
        normalized,
        mask=query_valid[:, None] & (value_dims[None, :] < head_dim),
    )
    if value_tile_id == 0:
        lse = tl.where(
            valid_sum,
            (row_max + tl.math.log2(tl.where(valid_sum, row_sum, 1.0)))
            * 0.6931471805599453,
            -float("inf"),
        )
        stats_offsets = (batch * tokens + query_tokens) * heads + head
        tl.store(average_lse_ptr + stats_offsets, lse, mask=query_valid)


@triton.jit
def _exact_gap_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kc_ptr,
    route_ptr,
    exact_output_ptr,
    exact_lse_ptr,
    mean_gap_ptr,
    scale_log2,
    tokens,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    value_tile: tl.constexpr,
    block_size: tl.constexpr,
    route_group_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    value_tile_id, query_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // heads, batch_head % heads
    query_tokens = query_block * block_size + tl.arange(0, block_size)
    query_valid = query_tokens < tokens
    token_offsets = tl.arange(0, block_size)
    dims = tl.arange(0, head_dim)
    value_dims = value_tile_id * value_tile + tl.arange(0, value_tile)
    query_offsets = (
        ((batch * tokens + query_tokens[:, None]).to(tl.int64) * heads + head)
        * head_dim
        + dims[None, :]
    )
    query = tl.load(q_ptr + query_offsets, mask=query_valid[:, None], other=0.0)

    output = tl.zeros((block_size, value_tile), dtype=tl.float32)
    row_sum = tl.zeros((block_size,), dtype=tl.float32)
    row_max = tl.full((block_size,), -float("inf"), tl.float32)
    gap_sum = tl.zeros((block_size,), dtype=tl.float32)
    gap_count = tl.zeros((), dtype=tl.int32)
    group_offsets = tl.max_contiguous(
        tl.arange(0, route_group_size), route_group_size
    )

    for group_start in range(0, blocks, route_group_size):
        block_indices = group_start + group_offsets
        valid_blocks = block_indices < blocks
        route_offsets = (
            ((batch * blocks + query_block) * heads + head) * blocks
            + block_indices
        )
        exact = tl.load(route_ptr + route_offsets, mask=valid_blocks, other=0) != 0
        exact = exact & valid_blocks
        exact_offsets = tl.where(exact, group_offsets, route_group_size)
        exact_count = tl.sum(exact.to(tl.int32), axis=0)
        gap_count += exact_count

        for _ in range(exact_count):
            offset = tl.min(exact_offsets)
            key_block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, route_group_size, exact_offsets
            )
            key_start = key_block * block_size
            key_tokens = key_start + token_offsets
            key_valid = key_tokens < tokens
            key_offsets = (
                ((batch * tokens + key_tokens[:, None]).to(tl.int64) * heads + head)
                * head_dim
                + dims[None, :]
            )
            keys = tl.load(k_ptr + key_offsets, mask=key_valid[:, None], other=0.0)
            exact_scores = tl.dot(query, keys.T).to(tl.float32) * scale_log2
            exact_scores = tl.where(key_valid[None, :], exact_scores, -float("inf"))

            block_max = tl.max(exact_scores, axis=1)
            block_probabilities = tl.math.exp2(exact_scores - block_max[:, None])
            block_sum = tl.sum(block_probabilities, axis=1)
            centroid_offsets = (
                ((batch * blocks + key_block) * heads + head) * head_dim + dims
            )
            key_centroid = tl.load(kc_ptr + centroid_offsets)
            mean_score = tl.sum(query * key_centroid[None, :], axis=1) * scale_log2
            block_length = tl.minimum(block_size, tokens - key_start).to(tl.float32)
            block_lse = block_max + tl.math.log2(block_sum)
            gap_sum += (
                block_lse - mean_score - tl.math.log2(block_length)
            ) * 0.6931471805599453

            new_max = tl.maximum(row_max, block_max)
            alpha = tl.math.exp2(row_max - new_max)
            probabilities = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            value_offsets = (
                ((batch * tokens + key_tokens[:, None]).to(tl.int64) * heads + head)
                * head_dim
                + value_dims[None, :]
            )
            values = tl.load(
                v_ptr + value_offsets,
                mask=key_valid[:, None] & (value_dims[None, :] < head_dim),
                other=0.0,
            )
            output = output * alpha[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            row_max = new_max

    valid_sum = row_sum > 0.0
    normalized = tl.where(valid_sum[:, None], output / row_sum[:, None], 0.0)
    output_offsets = (
        ((batch * tokens + query_tokens[:, None]).to(tl.int64) * heads + head)
        * head_dim
        + value_dims[None, :]
    )
    tl.store(
        exact_output_ptr + output_offsets,
        normalized,
        mask=query_valid[:, None] & (value_dims[None, :] < head_dim),
    )
    if value_tile_id == 0:
        lse = tl.where(
            valid_sum,
            (row_max + tl.math.log2(tl.where(valid_sum, row_sum, 1.0)))
            * 0.6931471805599453,
            -float("inf"),
        )
        mean_gap = tl.where(gap_count > 0, gap_sum / gap_count.to(tl.float32), 0.0)
        stats_offsets = (batch * tokens + query_tokens) * heads + head
        tl.store(exact_lse_ptr + stats_offsets, lse, mask=query_valid)
        tl.store(mean_gap_ptr + stats_offsets, mean_gap, mask=query_valid)

