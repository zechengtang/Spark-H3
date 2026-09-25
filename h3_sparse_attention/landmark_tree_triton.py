"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math


import torch


_PROXY_CAPACITY_CACHE: dict[
    tuple[torch.device, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]
] = {}


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only installs
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _tile_l2_argmin_kernel(
        samples,
        sample_indices,
        centers,
        center_norm_sq,
        labels,
        min_cost,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        landmarks: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STORE_COST: tl.constexpr,
        INDIRECT: tl.constexpr,
    ):
        batch = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < tokens
        if INDIRECT:
            source_row = tl.load(
                sample_indices + batch * tokens + rows, mask=row_mask, other=0
            ).to(tl.int64)
            sample_offset = source_row[:, None] * dim
        else:
            sample_offset = batch * tokens * dim + rows[:, None] * dim
        x = tl.load(
            samples + sample_offset + dims[None, :],
            mask=row_mask[:, None] & (dims[None, :] < dim),
            other=0.0,
        )
        best_cost = tl.full((BLOCK_M,), float("inf"), tl.float32)
        best_label = tl.zeros((BLOCK_M,), tl.int32)
        for start_n in tl.static_range(0, landmarks, BLOCK_N):
            columns = start_n + tl.arange(0, BLOCK_N)
            column_mask = columns < landmarks
            center = tl.load(
                centers
                + batch * landmarks * dim
                + columns[:, None] * dim
                + dims[None, :],
                mask=column_mask[:, None] & (dims[None, :] < dim),
                other=0.0,
            )
            dot = tl.dot(x, tl.trans(center), out_dtype=tl.float32)
            norm = tl.load(
                center_norm_sq + batch * landmarks + columns,
                mask=column_mask,
                other=float("inf"),
            ).to(tl.float32)
            cost = norm[None, :] - 2.0 * dot
            cost = tl.where(column_mask[None, :], cost, float("inf"))
            local_slot = tl.argmin(cost, axis=1, tie_break_left=True)
            local_cost = tl.min(cost, axis=1)
            local_label = start_n + local_slot
            replace = (local_cost < best_cost) | (
                (local_cost == best_cost) & (local_label < best_label)
            )
            best_cost = tl.where(replace, local_cost, best_cost)
            best_label = tl.where(replace, local_label, best_label)
        tl.store(labels + batch * tokens + rows, best_label, mask=row_mask)
        if STORE_COST:
            tl.store(min_cost + batch * tokens + rows, best_cost, mask=row_mask)


    @triton.jit
    def _squared_norm_kernel(
        samples,
        output,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        mask = (rows[:, None] < tokens) & (dims[None, :] < dim)
        value = tl.load(
            samples + batch * tokens * dim + rows[:, None] * dim + dims[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        norm = tl.sum(value * value, axis=1)
        tl.store(output + batch * tokens + rows, norm, mask=rows < tokens)


    @triton.jit
    def _deterministic_segmented_mean_kernel(
        samples,
        sample_indices,
        labels,
        old_centers,
        new_centers,
        counts,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        landmarks: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INPUT_BF16: tl.constexpr,
        INDIRECT: tl.constexpr,
    ):
        center_block = tl.program_id(0)
        dim_block = tl.program_id(1)
        batch = tl.program_id(2)
        center = center_block * BLOCK_K + tl.arange(0, BLOCK_K)
        dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        total = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
        count = tl.zeros((BLOCK_K,), tl.int32)
        for start_m in tl.range(0, tokens, BLOCK_M, loop_unroll_factor=1):
            rows = start_m + tl.arange(0, BLOCK_M)
            row_mask = rows < tokens
            label = tl.load(
                labels + batch * tokens + rows, mask=row_mask, other=-1
            )
            member = (
                (center[:, None] < landmarks)
                & row_mask[None, :]
                & (label[None, :] == center[:, None])
            )
            if INDIRECT:
                source_row = tl.load(
                    sample_indices + batch * tokens + rows,
                    mask=row_mask,
                    other=0,
                ).to(tl.int64)
                sample_offset = source_row[:, None] * dim
            else:
                sample_offset = batch * tokens * dim + rows[:, None] * dim
            value = tl.load(
                samples + sample_offset + dims[None, :],
                mask=row_mask[:, None] & (dims[None, :] < dim),
                other=0.0,
            )
            indicator = member.to(tl.bfloat16 if INPUT_BF16 else tl.float16)
            value = value.to(tl.bfloat16 if INPUT_BF16 else tl.float16)
            total = tl.dot(indicator, value, acc=total, out_dtype=tl.float32)
            count += tl.sum(member.to(tl.int32), axis=1)
        old = tl.load(
            old_centers
            + batch * landmarks * dim
            + center[:, None] * dim
            + dims[None, :],
            mask=(center[:, None] < landmarks) & (dims[None, :] < dim),
            other=0.0,
        ).to(tl.float32)
        mean = total / tl.maximum(count[:, None], 1).to(tl.float32)
        mean = tl.where(count[:, None] > 0, mean, old)
        tl.store(
            new_centers
            + batch * landmarks * dim
            + center[:, None] * dim
            + dims[None, :],
            mean,
            mask=(center[:, None] < landmarks) & (dims[None, :] < dim),
        )
        tl.store(
            counts + batch * landmarks + center,
            count,
            mask=(dim_block == 0) & (center < landmarks),
        )


    @triton.jit
    def _tile_tree_score_kernel(
        samples,
        sample_indices,
        centers,
        alpha,
        bias,
        output,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        landmarks: tl.constexpr,
        internal_nodes: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
        INDIRECT: tl.constexpr,
    ):
        batch = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        score_columns = tl.arange(0, BLOCK_S)
        row_mask = rows < tokens
        if INDIRECT:
            source_row = tl.load(
                sample_indices + batch * tokens + rows, mask=row_mask, other=0
            ).to(tl.int64)
            sample_offset = source_row[:, None] * dim
        else:
            sample_offset = batch * tokens * dim + rows[:, None] * dim
        x = tl.load(
            samples + sample_offset + dims[None, :],
            mask=row_mask[:, None] & (dims[None, :] < dim),
            other=0.0,
        )
        score = tl.zeros((BLOCK_M, BLOCK_S), tl.float32)
        for start_n in tl.static_range(0, landmarks, BLOCK_N):
            columns = start_n + tl.arange(0, BLOCK_N)
            column_mask = columns < landmarks
            center = tl.load(
                centers
                + batch * landmarks * dim
                + columns[:, None] * dim
                + dims[None, :],
                mask=column_mask[:, None] & (dims[None, :] < dim),
                other=0.0,
            )
            wide_tile = tl.dot(x, tl.trans(center), out_dtype=tl.float32)
            coefficient = tl.load(
                alpha
                + batch * landmarks * internal_nodes
                + columns[:, None] * internal_nodes
                + score_columns[None, :],
                mask=column_mask[:, None]
                & (score_columns[None, :] < internal_nodes),
                other=0.0,
            ).to(tl.float32)
            # Alpha is intentionally retained in FP32.  IEEE input precision
            # avoids silently changing exact-capacity boundaries through TF32.
            score += tl.dot(
                wide_tile,
                coefficient,
                acc=tl.zeros((BLOCK_M, BLOCK_S), tl.float32),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        node_bias = tl.load(
            bias + batch * internal_nodes + score_columns,
            mask=score_columns < internal_nodes,
            other=0.0,
        ).to(tl.float32)
        score = 2.0 * score + node_bias[None, :]
        tl.store(
            output
            + batch * tokens * internal_nodes
            + rows[:, None] * internal_nodes
            + score_columns[None, :],
            score,
            mask=row_mask[:, None]
            & (score_columns[None, :] < internal_nodes),
        )


    @triton.jit
    def _landmark_pair_distance_kernel(
        centers,
        output,
        landmarks: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Form the small Kc×Kc squared-distance matrix in one program."""

        batch = tl.program_id(0)
        landmark = tl.arange(0, BLOCK_K)
        dims = tl.arange(0, BLOCK_D)
        valid_k = landmark < landmarks
        valid_d = dims < dim
        center = tl.load(
            centers
            + batch * landmarks * dim
            + landmark[:, None] * dim
            + dims[None, :],
            mask=valid_k[:, None] & valid_d[None, :],
            other=0.0,
        )
        fp32 = center.to(tl.float32)
        norm = tl.sum(fp32 * fp32, axis=1)
        gram = tl.dot(center, tl.trans(center), out_dtype=tl.float32)
        distance = norm[:, None] + norm[None, :] - 2.0 * gram
        tl.store(
            output
            + batch * landmarks * landmarks
            + landmark[:, None] * landmarks
            + landmark[None, :],
            distance,
            mask=valid_k[:, None] & valid_k[None, :],
        )


    @triton.jit
    def _score_index_key_kernel(
        scores,
        original_indices,
        keys,
        tokens: tl.constexpr,
        internal_nodes: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        nodes = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < tokens) & (nodes[None, :] < internal_nodes)
        score = tl.load(
            scores
            + batch * tokens * internal_nodes
            + rows[:, None] * internal_nodes
            + nodes[None, :],
            mask=mask,
            other=0.0,
        )
        bits = score.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        negative = (bits & 0x80000000) != 0
        ordered = tl.where(negative, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
        signed_score = ordered - 0x80000000
        original = tl.load(
            original_indices + batch * tokens + rows,
            mask=rows < tokens,
            other=0,
        ).to(tl.int64)
        key = signed_score * 0x100000000 + original[:, None]
        tl.store(
            keys
            + batch * tokens * internal_nodes
            + rows[:, None] * internal_nodes
            + nodes[None, :],
            key,
            mask=mask,
        )


    @triton.jit
    def _active_score_index_key_kernel(
        scores,
        original_indices,
        active_node,
        keys,
        tokens: tl.constexpr,
        internal_nodes: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Encode only the score column active at the current tree depth."""

        batch = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < tokens
        node = tl.load(
            active_node + batch * tokens + rows, mask=valid, other=0
        ).to(tl.int64)
        score = tl.load(
            scores
            + batch * tokens * internal_nodes
            + rows * internal_nodes
            + node,
            mask=valid,
            other=0.0,
        )
        bits = score.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        negative = (bits & 0x80000000) != 0
        ordered = tl.where(negative, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
        signed_score = ordered - 0x80000000
        original = tl.load(
            original_indices + batch * tokens + rows, mask=valid, other=0
        ).to(tl.int64)
        key = signed_score * 0x100000000 + original
        tl.store(keys + batch * tokens + rows, key, mask=valid)


    @triton.jit
    def _route_active_depth_kernel(
        active_node,
        keys,
        thresholds,
        tokens: tl.constexpr,
        node_offset: tl.constexpr,
        nodes: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        batch = tl.program_id(1)
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < tokens
        node = tl.load(
            active_node + batch * tokens + rows, mask=valid, other=node_offset
        ).to(tl.int64)
        local_node = node - node_offset
        threshold = tl.load(
            thresholds + batch * nodes + local_node,
            mask=valid & (local_node >= 0) & (local_node < nodes),
            other=0,
        )
        key = tl.load(keys + batch * tokens + rows, mask=valid, other=0)
        next_node = 2 * node + 1 + (key > threshold).to(tl.int64)
        tl.store(active_node + batch * tokens + rows, next_node, mask=valid)


    @triton.jit
    def _label_tile_histogram_kernel(
        labels,
        counts,
        tokens: tl.constexpr,
        children: tl.constexpr,
        tiles: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        child = tl.arange(0, BLOCK_C)
        valid = rows < tokens
        label = tl.load(labels + batch * tokens + rows, mask=valid, other=-1)
        member = (
            valid[:, None]
            & (child[None, :] < children)
            & (label[:, None] == child[None, :])
        )
        count = tl.sum(member.to(tl.int32), axis=0)
        tl.store(
            counts + (batch * tiles + tile) * children + child,
            count,
            mask=child < children,
        )


    @triton.jit
    def _label_tile_prefix_kernel(
        counts,
        prefix,
        tiles: tl.constexpr,
        children: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        child = tl.program_id(0)
        batch = tl.program_id(1)
        tile = tl.arange(0, BLOCK_T)
        valid = tile < tiles
        count = tl.load(
            counts + (batch * tiles + tile) * children + child,
            mask=valid,
            other=0,
        )
        before = tl.cumsum(count, axis=0) - count
        tl.store(
            prefix + (batch * tiles + tile) * children + child,
            before,
            mask=valid,
        )


    @triton.jit
    def _binary_mask_tile_count_kernel(
        selected,
        source_order,
        counts,
        tokens: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Count selected Hilbert-ordered rows without materializing labels."""

        tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < tokens
        source = tl.load(source_order + rows, mask=valid, other=0)
        member = tl.load(
            selected + batch * tokens + source, mask=valid, other=0
        ).to(tl.int1)
        tl.store(counts + batch * tl.num_programs(0) + tile, tl.sum(member, axis=0))


    @triton.jit
    def _binary_mask_tile_prefix_kernel(
        counts,
        prefix,
        tiles: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        batch = tl.program_id(0)
        tile = tl.arange(0, BLOCK_T)
        valid = tile < tiles
        count = tl.load(counts + batch * tiles + tile, mask=valid, other=0)
        before = tl.cumsum(count, axis=0) - count
        tl.store(prefix + batch * tiles + tile, before, mask=valid)


    @triton.jit
    def _binary_mask_stable_scatter_kernel(
        selected,
        source_order,
        prefix,
        output,
        tokens: tl.constexpr,
        left_count: tl.constexpr,
        tiles: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < tokens
        source = tl.load(source_order + rows, mask=valid, other=0)
        member = tl.load(
            selected + batch * tokens + source, mask=valid, other=0
        ).to(tl.int1)
        selected_rank = tl.cumsum(member.to(tl.int64), axis=0) - 1
        active_rank = tl.cumsum((valid & ~member).to(tl.int64), axis=0) - 1
        selected_before = tl.load(prefix + batch * tiles + tile).to(tl.int64)
        active_before = tile * BLOCK_M - selected_before
        destination = tl.where(
            member,
            left_count + selected_before + selected_rank,
            active_before + active_rank,
        )
        tl.store(output + batch * tokens + destination, source, mask=valid)


    @triton.jit
    def _stable_partition_scatter_kernel(
        labels,
        source_indices,
        offsets,
        prefix,
        order,
        destinations,
        tokens: tl.constexpr,
        children: tl.constexpr,
        tiles: tl.constexpr,
        source_batch_stride,
        BLOCK_M: tl.constexpr,
        WRITE_SOURCE: tl.constexpr,
        STORE_DESTINATION: tl.constexpr,
    ):
        tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < tokens
        label = tl.load(labels + batch * tokens + rows, mask=valid, other=0)
        destination = tl.zeros((BLOCK_M,), tl.int64)
        for child in tl.static_range(0, children):
            member = valid & (label == child)
            local_rank = tl.cumsum(member.to(tl.int64), axis=0) - 1
            base = tl.load(offsets + child).to(tl.int64)
            base += tl.load(
                prefix + (batch * tiles + tile) * children + child
            ).to(tl.int64)
            destination = tl.where(member, base + local_rank, destination)
        if WRITE_SOURCE:
            output_value = tl.load(
                source_indices + batch * source_batch_stride + rows,
                mask=valid,
                other=0,
            )
        else:
            output_value = rows
        tl.store(
            order + batch * tokens + destination,
            output_value,
            mask=valid,
        )
        if STORE_DESTINATION:
            tl.store(
                destinations + batch * tokens + rows,
                destination,
                mask=valid,
            )


    @triton.jit
    def _partition_feature_scatter_kernel(
        samples,
        destinations,
        output,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        dim_block = tl.program_id(1)
        batch = tl.program_id(2)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        row_valid = rows < tokens
        dim_valid = dims < dim
        destination = tl.load(
            destinations + batch * tokens + rows, mask=row_valid, other=0
        )
        value = tl.load(
            samples
            + batch * tokens * dim
            + rows[:, None] * dim
            + dims[None, :],
            mask=row_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        tl.store(
            output
            + batch * tokens * dim
            + destination[:, None] * dim
            + dims[None, :],
            value,
            mask=row_valid[:, None] & dim_valid[None, :],
        )


    @triton.jit
    def _stable_grouped_scatter_kernel(
        labels,
        source_indices,
        source_rows,
        offsets,
        prefix,
        group_base,
        group_rank,
        capacities,
        order,
        grouped_indices,
        grouped_global_indices,
        grouped_destinations,
        tokens: tl.constexpr,
        children: tl.constexpr,
        tiles: tl.constexpr,
        batch_size: tl.constexpr,
        original_tokens: tl.constexpr,
        source_batch_stride,
        BLOCK_M: tl.constexpr,
    ):
        tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < tokens
        label = tl.load(labels + batch * tokens + rows, mask=valid, other=0)
        standard_destination = tl.zeros((BLOCK_M,), tl.int64)
        grouped_destination = tl.full((BLOCK_M,), -1, tl.int64)
        for child in tl.static_range(0, children):
            member = valid & (label == child)
            local_rank = tl.cumsum(member.to(tl.int64), axis=0) - 1
            tile_before = tl.load(
                prefix + (batch * tiles + tile) * children + child
            ).to(tl.int64)
            standard_base = tl.load(offsets + child).to(tl.int64)
            standard_base += tile_before
            standard_destination = tl.where(
                member, standard_base + local_rank, standard_destination
            )
            base = tl.load(group_base + child).to(tl.int64)
            rank = tl.load(group_rank + child).to(tl.int64)
            capacity = tl.load(capacities + child).to(tl.int64)
            destination = (
                base
                + (rank * batch_size + batch) * capacity
                + tile_before
                + local_rank
            )
            grouped_destination = tl.where(
                member & (base >= 0), destination, grouped_destination
            )
        output_value = tl.load(
            source_indices + batch * source_batch_stride + rows,
            mask=valid,
            other=0,
        )
        tl.store(
            order + batch * tokens + standard_destination,
            output_value,
            mask=valid,
        )
        grouped_valid = valid & (grouped_destination >= 0)
        tl.store(
            grouped_indices + grouped_destination,
            output_value,
            mask=grouped_valid,
        )
        source_row = tl.load(source_rows + batch).to(tl.int64)
        tl.store(
            grouped_global_indices + grouped_destination,
            output_value + source_row * original_tokens,
            mask=grouped_valid,
        )
        tl.store(
            grouped_destinations + batch * tokens + rows,
            grouped_destination,
            mask=valid,
        )


    @triton.jit
    def _grouped_feature_scatter_kernel(
        samples,
        destinations,
        output,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        dim_block = tl.program_id(1)
        batch = tl.program_id(2)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        row_valid = rows < tokens
        dim_valid = dims < dim
        destination = tl.load(
            destinations + batch * tokens + rows, mask=row_valid, other=-1
        )
        valid = row_valid & (destination >= 0)
        value = tl.load(
            samples
            + batch * tokens * dim
            + rows[:, None] * dim
            + dims[None, :],
            mask=valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        tl.store(
            output + destination[:, None] * dim + dims[None, :],
            value,
            mask=valid[:, None] & dim_valid[None, :],
        )


    @triton.jit
    def _stable_label_partition_kernel(
        labels,
        source_indices,
        offsets,
        order,
        tokens: tl.constexpr,
        children: tl.constexpr,
        source_batch_stride,
        BLOCK_M: tl.constexpr,
        WRITE_SOURCE: tl.constexpr,
    ):
        label_id = tl.program_id(0)
        batch = tl.program_id(1)
        output_offset = tl.load(offsets + label_id).to(tl.int64)
        running = tl.zeros((), tl.int64)
        for start_m in tl.range(0, tokens, BLOCK_M, loop_unroll_factor=1):
            rows = start_m + tl.arange(0, BLOCK_M)
            valid = rows < tokens
            value = tl.load(
                labels + batch * tokens + rows, mask=valid, other=-1
            )
            member = valid & (value == label_id)
            local_rank = tl.cumsum(member.to(tl.int64), axis=0) - 1
            destination = output_offset + running + local_rank
            if WRITE_SOURCE:
                output_value = tl.load(
                    source_indices + batch * source_batch_stride + rows,
                    mask=valid,
                    other=0,
                )
            else:
                output_value = rows
            tl.store(
                order + batch * tokens + destination,
                output_value,
                mask=member,
            )
            running += tl.sum(member.to(tl.int64), axis=0)


    @triton.jit
    def _weighted_proxy_node_kernel(
        centers,
        pair_distance,
        active_weight,
        left_capacity,
        right_capacity,
        alpha,
        bias,
        landmarks: tl.constexpr,
        dim: tl.constexpr,
        internal_nodes: tl.constexpr,
        node_offset: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ITERATIONS: tl.constexpr,
        WRITE_CHILDREN: tl.constexpr,
        DIRECT_OUTPUT: tl.constexpr,
    ):
        """Build one complete proxy-tree depth with one program per node/row."""

        local_node = tl.program_id(0)
        batch = tl.program_id(1)
        node = node_offset + local_node
        landmark = tl.arange(0, BLOCK_K)
        dims = tl.arange(0, BLOCK_D)
        landmark_mask = landmark < landmarks
        dim_mask = dims < dim
        weight = tl.load(
            active_weight
            + batch * internal_nodes * landmarks
            + node * landmarks
            + landmark,
            mask=landmark_mask,
            other=0,
        ).to(tl.int32)

        first_slot = landmark[:, None]
        second_slot = landmark[None, :]
        pair_valid = (
            (first_slot < landmarks)
            & (second_slot < landmarks)
            & (first_slot < second_slot)
            & (weight[:, None] > 0)
            & (weight[None, :] > 0)
        )
        distance = tl.load(
            pair_distance
            + batch * landmarks * landmarks
            + first_slot * landmarks
            + second_slot,
            mask=pair_valid,
            other=-float("inf"),
        )
        maximum_distance = tl.max(tl.max(distance, axis=1), axis=0)
        pair_index = first_slot * BLOCK_K + second_slot
        candidate = tl.where(
            distance == maximum_distance, pair_index, BLOCK_K * BLOCK_K
        )
        farthest = tl.min(tl.min(candidate, axis=1), axis=0)
        first = farthest // BLOCK_K
        second = farthest % BLOCK_K
        active_count = tl.sum((weight > 0).to(tl.int32), axis=0)
        only = tl.argmax((weight > 0).to(tl.int32), axis=0, tie_break_left=True)
        first = tl.where(active_count < 2, only, first)
        second = tl.where(active_count < 2, only, second)

        center = tl.load(
            centers
            + batch * landmarks * dim
            + landmark[:, None] * dim
            + dims[None, :],
            mask=landmark_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        left_center = tl.load(
            centers + batch * landmarks * dim + first * dim + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        right_center = tl.load(
            centers + batch * landmarks * dim + second * dim + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        left_cap = tl.load(left_capacity + node).to(tl.int32)
        right_cap = tl.load(right_capacity + node).to(tl.int32)
        sorted_landmark = landmark
        sorted_left_weight = tl.zeros((BLOCK_K,), tl.int32)
        sorted_right_weight = weight
        for _ in tl.static_range(ITERATIONS):
            direction = 2.0 * (right_center - left_center)
            node_bias = tl.sum(left_center * left_center, axis=0) - tl.sum(
                right_center * right_center, axis=0
            )
            delta = tl.sum(center * direction[None, :], axis=1) + node_bias
            delta = tl.where(landmark_mask, delta, float("inf"))
            # Pack the ordered FP32 bits and landmark id into one unique key.
            # This is the exact stable ``(delta, landmark_index)`` order and
            # avoids the former O(K^2) rank reconstruction matrices.
            delta = tl.where(delta == 0.0, 0.0, delta)
            bits = delta.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
            negative = (bits & 0x80000000) != 0
            ordered = tl.where(
                negative, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
            )
            sort_key = ordered * BLOCK_K + landmark.to(tl.int64)
            sorted_key = tl.sort(sort_key, dim=0, descending=False)
            sorted_landmark = (sorted_key % BLOCK_K).to(tl.int32)
            sorted_weight = tl.load(
                active_weight
                + batch * internal_nodes * landmarks
                + node * landmarks
                + sorted_landmark,
                mask=sorted_landmark < landmarks,
                other=0,
            ).to(tl.int32)
            prefix_before = tl.cumsum(sorted_weight, axis=0) - sorted_weight
            take = tl.maximum(left_cap - prefix_before, 0)
            take = tl.minimum(take, sorted_weight)
            sorted_left_weight = take
            sorted_right_weight = sorted_weight - take
            sorted_center = tl.load(
                centers
                + batch * landmarks * dim
                + sorted_landmark[:, None] * dim
                + dims[None, :],
                mask=(sorted_landmark[:, None] < landmarks)
                & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            left_center = tl.sum(
                sorted_center * sorted_left_weight[:, None].to(tl.float32), axis=0
            ) / left_cap.to(tl.float32)
            right_center = tl.sum(
                sorted_center * sorted_right_weight[:, None].to(tl.float32), axis=0
            ) / right_cap.to(tl.float32)

        if DIRECT_OUTPUT:
            tl.store(
                alpha + (batch * internal_nodes + node) * dim + dims,
                2.0 * (right_center - left_center),
                mask=dim_mask,
            )
        else:
            coefficient = sorted_right_weight.to(tl.float32) / right_cap.to(tl.float32)
            coefficient -= sorted_left_weight.to(tl.float32) / left_cap.to(tl.float32)
            tl.store(
                alpha
                + batch * landmarks * internal_nodes
                + sorted_landmark * internal_nodes
                + node,
                coefficient,
                mask=sorted_landmark < landmarks,
            )
        node_bias = tl.sum(left_center * left_center, axis=0) - tl.sum(
            right_center * right_center, axis=0
        )
        tl.store(bias + batch * internal_nodes + node, node_bias)
        if WRITE_CHILDREN:
            tl.store(
                active_weight
                + batch * internal_nodes * landmarks
                + (2 * node + 1) * landmarks
                + sorted_landmark,
                sorted_left_weight,
                mask=sorted_landmark < landmarks,
            )
            tl.store(
                active_weight
                + batch * internal_nodes * landmarks
                + (2 * node + 2) * landmarks
                + sorted_landmark,
                sorted_right_weight,
                mask=sorted_landmark < landmarks,
            )


    @triton.jit
    def _indexed_row_gather_kernel(
        source,
        row_indices,
        output,
        rows: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        row_mask = row < rows
        source_row = tl.load(row_indices + row, mask=row_mask, other=0)
        value = tl.load(
            source + source_row[:, None] * dim + dims[None, :],
            mask=row_mask[:, None] & (dims[None, :] < dim),
            other=0.0,
        )
        tl.store(
            output + row[:, None] * dim + dims[None, :],
            value,
            mask=row_mask[:, None] & (dims[None, :] < dim),
        )


def _validate_wide_inputs(samples: torch.Tensor, centers: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not samples.is_cuda or not centers.is_cuda:
        raise ValueError("fused landmark kernels require CUDA tensors")
    if samples.ndim != 3 or centers.ndim != 3:
        raise ValueError("samples/centers must be [batch, rows, dim]")
    if samples.shape[0] != centers.shape[0] or samples.shape[2] != centers.shape[2]:
        raise ValueError("samples and centers have incompatible shapes")
    if samples.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("samples must be FP16 or BF16")
    if centers.dtype != samples.dtype:
        raise ValueError("centers must use the sample dtype")
    if not samples.is_contiguous() or not centers.is_contiguous():
        raise ValueError("samples and centers must be contiguous")


def _validate_indexed_wide_inputs(
    source: torch.Tensor,
    row_indices: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[int, int, int]:
    if triton is None or not source.is_cuda or not centers.is_cuda:
        raise RuntimeError("indexed landmark kernels require Triton CUDA")
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("source must be contiguous [source_rows, dim]")
    if (
        row_indices.ndim != 2
        or row_indices.dtype != torch.long
        or not row_indices.is_contiguous()
    ):
        raise ValueError("row_indices must be contiguous int64 [batch, tokens]")
    batch, tokens = row_indices.shape
    dim = source.shape[1]
    if centers.shape[:1] != (batch,) or centers.shape[2:] != (dim,):
        raise ValueError("centers must match indexed batch and source dimension")
    if source.dtype not in (torch.float16, torch.bfloat16) or centers.dtype != source.dtype:
        raise ValueError("source/centers must use the same FP16/BF16 dtype")
    if not centers.is_contiguous():
        raise ValueError("centers must be contiguous")
    return batch, tokens, dim


@torch.no_grad()
def squared_norm(samples: torch.Tensor) -> torch.Tensor:
    """Fused FP32 squared norm without a full-size FP32 feature temporary."""

    if triton is None or not samples.is_cuda or samples.ndim != 3:
        raise RuntimeError("squared_norm requires a CUDA [batch, tokens, dim] tensor")
    batch, tokens, dim = samples.shape
    output = torch.empty((batch, tokens), device=samples.device, dtype=torch.float32)
    block_m = 32
    block_d = triton.next_power_of_2(dim)
    _squared_norm_kernel[(triton.cdiv(tokens, block_m), batch)](
        samples,
        output,
        tokens=tokens,
        dim=dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output


@torch.no_grad()
def tile_tree_score(
    samples: torch.Tensor,
    centers: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return ``2*(X@C.T)@Alpha+bias`` without materializing ``X@C.T``."""

    _validate_wide_inputs(samples, centers)
    batch, tokens, dim = samples.shape
    landmarks = centers.shape[1]
    if alpha.shape[:2] != (batch, landmarks) or alpha.dtype != torch.float32:
        raise ValueError("alpha must be FP32 [batch, landmarks, internal_nodes]")
    internal_nodes = alpha.shape[2]
    if bias.shape != (batch, internal_nodes) or bias.dtype != torch.float32:
        raise ValueError("bias must be FP32 [batch, internal_nodes]")
    if not alpha.is_contiguous() or not bias.is_contiguous():
        raise ValueError("alpha and bias must be contiguous")
    if not 1 <= internal_nodes <= 31:
        raise ValueError("tree score supports 1..31 internal nodes")
    output = torch.empty(
        (batch, tokens, internal_nodes), device=samples.device, dtype=torch.float32
    )
    block_m = 16
    block_d = max(16, triton.next_power_of_2(dim))
    block_n = 16
    block_s = triton.next_power_of_2(internal_nodes)
    _tile_tree_score_kernel[(triton.cdiv(tokens, block_m), batch)](
        samples,
        samples,
        centers,
        alpha,
        bias,
        output,
        tokens=tokens,
        dim=dim,
        landmarks=landmarks,
        internal_nodes=internal_nodes,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        BLOCK_S=block_s,
        INDIRECT=False,
        num_warps=4,
    )
    return output


@torch.no_grad()
def tile_tree_score_indexed(
    source: torch.Tensor,
    row_indices: torch.Tensor,
    centers: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Fused tree score reading X through source-row indices."""

    batch, tokens, dim = _validate_indexed_wide_inputs(
        source, row_indices, centers
    )
    landmarks = centers.shape[1]
    if alpha.shape[:2] != (batch, landmarks) or alpha.dtype != torch.float32:
        raise ValueError("alpha must be FP32 [batch, landmarks, internal_nodes]")
    internal_nodes = alpha.shape[2]
    if (
        bias.shape != (batch, internal_nodes)
        or bias.dtype != torch.float32
        or not alpha.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise ValueError("bias/alpha must be contiguous FP32")
    output = torch.empty(
        (batch, tokens, internal_nodes), device=source.device, dtype=torch.float32
    )
    block_m = 16
    _tile_tree_score_kernel[(triton.cdiv(tokens, block_m), batch)](
        source,
        row_indices,
        centers,
        alpha,
        bias,
        output,
        tokens=tokens,
        dim=dim,
        landmarks=landmarks,
        internal_nodes=internal_nodes,
        BLOCK_M=block_m,
        BLOCK_D=max(16, triton.next_power_of_2(dim)),
        BLOCK_N=16,
        BLOCK_S=triton.next_power_of_2(internal_nodes),
        INDIRECT=True,
        num_warps=4,
    )
    return output


@torch.no_grad()
def active_score_index_keys(
    scores: torch.Tensor,
    original_indices: torch.Tensor,
    active_node: torch.Tensor,
) -> torch.Tensor:
    """Encode only the currently active internal-node score per sample."""

    if triton is None or not scores.is_cuda:
        raise RuntimeError("active score key encoding requires Triton CUDA")
    if scores.ndim != 3 or scores.dtype != torch.float32 or not scores.is_contiguous():
        raise ValueError("scores must be contiguous FP32 [batch, tokens, nodes]")
    batch, tokens, internal_nodes = scores.shape
    expected = (batch, tokens)
    if (
        original_indices.shape != expected
        or original_indices.dtype != torch.long
        or active_node.shape != expected
        or active_node.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("indices must be int64 and active_node integer [batch, tokens]")
    keys = torch.empty(expected, device=scores.device, dtype=torch.long)
    block_m = 256
    _active_score_index_key_kernel[(triton.cdiv(tokens, block_m), batch)](
        scores,
        original_indices,
        active_node,
        keys,
        tokens=tokens,
        internal_nodes=internal_nodes,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return keys


@torch.no_grad()
def route_active_depth_(
    active_node: torch.Tensor,
    keys: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    node_offset: int,
) -> None:
    """Route one complete binary-tree depth in place."""

    if triton is None or not active_node.is_cuda:
        raise RuntimeError("depth routing requires Triton CUDA")
    if (
        active_node.ndim != 2
        or active_node.dtype not in (torch.int32, torch.int64)
        or keys.shape != active_node.shape
        or keys.dtype != torch.long
    ):
        raise ValueError("active_node must be integer and keys int64 [batch, tokens]")
    batch, tokens = active_node.shape
    if (
        thresholds.ndim != 2
        or thresholds.shape[0] != batch
        or thresholds.dtype != torch.long
        or not thresholds.is_contiguous()
    ):
        raise ValueError("thresholds must be contiguous int64 [batch, nodes]")
    nodes = thresholds.shape[1]
    block_m = 256
    _route_active_depth_kernel[(triton.cdiv(tokens, block_m), batch)](
        active_node,
        keys,
        thresholds,
        tokens=tokens,
        node_offset=node_offset,
        nodes=nodes,
        BLOCK_M=block_m,
        num_warps=4,
    )


def _validate_partition_inputs(
    labels: torch.Tensor,
    offsets: torch.Tensor,
    source_indices: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    if triton is None or not labels.is_cuda:
        raise RuntimeError("stable_label_partition requires Triton CUDA")
    if (
        labels.ndim != 2
        or labels.dtype not in (torch.int32, torch.int64)
        or not labels.is_contiguous()
    ):
        raise ValueError("labels must be contiguous int32/int64 [batch, tokens]")
    if offsets.ndim != 1 or offsets.dtype != torch.long or not offsets.is_cuda:
        raise ValueError("offsets must be a CUDA int64 vector")
    if source_indices is not None and (
        source_indices.shape != labels.shape
        or source_indices.dtype not in (torch.int32, torch.int64)
        or source_indices.stride(1) != 1
    ):
        raise ValueError("source_indices must be contiguous int32/int64 with label shape")
    batch, tokens = labels.shape
    children = offsets.numel()
    if not 2 <= children <= 64:
        raise ValueError("stable partition supports 2 to 64 labels")
    return batch, tokens, children


@torch.no_grad()
def _stable_label_partition_impl(
    labels: torch.Tensor,
    offsets: torch.Tensor,
    source_indices: torch.Tensor | None,
    *,
    samples: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, tokens, children = _validate_partition_inputs(
        labels, offsets, source_indices
    )
    if samples is not None and (
        samples.ndim != 3
        or samples.shape[:2] != labels.shape
        or samples.dtype not in (torch.float16, torch.bfloat16)
        or not samples.is_contiguous()
    ):
        raise ValueError("samples must be contiguous FP16/BF16 [batch, tokens, dim]")
    block_m = 256
    tiles = triton.cdiv(tokens, block_m)
    block_c = triton.next_power_of_2(children)
    counts = torch.empty(
        (batch, tiles, children), device=labels.device, dtype=torch.int32
    )
    prefix = torch.empty_like(counts)
    _label_tile_histogram_kernel[(tiles, batch)](
        labels,
        counts,
        tokens=tokens,
        children=children,
        tiles=tiles,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=4,
    )
    block_t = triton.next_power_of_2(tiles)
    _label_tile_prefix_kernel[(children, batch)](
        counts,
        prefix,
        tiles=tiles,
        children=children,
        BLOCK_T=block_t,
        num_warps=4,
    )
    order = torch.empty(
        (batch, tokens), device=labels.device,
        dtype=(source_indices.dtype if source_indices is not None else torch.long),
    )
    destinations = (
        torch.empty_like(labels) if samples is not None else order
    )
    _stable_partition_scatter_kernel[(tiles, batch)](
        labels,
        labels if source_indices is None else source_indices,
        offsets,
        prefix,
        order,
        destinations,
        tokens=tokens,
        children=children,
        tiles=tiles,
        source_batch_stride=(tokens if source_indices is None else source_indices.stride(0)),
        BLOCK_M=block_m,
        WRITE_SOURCE=source_indices is not None,
        STORE_DESTINATION=samples is not None,
        num_warps=4,
    )
    if samples is None:
        return order, None
    dim = samples.shape[2]
    partitioned = torch.empty_like(samples)
    feature_block_m = 8
    feature_block_d = 32
    _partition_feature_scatter_kernel[
        (triton.cdiv(tokens, feature_block_m), triton.cdiv(dim, feature_block_d), batch)
    ](
        samples,
        destinations,
        partitioned,
        tokens=tokens,
        dim=dim,
        BLOCK_M=feature_block_m,
        BLOCK_D=feature_block_d,
        num_warps=4,
    )
    return order, partitioned


@torch.no_grad()
def stable_label_partition(
    labels: torch.Tensor,
    offsets: torch.Tensor,
    source_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-pass stable counting partition for a small fixed label alphabet."""

    order, _ = _stable_label_partition_impl(
        labels, offsets, source_indices, samples=None
    )
    return order


@torch.no_grad()
def stable_binary_mask_partition(
    selected: torch.Tensor,
    source_order: torch.Tensor,
    *,
    left_count: int,
) -> torch.Tensor:
    """Stable ``unselected, selected`` partition in ``source_order`` order."""

    if triton is None or not selected.is_cuda:
        raise RuntimeError("binary stable partition requires Triton CUDA")
    if selected.ndim != 2 or selected.dtype != torch.bool or not selected.is_contiguous():
        raise ValueError("selected must be contiguous bool [batch, tokens]")
    batch, tokens = selected.shape
    if (
        source_order.shape != (tokens,)
        or source_order.dtype != torch.long
        or not source_order.is_cuda
        or not source_order.is_contiguous()
    ):
        raise ValueError("source_order must be contiguous CUDA int64 [tokens]")
    if not 0 <= left_count <= tokens:
        raise ValueError("left_count must be within the token range")
    block_m = 256
    tiles = triton.cdiv(tokens, block_m)
    counts = torch.empty((batch, tiles), device=selected.device, dtype=torch.int32)
    prefix = torch.empty_like(counts)
    _binary_mask_tile_count_kernel[(tiles, batch)](
        selected,
        source_order,
        counts,
        tokens=tokens,
        BLOCK_M=block_m,
        num_warps=4,
    )
    _binary_mask_tile_prefix_kernel[(batch,)](
        counts,
        prefix,
        tiles=tiles,
        BLOCK_T=triton.next_power_of_2(tiles),
        num_warps=4,
    )
    output = torch.empty((batch, tokens), device=selected.device, dtype=torch.long)
    _binary_mask_stable_scatter_kernel[(tiles, batch)](
        selected,
        source_order,
        prefix,
        output,
        tokens=tokens,
        left_count=left_count,
        tiles=tiles,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return output


@torch.no_grad()
def stable_label_partition_with_features(
    labels: torch.Tensor,
    offsets: torch.Tensor,
    source_indices: torch.Tensor,
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable-partition source ids and feature rows with shared destinations."""

    order, partitioned = _stable_label_partition_impl(
        labels, offsets, source_indices, samples=samples
    )
    assert partitioned is not None
    return order, partitioned


@torch.no_grad()
def build_weighted_proxy_tree(
    centers: torch.Tensor,
    weights: torch.Tensor,
    child_capacities: tuple[int, ...],
    proxy_iterations: int,
    *,
    direct_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build weighted proxies, returning coefficients or FP32 margin directions.

    The default coefficient layout remains [B,K,S]; direct output is [B,S,D].
    Both variants use the same weighted splits and raw centroid updates.
    """

    if triton is None or not centers.is_cuda:
        raise RuntimeError("the fused proxy tree requires Triton CUDA")
    if centers.ndim != 3 or centers.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("centers must be CUDA FP16/BF16 [batch, landmarks, dim]")
    batch, landmarks, dim = centers.shape
    children = len(child_capacities)
    if not 16 <= landmarks <= 256:
        raise ValueError("the fused proxy tree requires 16–256 landmarks")
    if weights.shape != (batch, landmarks) or weights.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("weights must be integer [batch, landmarks]")
    if children not in (2, 4, 8, 16, 32):
        raise ValueError("child capacities must describe 2, 4, 8, 16, or 32 leaves")
    if proxy_iterations not in (1, 2, 3, 4):
        raise ValueError("proxy_iterations must be 1..4")
    internal_nodes = children - 1
    alpha = torch.empty(
        (batch, internal_nodes, dim) if direct_output else (batch, landmarks, internal_nodes),
        device=centers.device, dtype=torch.float32
    )
    bias = torch.empty(
        (batch, internal_nodes), device=centers.device, dtype=torch.float32
    )
    pair_distance = torch.empty(
        (batch, landmarks, landmarks),
        device=centers.device,
        dtype=torch.float32,
    )
    _landmark_pair_distance_kernel[(batch,)](
        centers,
        pair_distance,
        landmarks=landmarks,
        dim=dim,
        BLOCK_K=triton.next_power_of_2(landmarks),
        BLOCK_D=max(16, triton.next_power_of_2(dim)),
        num_warps=8,
    )
    active = torch.zeros(
        (batch, internal_nodes, landmarks), device=centers.device, dtype=torch.int32
    )
    active[:, 0].copy_(weights)
    capacity_key = (centers.device, child_capacities)
    cached = _PROXY_CAPACITY_CACHE.get(capacity_key)
    if cached is None:
        left_values: list[int] = []
        right_values: list[int] = []
        ranges = [(0, children)]
        while ranges:
            next_ranges: list[tuple[int, int]] = []
            for start, end in ranges:
                middle = (start + end) // 2
                left_values.append(sum(child_capacities[start:middle]))
                right_values.append(sum(child_capacities[middle:end]))
                if middle - start > 1:
                    next_ranges.append((start, middle))
                if end - middle > 1:
                    next_ranges.append((middle, end))
            ranges = next_ranges
        cached = (
            torch.tensor(left_values, device=centers.device, dtype=torch.int32),
            torch.tensor(right_values, device=centers.device, dtype=torch.int32),
        )
        _PROXY_CAPACITY_CACHE[capacity_key] = cached
    left_capacity, right_capacity = cached
    block_d = triton.next_power_of_2(dim)
    depth = int(math.log2(children))
    for level in range(depth):
        node_offset = (1 << level) - 1
        nodes = 1 << level
        _weighted_proxy_node_kernel[(nodes, batch)](
            centers,
            pair_distance,
            active,
            left_capacity,
            right_capacity,
            alpha,
            bias,
            landmarks=landmarks,
            dim=dim,
            internal_nodes=internal_nodes,
            node_offset=node_offset,
            BLOCK_K=triton.next_power_of_2(landmarks),
            BLOCK_D=block_d,
            ITERATIONS=proxy_iterations,
            WRITE_CHILDREN=level + 1 < depth,
            DIRECT_OUTPUT=direct_output,
            num_warps=8,
        )
    return alpha, bias
