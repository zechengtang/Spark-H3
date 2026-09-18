"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math


import os


from typing import Any


import torch


import torch.nn.functional as F


_LOG2_E = math.log2(math.e)


@torch.no_grad()
def _sol_topk_policy_scores(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    video_tokens: int,
    topk_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Return stock-CuTe-scaled scores and fixed-budget policy metadata."""

    from .topk_sol_kernel import BLOCK_SIZE

    batch, tokens, heads, dim = q.shape
    blocks = math.ceil(tokens / BLOCK_SIZE)
    if key_centroids.shape != (batch, blocks, heads, dim):
        raise ValueError("SOL key centroid shape does not match Q")
    candidate_blocks = video_tokens // BLOCK_SIZE
    query_blocks = math.ceil(video_tokens / BLOCK_SIZE)
    if candidate_blocks < 1:
        raise ValueError("SOL Top-K routing requires a complete video block")

    padding = blocks * BLOCK_SIZE - tokens
    q_padded = F.pad(q, (0, 0, 0, 0, 0, padding))
    counts = torch.full(
        (blocks,), float(BLOCK_SIZE), device=q.device, dtype=torch.float32
    )
    counts[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    query_centroids = q_padded.view(
        batch, blocks, BLOCK_SIZE, heads, dim
    ).sum(dim=2, dtype=torch.float32) / counts.view(1, blocks, 1, 1)
    # Stock CuTe forms the same block mean after Q.Kc and compares in exp2
    # units. Scaling is monotone (and immaterial to Top-K), but matching its
    # units lets the cutoff tensor be consumed directly by the stock selector.
    scores = torch.einsum(
        "bqhd,bkhd->bqhk", query_centroids, key_centroids.float()
    ) * (dim**-0.5 * _LOG2_E)

    adjacent = torch.zeros((blocks, blocks), device=q.device, dtype=torch.bool)
    target_topk = max(1, round(topk_ratio * candidate_blocks))
    return scores, adjacent, candidate_blocks, query_blocks, target_topk


def _sol_topk_route_from_scores(
    scores: torch.Tensor,
    adjacent: torch.Tensor,
    *,
    candidate_blocks: int,
    target_topk: int,
    video_tokens: int,
    sink_tokens: int,
    score_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialise the legacy explicit route from a shared score table."""

    from .topk_sol_kernel import BLOCK_SIZE

    batch, blocks, heads, _ = scores.shape
    forced_candidate = adjacent[:, :candidate_blocks]
    eligible = scores[..., :candidate_blocks].masked_fill(
        forced_candidate[None, :, None, :], -torch.inf
    )
    selection_count = min(target_topk, candidate_blocks)
    if score_indices is None:
        score_indices = eligible.topk(selection_count, dim=-1).indices
    else:
        score_indices = score_indices[..., :selection_count]
    score_selected = torch.zeros_like(eligible, dtype=torch.bool)
    selected_is_additional = ~forced_candidate[None, :, None, :].expand(
        batch, -1, heads, -1
    ).gather(-1, score_indices)
    score_selected.scatter_(-1, score_indices, selected_is_additional)

    route = torch.zeros(
        (batch, blocks, heads, blocks), device=scores.device, dtype=torch.bool
    )
    route[..., :candidate_blocks] = score_selected
    route |= adjacent[None, :, None, :]
    sink_first = video_tokens // BLOCK_SIZE
    sink_last = math.ceil((video_tokens + sink_tokens) / BLOCK_SIZE)
    if sink_tokens:
        route[..., sink_first:sink_last] = True
    return route.contiguous()


@torch.no_grad()
def _sol_topk_route(
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    *,
    video_tokens: int,
    sink_tokens: int,
    topk_ratio: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build standard post-RoPE SOL routing with an additional Top-K budget.

    All complete video key blocks compete for the Top-K quota.
    Context/sink blocks are added afterwards without consuming that quota.
    """

    from .topk_sol_kernel import BLOCK_SIZE

    scores, adjacent, candidate_blocks, query_blocks, target_topk = (
        _sol_topk_policy_scores(
            q,
            key_centroids,
            video_tokens=video_tokens,
            topk_ratio=topk_ratio,
        )
    )
    blocks = scores.shape[1]
    route = _sol_topk_route_from_scores(
        scores,
        adjacent,
        candidate_blocks=candidate_blocks,
        target_topk=target_topk,
        video_tokens=video_tokens,
        sink_tokens=sink_tokens,
    )
    sink_first = video_tokens // BLOCK_SIZE
    sink_last = math.ceil((video_tokens + sink_tokens) / BLOCK_SIZE)

    stats = {
        "block_size": BLOCK_SIZE,
        "blocks": blocks,
        "candidate_video_blocks": candidate_blocks,
        "query_video_blocks": query_blocks,
        "sink_blocks": max(0, sink_last - sink_first),
        "route_threshold_mode": "topk_post_rope_mean",
        "route_topk_ratio": topk_ratio,
        "target_topk_blocks_per_query": target_topk,
    }
    return route, stats


def _sol_topk_analytic_density(route_stats: dict[str, Any]) -> dict[str, float]:
    """Compute fixed-budget density without allocating a BxQxHxK route."""

    blocks = route_stats["blocks"]
    candidate_blocks = route_stats["candidate_video_blocks"]
    query_blocks = route_stats["query_video_blocks"]
    sink_blocks = route_stats["sink_blocks"]
    target_topk = route_stats["target_topk_blocks_per_query"]

    selected = min(target_topk, candidate_blocks)
    return {
        "additional_topk_density": selected / candidate_blocks,
        "effective_video_density": selected / candidate_blocks,
        "effective_packed_density": (selected + sink_blocks) / blocks,
    }


def _torch_headwise_permute_video_tokens(
    values: torch.Tensor,
    permutation: torch.Tensor,
    *,
    video_tokens: int,
) -> torch.Tensor:
    """Apply an independent video-token permutation to every attention head.

    ``values`` is BTHD while ``permutation`` is BHN.  Non-video context is
    appended unchanged so the existing Sol sink boundary remains valid.
    """

    if values.ndim != 4:
        raise ValueError("values must have shape [batch, tokens, heads, dim]")
    batch, tokens, heads, dim = values.shape
    if permutation.shape != (batch, heads, video_tokens):
        raise ValueError(
            "permutation must have shape "
            f"{(batch, heads, video_tokens)}, got {tuple(permutation.shape)}"
        )
    if not 0 <= video_tokens <= tokens:
        raise ValueError("video_tokens is outside the packed token dimension")
    video = values[:, :video_tokens].permute(0, 2, 1, 3)
    gathered = torch.gather(
        video,
        2,
        permutation.unsqueeze(-1).expand(-1, -1, -1, dim),
    ).permute(0, 2, 1, 3)
    if video_tokens == tokens:
        return gathered.contiguous()
    return torch.cat((gathered, values[:, video_tokens:]), dim=1).contiguous()


def _headwise_permute_video_tokens(
    values: torch.Tensor,
    permutation: torch.Tensor,
    *,
    video_tokens: int,
) -> torch.Tensor:
    """Apply the fused CUDA permutation, retaining PyTorch as the oracle."""

    if values.is_cuda and values.is_contiguous() and permutation.is_contiguous():
        from .landmark_tree_v2_triton import headwise_permute_bthd

        return headwise_permute_bthd(
            values, permutation, video_tokens=video_tokens
        )
    return _torch_headwise_permute_video_tokens(
        values, permutation, video_tokens=video_tokens
    )


@torch.no_grad()
def _landmark_tree_v2_combined_permutations(
    controller: _Controller,
    query_bthd: torch.Tensor,
    key_bthd: torch.Tensor,
    layout: PackedLayout,
):
    """Metric transform plus prepared-plan tree build on the current stream.

    Returns ``(plan, metric_indices, combined_permutation, combined_inverse)``
    where the combined tensors are the plan's ``[2*B*H, video_tokens]`` static
    outputs: keys first, queries second.
    """

    from .landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    from .landmark_direction import landmark_direction_factors
    from .mahalanobis_kmeans import (
        hilbert_midpoint_sample_indices,
    )
    from .rope_sol_kernel import BLOCK_SIZE

    if query_bthd.shape != key_bthd.shape or query_bthd.ndim != 4:
        raise ValueError("paired landmark-tree-v2 blocking requires equal BTHD Q/K")
    batch, _, heads, dim = query_bthd.shape
    video_tokens = int(layout.video_tokens)
    flat_batch = batch * heads
    clusters = video_tokens // BLOCK_SIZE
    metric_key = (
        "landmark_tree_v2_metric_indices",
        query_bthd.device.type,
        query_bthd.device.index,
        layout.grid,
        clusters,
    )
    metric_indices = controller.rope_sol_key_clustering_static.get(metric_key)
    if metric_indices is None:
        metric_indices = hilbert_midpoint_sample_indices(
            layout.grid, clusters, device=query_bthd.device
        )
        controller.rope_sol_key_clustering_static[metric_key] = metric_indices
    query = (
        query_bthd[:, :video_tokens]
        .permute(0, 2, 1, 3)
        .reshape(flat_batch, video_tokens, dim)
    )
    key = (
        key_bthd[:, :video_tokens]
        .permute(0, 2, 1, 3)
        .reshape(flat_batch, video_tokens, dim)
    )
    query_metric, key_metric = landmark_direction_factors(
        query,
        key,
        metric_indices,
        ridge=controller.config.rope_sol_key_ridge_epsilon,
        moment=controller.config.landmark_tree_v2_moment_mode,
    )
    plan_key = (
        "prepared_landmark_tree_v2_qk",
        controller.config.landmark_tree_v2_chunk_frames if controller.config.landmark_tree_v2_chunk_frames is not None else int(os.environ.get("H3_TEMPORAL_CHUNK_FRAMES", "0")),
        controller.config.landmark_tree_v2_minimum_frames if controller.config.landmark_tree_v2_minimum_frames is not None else int(os.environ.get("H3_TEMPORAL_MIN_FRAMES", "0")),
        controller.config.landmark_tree_v2_initial_order,
        controller.config.landmark_tree_v2_children,
        controller.config.landmark_tree_v2_fanout_mode,
        controller.config.landmark_tree_v2_final_fanout,
        controller.config.landmark_tree_v2_root_fanout,
        controller.config.landmark_tree_v2_landmark_mode,
        controller.config.landmark_tree_v2_landmark_count,
        controller.config.landmark_tree_v2_aggregation,
        controller.config.landmark_tree_v2_distance,
        controller.config.landmark_tree_v2_mean_mode,
        controller.config.landmark_tree_v2_moment_mode,
        controller.config.landmark_tree_v2_order_mode,
        controller.config.landmark_tree_v2_group_size,
        query.device.type,
        query.device.index,
        2 * flat_batch,
        video_tokens,
        dim,
        layout.grid,
    )
    plan = controller.rope_sol_key_clustering_static.get(plan_key)
    if plan is None:
        plan = PreparedLandmarkTreeV2Permutation(
            minimum_frames=controller.config.landmark_tree_v2_minimum_frames,
            chunk_frames=controller.config.landmark_tree_v2_chunk_frames,
            distance=controller.config.landmark_tree_v2_distance,
            max_children=controller.config.landmark_tree_v2_children,
            fanout_mode=controller.config.landmark_tree_v2_fanout_mode,
            final_fanout=controller.config.landmark_tree_v2_final_fanout,
            root_fanout=controller.config.landmark_tree_v2_root_fanout,
            landmark_mode=controller.config.landmark_tree_v2_landmark_mode,
            landmark_count=controller.config.landmark_tree_v2_landmark_count,
            aggregation=controller.config.landmark_tree_v2_aggregation,
            input_unit_means=controller.config.landmark_tree_v2_mean_mode == "input_unit",
            metric_unit_means=controller.config.landmark_tree_v2_mean_mode == "metric_unit",
            order_mode=controller.config.landmark_tree_v2_order_mode,
            group_size=controller.config.landmark_tree_v2_group_size,
            batch=2 * flat_batch,
            tokens=video_tokens,
            dim=dim,
            grid_shape=layout.grid,
            initial_order=controller.config.landmark_tree_v2_initial_order,
            device=query.device,
        )
        controller.rope_sol_key_clustering_static[plan_key] = plan
    transformed = plan.graph_input
    if transformed is None:
        transformed = torch.empty(
            (2 * flat_batch, video_tokens, dim),
            device=query.device,
            dtype=torch.bfloat16,
        )
    torch.bmm(
        key.to(torch.bfloat16),
        query_metric.to(torch.bfloat16),
        out=transformed[:flat_batch],
    )
    torch.bmm(
        query.to(torch.bfloat16),
        key_metric.to(torch.bfloat16),
        out=transformed[flat_batch:],
    )
    del query_metric, key_metric
    inverse_norms = None
    if controller.config.landmark_tree_v2_mean_mode == "input_unit":
        inverse_norms = plan.graph_inverse_norms
        if inverse_norms is None:
            inverse_norms = torch.empty((2 * flat_batch, video_tokens, 1), device=query.device, dtype=torch.float32)
        inverse_norms[:flat_batch].copy_(key.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal())
        inverse_norms[flat_batch:].copy_(query.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal())
    if plan.graph_active:
        combined_permutation, combined_inverse = plan.replay()
    else:
        combined_permutation, combined_inverse = plan.run(transformed, inverse_norms=inverse_norms)
    controller.landmark_reblock_hierarchy = plan.hierarchy
    return plan, metric_indices, combined_permutation, combined_inverse


@torch.no_grad()
def _landmark_tree_v2_qk_block_permutations(
    controller: _Controller,
    query_bthd: torch.Tensor,
    key_bthd: torch.Tensor,
    layout: PackedLayout,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    dict[str, Any],
]:
    """Paired opposite-second-moment Q/K permutations using landmark v2."""

    from .rope_sol_kernel import BLOCK_SIZE

    batch, _, heads, dim = query_bthd.shape
    video_tokens = int(layout.video_tokens)
    flat_batch = batch * heads
    plan, metric_indices, combined_permutation, combined_inverse = (
        _landmark_tree_v2_combined_permutations(controller, query_bthd, key_bthd, layout)
    )
    key_permutation = combined_permutation[:flat_batch]
    key_inverse = combined_inverse[:flat_batch]
    query_permutation = combined_permutation[flat_batch:]
    query_inverse = combined_inverse[flat_batch:]
    num_excluded = video_tokens % BLOCK_SIZE
    active_blocks = (video_tokens - num_excluded) // BLOCK_SIZE
    common = {
        "algorithm": "recursive_landmark_tree_v2_block_mean_qk_parallel_batch",
        "metric_space": "mahalanobis_transformed",
        "similarity": ("cosine" if controller.config.landmark_tree_v2_distance == "cosine" else "negative_squared_distance"),
        "initial_order": controller.config.landmark_tree_v2_initial_order,
        "distance": controller.config.landmark_tree_v2_distance,
        "max_children": controller.config.landmark_tree_v2_children,
        "fanout_mode": controller.config.landmark_tree_v2_fanout_mode,
        "aggregation": controller.config.landmark_tree_v2_aggregation,
        "mean_mode": controller.config.landmark_tree_v2_mean_mode,
        "mean_norm_space": ("transformed_token" if controller.config.landmark_tree_v2_mean_mode == "metric_unit" else "original_post_rope_token"),
        "moment_mode": controller.config.landmark_tree_v2_moment_mode,
        "metric_centered": False,
        "raw_remainder_features": True,
        "order_mode": controller.config.landmark_tree_v2_order_mode,
        "landmark_initialization": ("contiguous_mean_token_interval_mean" if controller.config.landmark_tree_v2_landmark_mode == "mean" else "contiguous_interval_midpoint_token"),
        "landmark_mode": controller.config.landmark_tree_v2_landmark_mode,
        "coarse_landmarks": controller.config.landmark_tree_v2_landmark_count,
        "coarse_assignment_passes": 0,
        "group_size": controller.config.landmark_tree_v2_group_size,
        "root_children": (plan.max_children if isinstance(plan.max_children, int) else plan.max_children[0]),
        "later_children": (plan.max_children if isinstance(plan.max_children, int) else plan.max_children[min(1, len(plan.max_children) - 1)]),
        "root_fanout": plan.root_fanout,
        "final_fanout": plan.final_fanout,
        "proxy_iterations": 2,
        "remainder_policy": "largest_transformed_l2_norm_original_index_tie",
        "num_excluded": int(num_excluded),
        "strict_cluster_blocks": int(active_blocks),
        "forced_exact_excluded_blocks": math.ceil(num_excluded / BLOCK_SIZE),
        "kernel_video_blocks": math.ceil(video_tokens / BLOCK_SIZE),
        "metric_ridge_epsilon": float(controller.config.rope_sol_key_ridge_epsilon),
    }
    key_diagnostics = {
        **common,
        "metric": ("sampled_query_second_moment" if controller.config.landmark_tree_v2_moment_mode == "raw" else "sampled_unit_query_second_moment"),
        "metric_query_samples": int(metric_indices.numel()),
        "recursive_splits": int(plan.split_count),
        "cuda_graph": bool(plan.graph_active),
    }
    query_diagnostics = {
        **common,
        "metric": ("sampled_key_second_moment" if controller.config.landmark_tree_v2_moment_mode == "raw" else "sampled_unit_key_second_moment"),
        "metric_key_samples": int(metric_indices.numel()),
        "recursive_splits": int(plan.split_count),
        "cuda_graph": bool(plan.graph_active),
    }
    controller.counts["rope_sol_landmark_tree_v2_key_assignments"] += 1
    controller.counts["rope_sol_landmark_tree_v2_query_assignments"] += 1

    def unflatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, heads, video_tokens)

    return (
        unflatten(query_permutation),
        unflatten(query_inverse),
        unflatten(key_permutation),
        unflatten(key_inverse),
        query_diagnostics,
        key_diagnostics,
    )


def spark_attention(
    controller: _Controller,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: PackedLayout,
    layer: int,
    *, return_bthd: bool = False,
) -> torch.Tensor:
    """Run official Sol-Attn with exact H3 context K/V and dense context Q."""

    try:
        from sol_attn import get_sol_attn_backend, sol_attn
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Sol-Attn is unavailable; install NVlabs/Sana's "
            "techniques/sparse_backends package"
        ) from error

    cfg = controller.config
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise RuntimeError(
            "Sol-Attn requires contiguous BF16 Q/K/V with head dimension 128; "
            f"got dtype={q.dtype}, head_dim={q.shape[-1]}"
        )

    # The common processor has already placed the target-video grid first and
    # all packed context after it.  The official kernel accepts BTHD tensors.
    q_bthd, k_bthd, v_bthd = (
        tensor.permute(0, 2, 1, 3).contiguous() for tensor in (q, k, v)
    )
    sink_start = layout.video_tokens
    sink_tokens = layout.sequence_length - layout.video_tokens
    controller.sol_backend = get_sol_attn_backend(q.device)
    query_inverse_permutation = None
    virtual_query_data = None
    if cfg.sol_landmark_preprocess:
        landmark_builder = _landmark_tree_v2_qk_block_permutations
        (
            query_permutation,
            query_inverse_permutation,
            key_permutation,
            _,
            _,
            _,
        ) = landmark_builder(
            controller,
            q_bthd,
            k_bthd,
            layout,
        )
        if cfg.sol_virtual_query_levels_up is not None or cfg.sol_virtual_query_target_blocks is not None:
            from .landmark_virtual_q import virtual_query_layout, target_virtual_query_layout
            from .sol_numerator_virtual_q import validate_virtual_layout
            from .virtual_q_permute import permute_with_virtual_anchors
            cache = controller._virtual_query_layout_cache
            virtual_spec = (
                ('target',cfg.sol_virtual_query_target_blocks,cfg.sol_virtual_query_min_blocks,
                 cfg.sol_virtual_query_max_blocks)
                if cfg.sol_virtual_query_target_blocks is not None
                else ('levels_up',cfg.sol_virtual_query_levels_up)
            )
            hierarchy = controller.landmark_reblock_hierarchy
            if hierarchy is None:
                raise RuntimeError("reblocking did not publish its hierarchy")
            cache_key = (q_bthd.shape[1], q_bthd.device, virtual_spec, hierarchy)
            if cache_key not in cache:
                if cfg.sol_virtual_query_target_blocks is not None:
                    topology = target_virtual_query_layout(
                        layout.video_tokens,q_bthd.shape[1],cfg.sol_virtual_query_target_blocks,
                        cfg.sol_virtual_query_min_blocks,cfg.sol_virtual_query_max_blocks,
                        hierarchy=hierarchy)
                else:
                    topology = virtual_query_layout(layout.video_tokens, q_bthd.shape[1],
                        cfg.sol_virtual_query_levels_up, hierarchy=hierarchy)
                ranges_host, mapping_host = validate_virtual_layout(
                    topology['ranges'], topology['leaf_to_virtual'], q_bthd.shape[1])
                cache[cache_key] = (torch.tensor(ranges_host, device=q_bthd.device, dtype=torch.int64),
                                    torch.tensor(mapping_host, device=q_bthd.device, dtype=torch.int64),
                                    dict(topology['metadata']))
            ranges, mapping, topology_metadata = cache[cache_key]
            controller.sol_virtual_query_layout = topology_metadata
            anchors = None
            if cfg.sol_virtual_query_fused_permute:
                q_bthd, anchors = permute_with_virtual_anchors(q_bthd, query_permutation,
                    ranges, video_tokens=layout.video_tokens)
                controller.counts["sol_virtual_query_fused_permute_calls"] += 1
            else:
                q_bthd = _headwise_permute_video_tokens(
                    q_bthd, query_permutation, video_tokens=layout.video_tokens)
            virtual_query_data = (ranges, mapping, anchors)
        else:
            q_bthd = _headwise_permute_video_tokens(
                q_bthd, query_permutation, video_tokens=layout.video_tokens
            )
        k_bthd = _headwise_permute_video_tokens(
            k_bthd, key_permutation, video_tokens=layout.video_tokens
        )
        v_bthd = _headwise_permute_video_tokens(
            v_bthd, key_permutation, video_tokens=layout.video_tokens
        )
        controller.counts["sol_landmark_preprocess_calls"] += 1
    if cfg.sol_route_topk_ratio is not None:
        output = _spark_topk_attention(controller, q_bthd, k_bthd, v_bthd,
                                       layout, virtual_query_data,
                                       _query_tokens=layout.video_tokens if virtual_query_data is not None and layout.video_tokens % 64 == 0 else None)
    elif virtual_query_data is not None:
        from sol_attn.preprocess import prepare
        from .sol_numerator_virtual_q import virtual_q_attention, virtual_q_backend

        kc, vs, threshold = prepare(q_bthd, k_bthd, v_bthd, tau=cfg.sol_tau,
                                    scale=q_bthd.shape[-1] ** -0.5,
                                    thresh_type=cfg.sol_thresh_type)
        ranges, mapping, anchors = virtual_query_data
        output = virtual_q_attention(
            q_bthd, k_bthd, v_bthd, virtual_ranges=ranges, leaf_to_virtual=mapping,
            virtual_anchors=anchors, key_centroids=kc, value_sums=vs,
            threshold=threshold, sink_start=sink_start, sink_tokens=sink_tokens,
            force_local_blocks=cfg.sol_local_blocks_enabled,
            _query_tokens=layout.video_tokens if layout.video_tokens % 64 == 0 else None)
        controller.sol_backend = virtual_q_backend(q_bthd)
        controller.counts["sol_virtual_query_calls"] += 1
        controller.counts["sol_tau_virtual_query_calls"] += 1
    else:
        output = sol_attn(q_bthd, k_bthd, v_bthd, tau=cfg.sol_tau,
                          thresh_type=cfg.sol_thresh_type, kv_splits=cfg.sol_kv_splits,
                          sink_start=sink_start, sink_tokens=sink_tokens,
                          force_local_blocks=cfg.sol_local_blocks_enabled)
        controller.counts["sol_official_calls"] += 1

    # Exact sinks apply to K/V blocks.  Match the released H3 integration by
    # recomputing the corresponding query rows densely.
    if sink_tokens:
        output[:, sink_start:] = F.scaled_dot_product_attention(
            q_bthd[:, sink_start:].transpose(1, 2),
            k_bthd.transpose(1, 2),
            v_bthd.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        controller.counts["sol_dense_context_queries"] += 1
    if query_inverse_permutation is not None:
        output = _headwise_permute_video_tokens(
            output,
            query_inverse_permutation,
            video_tokens=layout.video_tokens,
        )
    return output if return_bthd else output.permute(0, 2, 1, 3).contiguous()


@torch.compiler.disable
@torch.no_grad()
def _spark_topk_attention(controller, q, k, v, layout, virtual_query_data=None, _query_tokens=None):
    """Source Spark Top-K policy with native or query-conditioned summaries."""
    from sol_attn.preprocess import _reduce_kv
    from .rope_sol_kernel import (
        sol_topk_threshold_attn, sol_topk_threshold_backend,
        rope_sol_attn, rope_sol_backend,
    )
    from .sol_topk_cutoff import gemm_radix_topk_cutoff, triton_gaussian_moment_cutoff

    cfg = controller.config
    from .sol_numerator_virtual_q import virtual_q_backend, reduce_virtual_key_centroids
    if virtual_query_data is not None and virtual_q_backend(q) == "sm120_fused_virtual_query":
        kc, vs = reduce_virtual_key_centroids(k), None
    else:
        kc, vs = _reduce_kv(k, v)
    sink_tokens = layout.sequence_length - layout.video_tokens
    backend = sol_topk_threshold_backend(q.device)
    partial_video = sink_tokens == 0 and layout.video_tokens // 64 < math.ceil(q.shape[1] / 64)
    threshold = route = summaries = None
    if cfg.sol_route_global_weighted_mean:
        from .global_weighted_route import global_weighted_route
        route, stats = global_weighted_route(q, k,
            video_tokens=layout.video_tokens, sink_tokens=sink_tokens,
            topk_ratio=cfg.sol_route_topk_ratio,
            weighted_side=cfg.sol_route_global_weighted_side, key_centroids=kc)
        controller.counts["sol_global_weighted_route_calls"] += 1
    elif virtual_query_data is not None and cfg.sol_virtual_query_route_score != "native_mean":
        from .sol_numerator_virtual_q import build_virtual_anchors, virtual_summaries
        from .spark_route import reweighted_route
        ranges, mapping, anchors = virtual_query_data
        if anchors is None:
            anchors = build_virtual_anchors(q, ranges)
        summaries = virtual_summaries(anchors, k, v)
        route, stats = reweighted_route(q, kc, mapping, summaries[0], summaries[2],
            video_tokens=layout.video_tokens, sink_tokens=sink_tokens,
            topk_ratio=cfg.sol_route_topk_ratio, mode=cfg.sol_virtual_query_route_score)
    elif backend is not None and not partial_video:
        cutoff = {"gemm_radix": gemm_radix_topk_cutoff,
                  "gaussian_moments": triton_gaussian_moment_cutoff}[cfg.sol_route_topk_cutoff_mode]
        threshold, stats = cutoff(q, kc, video_tokens=layout.video_tokens,
                                   sink_tokens=sink_tokens, topk_ratio=cfg.sol_route_topk_ratio)
    else:
        route, stats = _sol_topk_route(q, kc, video_tokens=layout.video_tokens,
                                      sink_tokens=sink_tokens, topk_ratio=cfg.sol_route_topk_ratio)
        stats["route_threshold_mode"] = (
            "topk_explicit_partial_video_fallback" if partial_video else "topk_explicit_non_sm120")

    if cfg.sol_log_density and controller.sol_route_density is None:
        if route is None:
            stats.update(_sol_topk_analytic_density(stats))
        else:
            qb, kb = stats["query_video_blocks"], stats["candidate_video_blocks"]
            stats.update(
                additional_topk_density=stats["target_topk_blocks_per_query"] / kb,
                effective_video_density=float(route[:, :qb, :, :kb].float().mean().item()),
                effective_packed_density=float(route[:, :qb].float().mean().item()))
        controller.sol_route_density = stats

    if virtual_query_data is not None:
        from .sol_numerator_virtual_q import virtual_q_attention, virtual_q_backend
        ranges, mapping, anchors = virtual_query_data
        output = virtual_q_attention(q, k, v, virtual_ranges=ranges, leaf_to_virtual=mapping,
            virtual_anchors=anchors, precomputed_summaries=summaries, key_centroids=kc,
            value_sums=vs, threshold=threshold, route=route, sink_start=layout.video_tokens,
            sink_tokens=sink_tokens, force_local_blocks=cfg.sol_local_blocks_enabled,
            _query_tokens=_query_tokens)
        controller.sol_backend = virtual_q_backend(q)
        controller.counts["sol_virtual_query_calls"] += 1
    elif route is None:
        output = sol_topk_threshold_attn(q, k, v, kc, vs, threshold,
            sink_start=layout.video_tokens, sink_tokens=sink_tokens,
            force_local_blocks=cfg.sol_local_blocks_enabled)
        controller.sol_backend = f"{backend}:{cfg.sol_route_topk_cutoff_mode}"
        controller.counts["sol_topk_threshold_calls"] += 1
    else:
        output = rope_sol_attn(q, k, v, kc, vs, route.to(torch.uint8),
                               sink_start=layout.video_tokens, sink_tokens=sink_tokens)
        controller.sol_backend = rope_sol_backend(q.device)
        controller.counts["sol_topk_explicit_fallback_calls"] += 1
    controller.counts["sol_topk_calls"] += 1
    return output
