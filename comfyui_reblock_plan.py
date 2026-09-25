"""ComfyUI-owned Landmark-v2 Q/K permutation planning.

This is intentionally separate from the Diffusers Spark dispatcher and its
controller.  It shares only low-level mathematical primitives.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def build_comfy_reblock_permutations(controller, query_bthd, key_bthd, layout):
    from h3_sparse_attention.landmark_direction import landmark_direction_factors
    from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices
    from h3_sparse_attention.rope_sol_kernel import BLOCK_SIZE

    if query_bthd.shape != key_bthd.shape or query_bthd.ndim != 4:
        raise ValueError("paired ComfyUI reblock requires equal BTHD Q/K")
    cfg = controller.config
    batch, _, heads, dim = query_bthd.shape
    video_tokens = int(layout.video_tokens)
    flat_batch = batch * heads
    clusters = video_tokens // BLOCK_SIZE
    fused_root_scores = (
        cfg.landmark_tree_v2_initial_order == "flat"
        and cfg.landmark_tree_v2_landmark_mode == "midpoint"
        and cfg.landmark_tree_v2_landmark_count == 32
        and cfg.landmark_tree_v2_aggregation == "linear"
        and cfg.landmark_tree_v2_distance == "cosine"
        and cfg.landmark_tree_v2_mean_mode == "raw"
        and cfg.landmark_tree_v2_group_size == 1
        and dim == 128
        and video_tokens % BLOCK_SIZE == 0
    )

    metric_key = (
        "metric_indices",
        query_bthd.device.type,
        query_bthd.device.index,
        layout.grid,
        fused_root_scores,
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
        ridge=cfg.rope_sol_key_ridge_epsilon,
        moment=cfg.landmark_tree_v2_moment_mode,
    )

    def get_plan(role, plan_batch):
        plan_key = (
            "prepared_qk", role,
            cfg.landmark_tree_v2_initial_order,
            cfg.landmark_tree_v2_children,
            cfg.landmark_tree_v2_fanout_mode,
            cfg.landmark_tree_v2_final_fanout,
            cfg.landmark_tree_v2_root_fanout,
            cfg.landmark_tree_v2_landmark_mode,
            cfg.landmark_tree_v2_landmark_count,
            cfg.landmark_tree_v2_aggregation,
            cfg.landmark_tree_v2_distance,
            cfg.landmark_tree_v2_mean_mode,
            cfg.landmark_tree_v2_moment_mode,
            cfg.landmark_tree_v2_order_mode,
            cfg.landmark_tree_v2_group_size,
            query.device.type,
            query.device.index,
            plan_batch,
            video_tokens,
            dim,
            layout.grid,
        )
        plan = controller.rope_sol_key_clustering_static.get(plan_key)
        if plan is not None:
            return plan
        plan = PreparedLandmarkTreeV2Permutation(
            distance=cfg.landmark_tree_v2_distance,
            max_children=cfg.landmark_tree_v2_children,
            fanout_mode=cfg.landmark_tree_v2_fanout_mode,
            final_fanout=cfg.landmark_tree_v2_final_fanout,
            root_fanout=cfg.landmark_tree_v2_root_fanout,
            landmark_mode=cfg.landmark_tree_v2_landmark_mode,
            landmark_count=cfg.landmark_tree_v2_landmark_count,
            aggregation=cfg.landmark_tree_v2_aggregation,
            input_unit_means=cfg.landmark_tree_v2_mean_mode == "input_unit",
            metric_unit_means=cfg.landmark_tree_v2_mean_mode == "metric_unit",
            order_mode=cfg.landmark_tree_v2_order_mode,
            group_size=cfg.landmark_tree_v2_group_size,
            batch=plan_batch,
            tokens=video_tokens,
            dim=dim,
            grid_shape=layout.grid,
            initial_order=cfg.landmark_tree_v2_initial_order,
            device=query.device,
            # The fused comfy-kitchen path scatters exact output directly
            # through query_permutation, so no inverse permutation is needed.
            return_inverse=False,
            # Keep this exact route scheduling optimization ComfyUI-local;
            # Diffusers plans retain their independent default execution.
            compact_direct_route=True,
        )
        controller.rope_sol_key_clustering_static[plan_key] = plan
        return plan

    if fused_root_scores:
        key_plan = get_plan("key_fused_root", flat_batch)
        query_plan = get_plan("query_fused_root", flat_batch)
        plan = None
    else:
        plan = get_plan("combined", 2 * flat_batch)
        key_plan = query_plan = None

    if fused_root_scores:
        key_transformed = key_plan.graph_input
        if key_transformed is None:
            key_transformed = torch.empty(
                (flat_batch, video_tokens, dim),
                device=query.device, dtype=torch.bfloat16,
            )
        query_transformed = query_plan.graph_input
        if query_transformed is None:
            query_transformed = torch.empty_like(key_transformed)
    else:
        transformed = plan.graph_input
        if transformed is None:
            transformed = torch.empty(
                (2 * flat_batch, video_tokens, dim),
                device=query.device,
                dtype=torch.bfloat16,
            )
    root_scores = None
    if fused_root_scores:
        from h3_sparse_attention.landmark_projection import project_bthd
        from h3_sparse_attention.landmark_v2_fused_node import fused_proxy_directions
        from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy

        # Cholesky returns a column-major view.  Materializing these tiny
        # 128x128 factors in row-major order lets the fused full-token kernel
        # use coalesced loads while preserving X @ L (not X @ L.T).
        query_factor = query_metric.to(torch.bfloat16).contiguous()
        key_factor = key_metric.to(torch.bfloat16).contiguous()
        root_key = (
            "fused_root_static", query.device.type, query.device.index,
            video_tokens, flat_batch, layout.grid,
            cfg.landmark_tree_v2_children,
            cfg.landmark_tree_v2_fanout_mode,
            cfg.landmark_tree_v2_final_fanout,
            cfg.landmark_tree_v2_root_fanout,
        )
        root_static = controller.rope_sol_key_clustering_static.get(root_key)
        if root_static is None:
            landmark = torch.arange(32, device=query.device)
            starts = landmark * video_tokens // 32
            ends = (landmark + 1) * video_tokens // 32
            midpoint_indices = starts + (ends - starts) // 2
            weights = (ends - starts).to(torch.int32)
            hierarchy = build_reblock_hierarchy(
                video_tokens,
                (cfg.landmark_tree_v2_children,),
                grid_shape=tuple(layout.grid),
                final_fanout=cfg.landmark_tree_v2_final_fanout,
                root_fanout=cfg.landmark_tree_v2_root_fanout,
                fanout_mode=cfg.landmark_tree_v2_fanout_mode,
            )
            root_capacities = tuple(
                BLOCK_SIZE * value for value in hierarchy.budgets(0, clusters)
            )
            root_static = (midpoint_indices, weights, root_capacities)
            controller.rope_sol_key_clustering_static[root_key] = root_static
        midpoint_indices, root_weights, root_capacities = root_static
        centers = torch.empty(
            (2 * flat_batch, 32, dim), device=query.device, dtype=torch.bfloat16
        )
        torch.bmm(
            key.index_select(1, midpoint_indices), query_factor,
            out=centers[:flat_batch],
        )
        torch.bmm(
            query.index_select(1, midpoint_indices), key_factor,
            out=centers[flat_batch:],
        )
        directions = fused_proxy_directions(
            centers,
            root_weights[None, :].expand(2 * flat_batch, -1),
            root_capacities,
        )
        key_root_scores = key_plan.graph_root_scores
        if key_root_scores is None:
            key_root_scores = torch.empty(
                (flat_batch, video_tokens, len(root_capacities) - 1),
                device=query.device, dtype=torch.float32,
            )
        query_root_scores = query_plan.graph_root_scores
        if query_root_scores is None:
            query_root_scores = torch.empty_like(key_root_scores)

        def project_key():
            project_bthd(
                key_bthd[:, :video_tokens], query_factor, out=key_transformed,
                block_m=128,
                score_directions=directions[:flat_batch],
                score_out=key_root_scores,
            )

        def project_query():
            project_bthd(
                query_bthd[:, :video_tokens], key_factor, out=query_transformed,
                block_m=128,
                score_directions=directions[flat_batch:],
                score_out=query_root_scores,
            )

        if key_plan.graph_active and query_plan.graph_active:
            stream_key = (
                "fused_root_stream", query.device.type, query.device.index
            )
            auxiliary = controller.rope_sol_key_clustering_static.get(stream_key)
            if auxiliary is None:
                auxiliary = torch.cuda.Stream(device=query.device)
                controller.rope_sol_key_clustering_static[stream_key] = auxiliary
            current = torch.cuda.current_stream(query.device)
            auxiliary.wait_stream(current)
            with torch.cuda.stream(auxiliary):
                project_key()
                key_permutation, key_inverse = key_plan.replay()
            project_query()
            query_permutation, query_inverse = query_plan.replay()
            current.wait_stream(auxiliary)
        else:
            project_key()
            project_query()
            key_permutation, key_inverse = key_plan.run(
                key_transformed, root_scores=key_root_scores
            )
            query_permutation, query_inverse = query_plan.run(
                query_transformed, root_scores=query_root_scores
            )
    else:
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

    inverse_norms = None
    if not fused_root_scores and cfg.landmark_tree_v2_mean_mode == "input_unit":
        inverse_norms = plan.graph_inverse_norms
        if inverse_norms is None:
            inverse_norms = torch.empty(
                (2 * flat_batch, video_tokens, 1),
                device=query.device,
                dtype=torch.float32,
            )
        inverse_norms[:flat_batch].copy_(
            key.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal()
        )
        inverse_norms[flat_batch:].copy_(
            query.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal()
        )

    if not fused_root_scores:
        if plan.graph_active:
            combined_permutation, combined_inverse = plan.replay()
        else:
            combined_permutation, combined_inverse = plan.run(
                transformed, inverse_norms=inverse_norms
            )
        key_permutation = combined_permutation[:flat_batch]
        query_permutation = combined_permutation[flat_batch:]
        key_inverse = (
            combined_inverse[:flat_batch] if combined_inverse is not None else None
        )
        query_inverse = (
            combined_inverse[flat_batch:] if combined_inverse is not None else None
        )
        hierarchy = plan.hierarchy
    else:
        hierarchy = query_plan.hierarchy
    controller.landmark_reblock_hierarchy = hierarchy
    controller.counts["reblock_plan_calls"] += 1

    def unflatten(value):
        if value is None:
            return None
        return value.reshape(batch, heads, video_tokens)

    return (
        unflatten(query_permutation),
        unflatten(query_inverse),
        unflatten(key_permutation),
        hierarchy,
    )
