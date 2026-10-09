"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math
import os


from typing import Any


import torch


import torch.nn.functional as F
from .device_policy import is_a800_80gb


_LOG2_E = math.log2(math.e)
_SPARK_BLOCK_SIZE = 64
_SUPPORTED_SPARK_CAPABILITIES = frozenset({(8, 0), (8, 9), (9, 0), (10, 0), (12, 0)})
_PROFILE_REBLOCK = os.environ.get("SPARK_PROFILE_REBLOCK") == "1"


def _profile_begin(controller, name: str):
    if not _PROFILE_REBLOCK:
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return name, start, end


def _profile_end(controller, sample) -> None:
    if sample is None:
        return
    name, start, end = sample
    end.record()
    samples = getattr(controller, "spark_reblock_profile", None)
    if samples is None:
        samples = []
        controller.spark_reblock_profile = samples
    samples.append((name, start, end))


def spark_reblock_profile_summary(controller, *, clear: bool = True):
    """Synchronize and summarize opt-in reblock CUDA event samples."""

    samples = getattr(controller, "spark_reblock_profile", None)
    if not samples:
        return None
    torch.cuda.synchronize()
    durations = {}
    for name, start, end in samples:
        durations.setdefault(name, []).append(start.elapsed_time(end))
    if clear:
        controller.spark_reblock_profile = []
    return {
        name: {
            "total_ms": sum(values),
            "calls": len(values),
            "mean_ms": sum(values) / len(values),
            "median_ms": sorted(values)[len(values) // 2],
            "max_ms": max(values),
        }
        for name, values in durations.items()
    }


def _spark_query_tokens(
    total_tokens: int,
    video_tokens: int,
    tail_mode: str = "dense",
) -> int:
    """Return the query prefix evaluated by the sparse attention kernel."""

    if tail_mode == "dense":
        return min(
            total_tokens,
            (video_tokens // _SPARK_BLOCK_SIZE) * _SPARK_BLOCK_SIZE,
        )
    if tail_mode != "pad":
        raise ValueError("tail_mode must be 'dense' or 'pad'")
    padded_tokens = math.ceil(video_tokens / _SPARK_BLOCK_SIZE) * _SPARK_BLOCK_SIZE
    if padded_tokens > total_tokens:
        raise ValueError(
            "the packed suffix is too short to pad the final video query block"
        )
    return padded_tokens


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


@torch.no_grad()
def _record_exact_block_verbose(
    controller,
    q: torch.Tensor,
    key_centroids: torch.Tensor,
    threshold: torch.Tensor | None,
    *,
    video_tokens: int,
    query_tokens: int,
    radius: int | None,
) -> None:
    """Accumulate local/Top-K overlap without materializing the full route."""

    if (
        os.environ.get("H3_VERBOSE_EXACT_BLOCKS") != "1"
        or radius is None
        or threshold is None
    ):
        return
    from .sol_topk_cutoff import BLOCK_SIZE, HEAD_DIM

    query_blocks = query_tokens // BLOCK_SIZE
    candidate_blocks = video_tokens // BLOCK_SIZE
    batch, _, heads, _ = q.shape
    query_centroids = (
        q[:, :query_tokens]
        .view(batch, query_blocks, BLOCK_SIZE, heads, HEAD_DIM)
        .sum(dim=2, dtype=torch.float32)
        .mul_(1.0 / BLOCK_SIZE)
        .to(torch.bfloat16)
    )
    qids = torch.arange(query_blocks, device=q.device)
    finite_rows = torch.isfinite(threshold[:, :query_blocks])
    local_candidates = torch.zeros((), device=q.device, dtype=torch.int64)
    already_selected = torch.zeros_like(local_candidates)
    for offset in range(-radius, radius + 1):
        kids = qids + offset
        valid = (kids >= 0) & (kids < candidate_blocks)
        gathered = key_centroids[:, kids.clamp(0, candidate_blocks - 1)]
        scores = torch.einsum("bqhd,bqhd->bqh", query_centroids, gathered).float()
        scores.mul_(HEAD_DIM**-0.5 * _LOG2_E)
        valid_rows = valid[None, :, None] & finite_rows
        local_candidates += valid_rows.sum()
        already_selected += (valid_rows & (scores > threshold[:, :query_blocks])).sum()
    added = local_candidates - already_selected
    accumulator = controller.exact_block_verbose_accumulator
    if accumulator is None:
        accumulator = {
            "radius": radius,
            "route_rows": torch.zeros_like(local_candidates),
            "local_candidates": torch.zeros_like(local_candidates),
            "already_selected": torch.zeros_like(local_candidates),
            "added_exact_blocks": torch.zeros_like(local_candidates),
        }
        controller.exact_block_verbose_accumulator = accumulator
    elif accumulator["radius"] != radius:
        raise RuntimeError("exact-block verbose radius changed within one run")
    accumulator["route_rows"].add_(finite_rows.sum())
    accumulator["local_candidates"].add_(local_candidates)
    accumulator["already_selected"].add_(already_selected)
    accumulator["added_exact_blocks"].add_(added)


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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the fused CUDA permutation, retaining PyTorch as the oracle."""

    if values.is_cuda and values.is_contiguous() and permutation.is_contiguous():
        from .landmark_tree_v2_triton import headwise_permute_bthd

        return headwise_permute_bthd(
            values, permutation, video_tokens=video_tokens, out=out
        )
    result = _torch_headwise_permute_video_tokens(
        values, permutation, video_tokens=video_tokens
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def _required_reblock_m2_side(m2_side: str, layout_reuse: str) -> str:
    """Intersect requested feature weighting with the layouts we actually build.

    The side names describe transformed clustering features: ``key`` consumes
    the query moment and ``query`` consumes the key moment.  A shared layout
    only builds its source side, so constructing the other moment cannot affect
    either returned permutation.
    """

    if layout_reuse == "independent":
        return m2_side
    if layout_reuse == "q_from_k":
        return "key" if m2_side in ("both", "key") else "none"
    if layout_reuse == "k_from_q":
        return "query" if m2_side in ("both", "query") else "none"
    raise ValueError(f"unknown layout reuse mode: {layout_reuse}")


def _m2_side_from_features(*, query: bool, key: bool) -> str:
    if query and key:
        return "both"
    if query:
        return "query"
    if key:
        return "key"
    return "none"


@torch.no_grad()
def _landmark_tree_v2_combined_permutations(
    controller: _Controller,
    query_bthd: torch.Tensor,
    key_bthd: torch.Tensor,
    layout: PackedLayout,
    *,
    expand_shared: bool = True,
):
    """Metric transform plus prepared-plan tree build on the current stream.

    Returns ``(plan, metric_indices, combined_permutation, combined_inverse)``.
    Independent layouts use ``[K; Q]`` batches.  A shared-layout ablation only
    constructs the source-side batch.  Direct diagnostic callers retain the
    historical doubled return by default; the production Q/K wrapper requests
    the single batch and aliases it instead of materializing duplicate indices.
    """

    from .landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    from .landmark_direction import (
        landmark_direction_factors,
        landmark_single_direction_factor,
    )
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
    layout_reuse = controller.config.landmark_tree_v2_layout_reuse
    device_capability = (
        tuple(torch.cuda.get_device_capability(query.device))
        if query.is_cuda else None
    )
    eager_low_memory = bool(
        query.is_cuda
        and video_tokens > 90_000
        and is_a800_80gb(query.device)
    )
    required_m2_side = _required_reblock_m2_side(
        controller.config.landmark_tree_v2_m2_side, layout_reuse
    )
    profile = _profile_begin(controller, "reblock_metric")
    if required_m2_side == "none":
        # None denotes an identity transform.  Keeping it implicit avoids both
        # a batched D x D identity allocation and an O(N D^2) identity BMM.
        query_metric = key_metric = None
    elif required_m2_side in ("query", "key"):
        single_metric = landmark_single_direction_factor(
            query,
            key,
            metric_indices,
            required_m2_side,
            ridge=controller.config.rope_sol_key_ridge_epsilon,
            moment=controller.config.landmark_tree_v2_moment_mode,
            m2_estimator=controller.config.landmark_tree_v2_m2_estimator,
        )
        query_metric = single_metric if required_m2_side == "key" else None
        key_metric = single_metric if required_m2_side == "query" else None
    else:
        query_metric, key_metric = landmark_direction_factors(
            query,
            key,
            metric_indices,
            ridge=controller.config.rope_sol_key_ridge_epsilon,
            moment=controller.config.landmark_tree_v2_moment_mode,
            m2_estimator=controller.config.landmark_tree_v2_m2_estimator,
            m2_side=required_m2_side,
        )
    _profile_end(controller, profile)
    plan_batch = 2 * flat_batch if layout_reuse == "independent" else flat_batch
    plan_key = (
        "prepared_landmark_tree_v2_qk",
        controller.config.landmark_tree_v2_initial_order,
        controller.config.landmark_tree_v2_children,
        controller.config.landmark_tree_v2_fanout_mode,
        controller.config.landmark_tree_v2_final_fanout,
        controller.config.landmark_tree_v2_root_fanout,
        controller.config.landmark_tree_v2_landmark_mode,
        controller.config.landmark_tree_v2_landmark_count,
        controller.config.landmark_tree_v2_midpoint_direction_mode,
        controller.config.landmark_tree_v2_aggregation,
        controller.config.landmark_tree_v2_distance,
        controller.config.landmark_tree_v2_mean_mode,
        controller.config.landmark_tree_v2_moment_mode,
        controller.config.landmark_tree_v2_order_mode,
        controller.config.landmark_tree_v2_group_size,
        controller.config.landmark_tree_v2_m2_side,
        controller.config.landmark_tree_v2_m2_estimator,
        controller.config.landmark_tree_v2_proxy_iterations,
        controller.config.landmark_tree_v2_proxy_seed_rule,
        controller.config.landmark_tree_v2_proxy_update_rule,
        layout_reuse,
        query.device.type,
        query.device.index,
        plan_batch,
        video_tokens,
        dim,
        layout.grid,
    )
    plan = controller.rope_sol_key_clustering_static.get(plan_key)
    if plan is None:
        plan = PreparedLandmarkTreeV2Permutation(
            distance=controller.config.landmark_tree_v2_distance,
            max_children=controller.config.landmark_tree_v2_children,
            fanout_mode=controller.config.landmark_tree_v2_fanout_mode,
            final_fanout=controller.config.landmark_tree_v2_final_fanout,
            root_fanout=controller.config.landmark_tree_v2_root_fanout,
            landmark_mode=controller.config.landmark_tree_v2_landmark_mode,
            landmark_count=controller.config.landmark_tree_v2_landmark_count,
            midpoint_direction_mode=controller.config.landmark_tree_v2_midpoint_direction_mode,
            aggregation=controller.config.landmark_tree_v2_aggregation,
            input_unit_means=controller.config.landmark_tree_v2_mean_mode == "input_unit",
            metric_unit_means=controller.config.landmark_tree_v2_mean_mode == "metric_unit",
            order_mode=controller.config.landmark_tree_v2_order_mode,
            group_size=controller.config.landmark_tree_v2_group_size,
            proxy_iterations=controller.config.landmark_tree_v2_proxy_iterations,
            seed_rule=controller.config.landmark_tree_v2_proxy_seed_rule,
            update_rule=controller.config.landmark_tree_v2_proxy_update_rule,
            batch=plan_batch,
            tokens=video_tokens,
            dim=dim,
            grid_shape=layout.grid,
            initial_order=controller.config.landmark_tree_v2_initial_order,
            device=query.device,
            compact_indices=device_capability in ((8, 0), (8, 9), (12, 0)),
        )
        controller.rope_sol_key_clustering_static[plan_key] = plan
    # This eager fallback is designed and validated for A800 80GB long-sequence
    # inference. A successful graph capture keeps its private pool alive through
    # attention. Shared-layout plans are small enough to capture while the
    # larger independent plan often falls back to eager execution, which made
    # the nominally cheaper shared path use more peak memory.
    transformed = None if eager_low_memory else plan.acquire_graph_input()
    if transformed is None:
        transformed = torch.empty(
            (plan_batch, video_tokens, dim),
            device=query.device,
            dtype=torch.bfloat16,
        )
    profile = _profile_begin(controller, "reblock_transform_bmm")
    if layout_reuse in ("independent", "q_from_k"):
        if query_metric is None:
            transformed[:flat_batch].copy_(key)
        else:
            torch.bmm(
                key.to(torch.bfloat16), query_metric.to(torch.bfloat16),
                out=transformed[:flat_batch],
            )
    if layout_reuse == "independent":
        if key_metric is None:
            transformed[flat_batch:].copy_(query)
        else:
            torch.bmm(
                query.to(torch.bfloat16), key_metric.to(torch.bfloat16),
                out=transformed[flat_batch:],
            )
    elif layout_reuse == "k_from_q":
        if key_metric is None:
            transformed[:flat_batch].copy_(query)
        else:
            torch.bmm(
                query.to(torch.bfloat16), key_metric.to(torch.bfloat16),
                out=transformed[:flat_batch],
            )
    _profile_end(controller, profile)
    # M2 ablations must not gain or lose quality merely because their feature
    # norm chooses a different exact tail.  Reconstruct the production
    # Hilbert-M2 feature norm and freeze that identity for both independent
    # sides (or for the single source side of a shared layout).
    excluded_indices = None
    remainder = video_tokens % BLOCK_SIZE
    needs_baseline_metric_tail = (
        controller.config.landmark_tree_v2_m2_side != "both"
        or controller.config.landmark_tree_v2_m2_estimator != "hilbert_midpoint"
    )
    needs_flat_order_tail = controller.config.landmark_tree_v2_initial_order != "flat"
    # A multiple-of-64 sequence has no protected remainder.  Besides avoiding
    # the reference M2/BMM, retaining None here is essential: an empty tensor
    # would deliberately force PreparedLandmarkTreeV2Permutation into eager
    # mode and disable its CUDA graph replay.
    if remainder and (needs_baseline_metric_tail or needs_flat_order_tail):
        from .landmark_tree_clustering import _largest_norm_remainder

        profile = _profile_begin(controller, "reblock_frozen_tail_reference")
        if needs_baseline_metric_tail:
            baseline_side = _required_reblock_m2_side("both", layout_reuse)
            current_is_baseline = (
                controller.config.landmark_tree_v2_m2_estimator == "hilbert_midpoint"
            )
            reuse_key_feature = (
                current_is_baseline
                and baseline_side in ("both", "key")
                and required_m2_side in ("both", "key")
            )
            reuse_query_feature = (
                current_is_baseline
                and baseline_side in ("both", "query")
                and required_m2_side in ("both", "query")
            )
            missing_side = _m2_side_from_features(
                query=(
                    baseline_side in ("both", "query")
                    and not reuse_query_feature
                ),
                key=(
                    baseline_side in ("both", "key")
                    and not reuse_key_feature
                ),
            )
            if missing_side == "none":
                reference = transformed
                baseline_query_metric = baseline_key_metric = None
            else:
                if missing_side in ("query", "key"):
                    baseline_single_metric = landmark_single_direction_factor(
                        query, key, metric_indices, missing_side,
                        ridge=controller.config.rope_sol_key_ridge_epsilon,
                        moment=controller.config.landmark_tree_v2_moment_mode,
                        m2_estimator="hilbert_midpoint",
                    )
                    baseline_query_metric = (
                        baseline_single_metric if missing_side == "key" else None
                    )
                    baseline_key_metric = (
                        baseline_single_metric if missing_side == "query" else None
                    )
                else:
                    baseline_query_metric, baseline_key_metric = landmark_direction_factors(
                        query, key, metric_indices,
                        ridge=controller.config.rope_sol_key_ridge_epsilon,
                        moment=controller.config.landmark_tree_v2_moment_mode,
                        m2_estimator="hilbert_midpoint", m2_side=missing_side,
                    )
                reference = torch.empty_like(transformed)
                if layout_reuse in ("independent", "q_from_k"):
                    if reuse_key_feature:
                        reference[:flat_batch].copy_(transformed[:flat_batch])
                    else:
                        torch.bmm(
                            key.to(torch.bfloat16),
                            baseline_query_metric.to(torch.bfloat16),
                            out=reference[:flat_batch],
                        )
                if layout_reuse == "independent":
                    if reuse_query_feature:
                        reference[flat_batch:].copy_(transformed[flat_batch:])
                    else:
                        torch.bmm(
                            query.to(torch.bfloat16),
                            baseline_key_metric.to(torch.bfloat16),
                            out=reference[flat_batch:],
                        )
                elif layout_reuse == "k_from_q":
                    if reuse_query_feature:
                        reference[:flat_batch].copy_(transformed[:flat_batch])
                    else:
                        torch.bmm(
                            query.to(torch.bfloat16),
                            baseline_key_metric.to(torch.bfloat16),
                            out=reference[:flat_batch],
                        )
        else:
            # Changing only the root order must not change which original
            # tokens occupy the protected exact remainder.  The current
            # transformed features are already the baseline both-side
            # Hilbert-M2 features, so select the tail before applying the
            # alternative root permutation.
            reference = transformed
        mask = _largest_norm_remainder(reference, remainder, validate=True)
        original_ids = torch.arange(
            video_tokens, device=query.device, dtype=torch.long
        ).expand(plan_batch, -1)
        excluded_indices = original_ids[mask].reshape(plan_batch, remainder)
        controller.spark_reblock_frozen_tail_indices = excluded_indices.detach()
        if needs_baseline_metric_tail:
            del reference, baseline_query_metric, baseline_key_metric
        _profile_end(controller, profile)
    inverse_norms = None
    if controller.config.landmark_tree_v2_mean_mode == "input_unit":
        inverse_norms = plan.graph_inverse_norms
        if inverse_norms is None:
            inverse_norms = torch.empty((plan_batch, video_tokens, 1), device=query.device, dtype=torch.float32)
        if layout_reuse in ("independent", "q_from_k"):
            inverse_norms[:flat_batch].copy_(key.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal())
        if layout_reuse == "independent":
            inverse_norms[flat_batch:].copy_(query.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal())
        elif layout_reuse == "k_from_q":
            inverse_norms[:flat_batch].copy_(query.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).reciprocal())
    profile = _profile_begin(controller, "reblock_tree")
    if not eager_low_memory and plan.graph_active and excluded_indices is None:
        combined_permutation, combined_inverse = plan.replay()
    else:
        combined_permutation, combined_inverse = plan.run(
            transformed, inverse_norms=inverse_norms,
            excluded_indices=excluded_indices,
            allow_cuda_graph=not eager_low_memory,
        )
    _profile_end(controller, profile)
    # The graph input is dead until the next reblock build. On the SM80 long
    # path it is large enough to serve as the shared video-permutation
    # workspace, avoiding another full BTHD allocation during attention.
    controller.spark_reblock_permute_workspace = transformed
    del query_metric, key_metric
    if layout_reuse != "independent" and expand_shared:
        combined_permutation = torch.cat(
            (combined_permutation, combined_permutation), dim=0
        )
        combined_inverse = torch.cat((combined_inverse, combined_inverse), dim=0)
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
        _landmark_tree_v2_combined_permutations(
            controller, query_bthd, key_bthd, layout, expand_shared=False
        )
    )
    layout_reuse = controller.config.landmark_tree_v2_layout_reuse
    if layout_reuse == "independent":
        key_permutation = combined_permutation[:flat_batch]
        key_inverse = combined_inverse[:flat_batch]
        query_permutation = combined_permutation[flat_batch:]
        query_inverse = combined_inverse[flat_batch:]
    else:
        # Shared-layout modes deliberately use the same physical ordering for
        # Q and K.  Keep that relationship as an alias: torch.cat would retain
        # two redundant index batches in addition to the CUDA graph's source
        # permutation and inverse on memory-bound long-sequence inference.
        key_permutation = query_permutation = combined_permutation
        key_inverse = query_inverse = combined_inverse
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
        "m2_side": controller.config.landmark_tree_v2_m2_side,
        "m2_estimator": controller.config.landmark_tree_v2_m2_estimator,
        "metric_centered": False,
        "raw_remainder_features": True,
        "order_mode": controller.config.landmark_tree_v2_order_mode,
        "landmark_initialization": (
            "contiguous_mean_token_interval_mean"
            if controller.config.landmark_tree_v2_landmark_mode == "mean"
            else "contiguous_interval_midpoint_token"
        ),
        "landmark_mode": controller.config.landmark_tree_v2_landmark_mode,
        "landmark_input_space": "m2_transformed",
        "landmark_reduction_dtype": (
            "float32" if plan.landmark_mode == "mean"
            else "not_applicable_midpoint_selection"
        ),
        "landmark_metric_transform_dtype": "already_transformed_input",
        "landmark_transform_order": "before_compression",
        "midpoint_direction_mode": controller.config.landmark_tree_v2_midpoint_direction_mode,
        "coarse_landmarks": controller.config.landmark_tree_v2_landmark_count,
        "coarse_assignment_passes": 0,
        "group_size": controller.config.landmark_tree_v2_group_size,
        "root_children": (plan.max_children if isinstance(plan.max_children, int) else plan.max_children[0]),
        "later_children": (plan.max_children if isinstance(plan.max_children, int) else plan.max_children[min(1, len(plan.max_children) - 1)]),
        "root_fanout": plan.root_fanout,
        "final_fanout": plan.final_fanout,
        "proxy_iterations": controller.config.landmark_tree_v2_proxy_iterations,
        "proxy_seed_rule": controller.config.landmark_tree_v2_proxy_seed_rule,
        "proxy_update_rule": controller.config.landmark_tree_v2_proxy_update_rule,
        "layout_reuse": controller.config.landmark_tree_v2_layout_reuse,
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

    q_bthd, k_bthd, v_bthd = (
        tensor.permute(0, 2, 1, 3).contiguous() for tensor in (q, k, v)
    )
    return spark_attention_bthd(
        controller, q_bthd, k_bthd, v_bthd, layout, layer,
        return_bthd=return_bthd,
    )


def _sm80_low_memory_reblock(tensor: torch.Tensor) -> bool:
    """Select the long-sequence storage-reuse path only on A800 80GB."""
    return bool(
        tensor.is_cuda
        and tensor.shape[1] > 90_000
        and is_a800_80gb(tensor.device)
    )


def spark_attention_bthd(
    controller: _Controller,
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    layout: PackedLayout,
    layer: int,
    *, return_bthd: bool = True,
) -> torch.Tensor:
    """Spark attention for native BTHD producers such as ComfyUI's H3 path."""

    cfg = controller.config
    if q_bthd.dtype != torch.bfloat16 or q_bthd.shape[-1] != 128:
        raise RuntimeError(
            "Sol-Attn requires contiguous BF16 Q/K/V with head dimension 128; "
            f"got dtype={q_bthd.dtype}, head_dim={q_bthd.shape[-1]}"
        )
    if q_bthd.shape != k_bthd.shape or q_bthd.shape != v_bthd.shape:
        raise RuntimeError("Spark attention requires matching BTHD Q/K/V tensors")
    if not q_bthd.is_contiguous() or not k_bthd.is_contiguous() or not v_bthd.is_contiguous():
        raise RuntimeError("Spark attention requires contiguous BTHD Q/K/V tensors")
    if q_bthd.device.type != "cuda":
        raise RuntimeError("Spark-H3 requires CUDA tensors")
    capability = tuple(torch.cuda.get_device_capability(q_bthd.device))
    if capability not in _SUPPORTED_SPARK_CAPABILITIES:
        supported = ", ".join(
            f"SM{major}{minor}" for major, minor in sorted(_SUPPORTED_SPARK_CAPABILITIES)
        )
        raise RuntimeError(
            f"Spark-H3 has no supported kernel for SM{capability[0]}{capability[1]}; "
            f"supported architectures are {supported}"
        )
    sm80_low_memory = _sm80_low_memory_reblock(q_bthd)
    reblock_workspace = (
        getattr(controller, "spark_reblock_permute_workspace", None)
        if sm80_low_memory else None
    )

    # The producer has already placed the target-video grid first and all
    # packed context after it.
    sink_start = layout.video_tokens
    sink_tokens = layout.sequence_length - layout.video_tokens
    legacy_full_query = cfg.sol_legacy_full_query
    sparse_query_tokens = (
        layout.video_tokens
        if legacy_full_query and layout.video_tokens % _SPARK_BLOCK_SIZE == 0
        else q_bthd.shape[1]
        if legacy_full_query
        else _spark_query_tokens(
            q_bthd.shape[1], layout.video_tokens, cfg.sol_video_tail_mode
        )
    )
    needs_query_padding = (
        cfg.sol_video_tail_mode == "pad"
        and sparse_query_tokens > layout.video_tokens
    )
    controller.sol_backend = "comfy-kitchen-spark"
    query_inverse_permutation = None
    virtual_query_data = None
    if cfg.sol_landmark_preprocess:
        landmark_builder = _landmark_tree_v2_qk_block_permutations
        reuse_layers = int(getattr(controller, "spark_reblock_reuse_layers", 1))
        reuse_start = int(getattr(controller, "spark_reblock_reuse_start_layer", 0))
        reuse_group = (
            (layer - reuse_start) // reuse_layers
            if reuse_layers > 1 and layer >= reuse_start
            else None
        )
        reuse_key = (controller.evaluation_index, reuse_group)
        cached_reblock = getattr(controller, "spark_last_reblock", None)
        group_start = reuse_start + reuse_group * reuse_layers if reuse_group is not None else layer
        if reuse_group is not None and layer != group_start and cached_reblock is not None and cached_reblock[0] == reuse_key:
            (
                query_permutation,
                query_inverse_permutation,
                key_permutation,
                hierarchy,
            ) = cached_reblock[1]
            controller.landmark_reblock_hierarchy = hierarchy
            controller.counts["sol_landmark_reuse_calls"] += 1
        else:
            profile = _profile_begin(controller, "reblock_permutation_build_total")
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
            _profile_end(controller, profile)
            if reuse_group is not None:
                controller.spark_last_reblock = (
                    reuse_key,
                    (
                        query_permutation,
                        query_inverse_permutation,
                        key_permutation,
                        controller.landmark_reblock_hierarchy,
                    ),
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
            if (
                cfg.sol_virtual_query_fused_permute
                and not needs_query_padding
            ):
                profile = _profile_begin(controller, "reblock_q_and_anchors")
                anchor_dtype = (
                    torch.float32
                    if cfg.sol_global_anchor_dtype == "float32"
                    else q_bthd.dtype
                )
                if sm80_low_memory and reblock_workspace is not None:
                    from .landmark_tree_v2_triton import headwise_permute_bthd_low_memory
                    from .sol_numerator_virtual_q import build_virtual_anchors
                    q_bthd = headwise_permute_bthd_low_memory(
                        q_bthd, query_permutation,
                        video_tokens=layout.video_tokens,
                        workspace=reblock_workspace,
                    )
                    anchors = build_virtual_anchors(
                        q_bthd, ranges, dtype=anchor_dtype
                    )
                else:
                    q_bthd, anchors = permute_with_virtual_anchors(
                        q_bthd, query_permutation, ranges,
                        video_tokens=layout.video_tokens,
                        anchor_dtype=anchor_dtype,
                        reuse_input=sm80_low_memory,
                    )
                _profile_end(controller, profile)
                controller.counts["sol_virtual_query_fused_permute_calls"] += 1
            else:
                profile = _profile_begin(controller, "reblock_q")
                if sm80_low_memory:
                    from .landmark_tree_v2_triton import headwise_permute_bthd_low_memory
                    q_bthd = headwise_permute_bthd_low_memory(
                        q_bthd, query_permutation,
                        video_tokens=layout.video_tokens,
                        workspace=reblock_workspace,
                    )
                else:
                    q_bthd = _headwise_permute_video_tokens(
                        q_bthd, query_permutation, video_tokens=layout.video_tokens)
                _profile_end(controller, profile)
            virtual_query_data = (ranges, mapping, anchors)
        else:
            profile = _profile_begin(controller, "reblock_q")
            if sm80_low_memory:
                from .landmark_tree_v2_triton import headwise_permute_bthd_low_memory
                q_bthd = headwise_permute_bthd_low_memory(
                    q_bthd, query_permutation,
                    video_tokens=layout.video_tokens,
                    workspace=reblock_workspace,
                )
            else:
                q_bthd = _headwise_permute_video_tokens(
                    q_bthd, query_permutation, video_tokens=layout.video_tokens
                )
            _profile_end(controller, profile)
        profile = _profile_begin(controller, "reblock_kv")
        if (
            k_bthd.is_cuda
            and k_bthd.is_contiguous()
            and v_bthd.is_contiguous()
            and key_permutation.is_contiguous()
        ):
            from .landmark_tree_v2_triton import (
                headwise_permute_pair_bthd,
                headwise_permute_pair_bthd_low_memory,
            )

            # The 345-frame/768p case is within ~100 MiB of an A800's limit.
            # Reuse the dead K input as its permuted destination there, avoiding
            # one simultaneous full-size BTHD output.  Shorter production
            # shapes retain the faster paired gather.
            permute_pair = (
                headwise_permute_pair_bthd_low_memory
                if sm80_low_memory
                else headwise_permute_pair_bthd
            )
            k_bthd, v_bthd = permute_pair(
                k_bthd,
                v_bthd,
                key_permutation,
                video_tokens=layout.video_tokens,
                **(
                    {"workspace": reblock_workspace}
                    if sm80_low_memory and reblock_workspace is not None
                    else {}
                ),
            )
        else:
            k_bthd = _headwise_permute_video_tokens(
                k_bthd, key_permutation, video_tokens=layout.video_tokens
            )
            v_bthd = _headwise_permute_video_tokens(
                v_bthd, key_permutation, video_tokens=layout.video_tokens
            )
        if sm80_low_memory:
            # The transform buffer has now served Q, K, and V.  Do not retain
            # it through score construction or the output projection.
            controller.spark_reblock_permute_workspace = None
            reblock_workspace = None
        _profile_end(controller, profile)
        controller.counts["sol_landmark_preprocess_calls"] += 1
    elif getattr(controller, "spark_identity_reweight", False):
        # Ablation-only path: preserve the original physical video-token order
        # while applying the same target-189 virtual-query reweighting used by
        # full Spark.  This isolates reweight cost from dynamic reblocking and
        from .landmark_virtual_q import target_virtual_query_layout
        from .reblock_hierarchy import build_reblock_hierarchy
        from .sol_numerator_virtual_q import build_virtual_anchors, validate_virtual_layout
        from .spark_defaults import (
            SPARK_REWEIGHT_MAX_BLOCKS,
            SPARK_REWEIGHT_MIN_BLOCKS,
            SPARK_REWEIGHT_TARGET_BLOCKS,
        )

        cache = controller._virtual_query_layout_cache
        cache_key = ("identity_target189", q_bthd.shape[1], q_bthd.device, layout.grid)
        if cache_key not in cache:
            hierarchy = build_reblock_hierarchy(
                layout.video_tokens,
                cfg.landmark_tree_v2_children,
                grid_shape=layout.grid,
                fanout_mode=cfg.landmark_tree_v2_fanout_mode,
            )
            topology = target_virtual_query_layout(
                layout.video_tokens,
                q_bthd.shape[1],
                SPARK_REWEIGHT_TARGET_BLOCKS,
                SPARK_REWEIGHT_MIN_BLOCKS,
                SPARK_REWEIGHT_MAX_BLOCKS,
                hierarchy=hierarchy,
            )
            ranges_host, mapping_host = validate_virtual_layout(
                topology["ranges"], topology["leaf_to_virtual"], q_bthd.shape[1]
            )
            cache[cache_key] = (
                torch.tensor(ranges_host, device=q_bthd.device, dtype=torch.int64),
                torch.tensor(mapping_host, device=q_bthd.device, dtype=torch.int64),
                dict(topology["metadata"]),
            )
        ranges, mapping, topology_metadata = cache[cache_key]
        controller.sol_virtual_query_layout = {
            **topology_metadata,
            "ablation": "identity_physical_order",
        }
        anchors = (
            None
            if needs_query_padding
            else build_virtual_anchors(q_bthd, ranges,
                dtype=(torch.float32 if cfg.sol_global_anchor_dtype == "float32" else q_bthd.dtype))
        )
        virtual_query_data = (ranges, mapping, anchors)
        controller.counts["sol_identity_reweight_calls"] += 1

    padded_context_queries = None
    if needs_query_padding:
        # Fill after any reblock permutation so the padding preserves the mean
        # of the actual video rows occupying the final physical query block.
        # K/V remain the original packed sequence, and the real context Q rows
        # are restored before dense attention below.
        tail_start = (layout.video_tokens // _SPARK_BLOCK_SIZE) * _SPARK_BLOCK_SIZE
        tail_mean = q_bthd[:, tail_start:layout.video_tokens].mean(
            dim=1, keepdim=True, dtype=torch.float32
        ).to(q_bthd.dtype)
        padded_context_queries = q_bthd[
            :, layout.video_tokens:sparse_query_tokens
        ].clone()
        q_bthd = q_bthd.clone()
        q_bthd[:, layout.video_tokens:sparse_query_tokens].copy_(tail_mean)
    dense_query_start = sink_start
    if cfg.sol_route_topk_ratio is not None:
        profile = _profile_begin(controller, "sparse_attention")
        output = _spark_topk_attention(controller, q_bthd, k_bthd, v_bthd,
                                       layout, virtual_query_data,
                                       _query_tokens=sparse_query_tokens)
        _profile_end(controller, profile)
        dense_query_start = (
            sink_start
            if needs_query_padding or legacy_full_query
            else sparse_query_tokens
        )
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
            tail_granularity=cfg.sol_tail_granularity,
            anchor_dtype=(torch.float32 if cfg.sol_global_anchor_dtype == "float32" else q_bthd.dtype),
            summary_math=cfg.sol_reweight_summary_math,
            logmass_key=cfg.sol_reweight_logmass_key,
            reweight_components=cfg.sol_reweight_components,
            _query_tokens=sparse_query_tokens)
        dense_query_start = (
            sink_start
            if needs_query_padding or legacy_full_query
            else sparse_query_tokens
        )
        controller.sol_backend = virtual_q_backend(q_bthd)
        controller.counts["sol_virtual_query_calls"] += 1
        controller.counts["sol_tau_virtual_query_calls"] += 1
    else:
        try:
            from sol_attn import get_sol_attn_backend, sol_attn
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Sol-Attn is unavailable; install NVlabs/Sana's "
                "techniques/sparse_backends package"
            ) from error
        controller.sol_backend = get_sol_attn_backend(q_bthd.device)
        output = sol_attn(q_bthd, k_bthd, v_bthd, tau=cfg.sol_tau,
                          thresh_type=cfg.sol_thresh_type, kv_splits=cfg.sol_kv_splits,
                          sink_start=sink_start, sink_tokens=sink_tokens,
                          force_local_blocks=cfg.sol_local_blocks_enabled)
        controller.counts["sol_official_calls"] += 1

    if padded_context_queries is not None:
        q_bthd[:, sink_start:sparse_query_tokens].copy_(padded_context_queries)

    # On every fixed-Top-K path, and on virtual-query tau paths, either keep the
    # mixed video/context block dense or replace its context query rows with
    # padding for sparse execution. Every real query still attends to complete
    # K/V, and every real non-video query is evaluated only by dense attention.
    # Plain Sol tau retains its stock full-query behavior and starts this dense
    # overwrite at sink_start.
    if dense_query_start < q_bthd.shape[1]:
        profile = _profile_begin(controller, "dense_suffix")
        output[:, dense_query_start:] = F.scaled_dot_product_attention(
            q_bthd[:, dense_query_start:].transpose(1, 2),
            k_bthd.transpose(1, 2),
            v_bthd.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        _profile_end(controller, profile)
        controller.counts["sol_dense_context_queries"] += 1
    if query_inverse_permutation is not None:
        profile = _profile_begin(controller, "reblock_output_inverse")
        # The permuted queries are dead after the dense suffix. Drop the final
        # local reference before allocating the equally sized inverse output so
        # the CUDA allocator can reuse that storage on memory-bound SM80 runs.
        inverse_out = None
        if (
            _sm80_low_memory_reblock(output)
        ):
            # At the 345-frame boundary another 1.39 GiB BTHD allocation does
            # not fit. K is dead after the sparse mainloop and dense suffix, so
            # reuse it as the non-aliasing destination of the inverse gather.
            inverse_out = k_bthd
        del q_bthd
        output = _headwise_permute_video_tokens(
            output,
            query_inverse_permutation,
            video_tokens=layout.video_tokens,
            out=inverse_out,
        )
        _profile_end(controller, profile)
    return output if return_bthd else output.permute(0, 2, 1, 3).contiguous()


def _effective_topk_execution(requested, capability):
    """Resolve architecture-specific execution for the public default."""
    if requested == "packed_external_no_route_qk" and capability not in (
        (8, 0),
        (8, 9),
        (12, 0),
    ):
        return "threshold"
    return requested


@torch.compiler.disable
@torch.no_grad()
def _spark_topk_attention(controller, q, k, v, layout, virtual_query_data=None, _query_tokens=None):
    """Source Spark Top-K policy with native or query-conditioned summaries."""
    from sol_attn.preprocess import _reduce_kv
    from .rope_sol_kernel import (
        sol_topk_threshold_attn, sol_topk_threshold_backend,
    )
    from .sol_vaware_compensation import exact_attention
    from .sol_topk_cutoff import (
        gemm_radix_topk_cutoff,
        gemm_topk_packed_route,
        triton_gaussian_moment_cutoff,
    )

    cfg = controller.config
    capability = tuple(torch.cuda.get_device_capability(q.device))
    execution = _effective_topk_execution(cfg.sol_route_topk_execution, capability)
    if execution != cfg.sol_route_topk_execution:
        controller.counts[
            f"route_execution_fallback:{cfg.sol_route_topk_execution}->{execution}"
        ] += 1
    if capability in ((8, 0), (8, 9)) and execution not in (
        "threshold", "fused", "packed_external_no_route_qk"
    ):
        raise RuntimeError(
            f"SM{capability[0]}{capability[1]} does not implement sol_route_topk_execution="
            f"{execution!r}; use 'threshold', 'fused', "
            "or 'packed_external_no_route_qk'"
        )
    from .sol_numerator_virtual_q import virtual_q_backend, reduce_virtual_key_centroids
    profile = _profile_begin(controller, "topk_reduce_kv")
    if (virtual_query_data is not None
            and cfg.sol_tail_granularity == "query"
            and virtual_q_backend(q).endswith("fused_virtual_query")):
        kc, vs = reduce_virtual_key_centroids(k), None
    else:
        kc, vs = _reduce_kv(k, v)
    _profile_end(controller, profile)
    sink_tokens = layout.sequence_length - layout.video_tokens
    backend = sol_topk_threshold_backend(q.device)
    partial_video = sink_tokens == 0 and layout.video_tokens // 64 < math.ceil(q.shape[1] / 64)
    threshold = route = summaries = None
    head_budget = getattr(controller, "head_topk_budget", None)
    if head_budget is not None:
        from .head_budget import route_with_head_budgets
        scores, _, kb, qb, target = _sol_topk_policy_scores(
            q, kc, video_tokens=layout.video_tokens, topk_ratio=cfg.sol_route_topk_ratio)
        route = route_with_head_budgets(scores, head_budget, kb, layout.video_tokens, sink_tokens)
        stats = dict(candidate_video_blocks=kb, query_video_blocks=qb,
            target_topk_blocks_per_query=target, route_threshold_mode="per_head_exact_topk")
    elif cfg.sol_route_global_weighted_mean:
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
            anchors = build_virtual_anchors(q, ranges,
                dtype=(torch.float32 if cfg.sol_global_anchor_dtype == "float32" else q.dtype))
        summaries = virtual_summaries(anchors, k, v,
            summary_math=cfg.sol_reweight_summary_math,
            logmass_key=cfg.sol_reweight_logmass_key,
            reweight_components=cfg.sol_reweight_components)
        route, stats = reweighted_route(q, kc, mapping, summaries[0], summaries[2],
            video_tokens=layout.video_tokens, sink_tokens=sink_tokens,
            topk_ratio=cfg.sol_route_topk_ratio, mode=cfg.sol_virtual_query_route_score)
    elif (
        backend is not None
        and not partial_video
        and execution == "fused"
        and virtual_query_data is not None
        and capability in ((8, 0), (8, 9), (12, 0))
        and _query_tokens is not None
    ):
        candidate_blocks = layout.video_tokens // 64
        stats = dict(
            block_size=64,
            blocks=math.ceil(q.shape[1] / 64),
            candidate_video_blocks=candidate_blocks,
            query_video_blocks=_query_tokens // 64,
            sink_blocks=math.ceil(sink_tokens / 64),
            target_topk_blocks_per_query=max(
                1, round(cfg.sol_route_topk_ratio * candidate_blocks)
            ),
            route_topk_ratio=cfg.sol_route_topk_ratio,
            route_threshold_mode=(
                f"sm{capability[0]}{capability[1]}_cta_local_exact_topk"
            ),
        )
        controller.counts["sol_topk_fused_route_calls"] += 1
    elif (
        backend is not None
        and not partial_video
        and execution == "packed_external_no_route_qk"
        and cfg.sol_route_topk_cutoff_mode == "gemm_radix"
        and capability in ((8, 0), (8, 9), (12, 0))
        and _query_tokens is not None
        and virtual_query_data is not None
        and virtual_q_backend(q).endswith("fused_virtual_query")
    ):
        route, stats = gemm_topk_packed_route(
            q,
            kc,
            video_tokens=layout.video_tokens,
            topk_ratio=cfg.sol_route_topk_ratio,
            query_tokens=_query_tokens,
        )
        controller.counts["sol_topk_packed_route_calls"] += 1
    elif backend is not None and not partial_video:
        profile = _profile_begin(controller, "topk_cutoff")
        cutoff = {"gemm_radix": gemm_radix_topk_cutoff,
                  "gaussian_moments": triton_gaussian_moment_cutoff}[cfg.sol_route_topk_cutoff_mode]
        cutoff_kwargs = dict(
            video_tokens=layout.video_tokens,
            sink_tokens=sink_tokens,
            topk_ratio=cfg.sol_route_topk_ratio,
        )
        if cfg.sol_route_topk_cutoff_mode == "gemm_radix":
            # Tie counts are diagnostics only.  Avoid two GPU-to-CPU .item()
            # synchronizations per layer in the production no-logging path.
            cutoff_kwargs["collect_tie_stats"] = cfg.sol_log_density
        threshold, stats = cutoff(q, kc, **cutoff_kwargs)
        _profile_end(controller, profile)
    else:
        route, stats = _sol_topk_route(q, kc, video_tokens=layout.video_tokens,
                                      sink_tokens=sink_tokens, topk_ratio=cfg.sol_route_topk_ratio)
        stats["route_threshold_mode"] = (
            "topk_explicit_partial_video_fallback" if partial_video else "topk_explicit_no_threshold_backend")

    _record_exact_block_verbose(
        controller,
        q,
        kc,
        threshold,
        video_tokens=layout.video_tokens,
        query_tokens=(
            _query_tokens
            if _query_tokens is not None
            else (layout.video_tokens // _SPARK_BLOCK_SIZE) * _SPARK_BLOCK_SIZE
        ),
        radius=cfg.sol_exact_block_radius,
    )

    if cfg.sol_log_density and controller.sol_route_density is None:
        if route is None or route.dtype == torch.int32:
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
        # The route-QK-free kernel requires an SM80/SM89/SM120 packed int32 route. Some
        # compatibility/ablation paths deliberately use a threshold or a bool
        # route instead; keep those paths functional by using the regular
        # external-route/threshold merge in that case.
        skip_external_route_qk = (
            execution == "packed_external_no_route_qk"
            and route is not None
            and route.dtype == torch.int32
        )
        if execution == "packed_external_no_route_qk" and not skip_external_route_qk:
            controller.counts[
                "route_qk_free_fallback:compatible_route_merge"
            ] += 1
        profile = _profile_begin(controller, "topk_fused_virtual")
        output = virtual_q_attention(q, k, v, virtual_ranges=ranges, leaf_to_virtual=mapping,
            virtual_anchors=anchors, precomputed_summaries=summaries, key_centroids=kc,
            value_sums=vs, threshold=threshold, route=route, sink_start=layout.video_tokens,
            sink_tokens=sink_tokens, force_local_blocks=cfg.sol_local_block_policy,
            tail_granularity=cfg.sol_tail_granularity,
            anchor_dtype=(torch.float32 if cfg.sol_global_anchor_dtype == "float32" else q.dtype),
            summary_math=cfg.sol_reweight_summary_math,
            logmass_key=cfg.sol_reweight_logmass_key,
            reweight_components=cfg.sol_reweight_components,
            _query_tokens=_query_tokens,
            fused_topk_ratio=(
                cfg.sol_route_topk_ratio
                if execution == "fused"
                else 0.0
            ),
            skip_external_route_qk=skip_external_route_qk)
        _profile_end(controller, profile)
        controller.sol_backend = virtual_q_backend(q)
        controller.counts["sol_virtual_query_calls"] += 1
    elif route is None:
        output = sol_topk_threshold_attn(q, k, v, kc, vs, threshold,
            sink_start=layout.video_tokens, sink_tokens=sink_tokens,
            force_local_blocks=cfg.sol_local_block_policy,
            _query_tokens=_query_tokens)
        controller.sol_backend = f"{backend}:{cfg.sol_route_topk_cutoff_mode}"
        controller.counts["sol_topk_threshold_calls"] += 1
    else:
        output, _, _ = exact_attention(
            q, k, v, kc, vs, route=route,
            sink_start=layout.video_tokens, sink_tokens=sink_tokens,
            force_local_blocks=cfg.sol_local_block_policy,
            _query_tokens=_query_tokens)
        output = output.to(v.dtype)
        controller.sol_backend = "vaware_exact_explicit_route"
        controller.counts["sol_topk_explicit_fallback_calls"] += 1
    controller.counts["sol_topk_calls"] += 1
    return output
