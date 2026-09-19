"""Block-mean, block-atomic hierarchical landmark routing.

Landmark-tree-v2 follows Adaptive Morton's levelwise parent batching, but uses
the existing deterministic weighted-landmark proxy tree instead of Morton
keys.  It deliberately remains separate from the production v1 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from .landmark_initial_order import InitialOrder, initial_token_indices, normalize_initial_order
from .landmark_v2_fused_node import use_fused_node as _use_fused_node
from .reblock_hierarchy import (
    ReblockHierarchy, build_reblock_hierarchy, normalize_fanout_mode,
    resolve_final_fanout, resolve_root_fanout,
)

from .landmark_tree_clustering import (
    _NodeGroup,
    _build_proxy_tree,
    _exact_tree_route,
    _initial_hilbert_partition as _initial_order_partition,
    _stable_counting_partition,
    _tree_score,
)


# Read before graph construction. Disable only to reproduce the materialized
# group-one path; changing it does not alter an already captured CUDA graph.
GROUP1_FAST_ENABLED = os.environ.get("H3_LMV2_GROUP1_FAST", "1") == "1"
# Gather the tree's own feature table as FP8 E4M3 instead of BF16.  The model
# features stay BF16; only the copy read by the landmark means, scores and
# fused node kernels is rounded, halving the dominant gather traffic.  Routing
# decisions differ from the BF16 table at near-ties only; measured blocking
# quality on real Q/K is unchanged (captured attention mass, coherence).
FP8_FEATURES_ENABLED = os.environ.get("H3_LMV2_FP8_FEATURES", "0") == "1"


def _use_group1_fastpath(source, group_size, distance, aggregation):
    if not GROUP1_FAST_ENABLED or group_size != 1 or not source.is_cuda:
        return False
    if not 0 < source.shape[-1] <= 256:
        return False
    if distance == "euclidean":
        return True
    if source.shape[-1] != 128:
        return False
    if aggregation == "max":
        return True
    from .landmark_v2_cosine import FAST_ENABLED
    return FAST_ENABLED


def _group1_indexed_scores(source, indices, centers, weights, capacities, distance, aggregation,
                           fp8_source=None):
    if distance == "euclidean":
        from .landmark_v2_euclidean import euclidean_proxy_scores
        return euclidean_proxy_scores(source, centers, weights, capacities, indices=indices)
    from .landmark_v2_cosine_fast import build_cosine_directions, fused_cosine_scores_indexed
    if aggregation == "max":
        from .landmark_v2_max import max_support_scores_indexed
        _, normalized, active = build_cosine_directions(
            centers, weights, capacities, return_partition=True)
        return max_support_scores_indexed(source, indices, normalized, active)
    from .landmark_v2_cosine import FAST_PRECISION
    directions = build_cosine_directions(centers, weights, capacities)
    if fp8_source is not None:
        return fused_cosine_scores_indexed(fp8_source, indices, directions, FAST_PRECISION, fp8=True)
    return fused_cosine_scores_indexed(source, indices, directions, FAST_PRECISION)


@dataclass(frozen=True)
class LandmarkTreeV2SplitStats:
    level: int
    node_tokens: int
    node_count: int
    target_leaves: int
    group_size: int
    representatives: int
    children: int
    landmarks: int


@dataclass(frozen=True)
class LandmarkTreeV2Result:
    block_indices: torch.Tensor
    excluded_indices: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    active_tokens: int
    num_excluded: int
    split_stats: tuple[LandmarkTreeV2SplitStats, ...]
    hierarchy: ReblockHierarchy


def _torch_indexed_group_mean(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    parents, tokens = global_indices.shape
    dim = source.shape[1]
    gathered = source.index_select(0, global_indices.reshape(-1))
    return (
        gathered.reshape(parents, tokens // group_size, group_size, dim)
        .float()
        .mean(2)
        .to(source.dtype)
    )


def _torch_indexed_group_means_8_and_4(
    source: torch.Tensor,
    global_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load each root group once while preserving direct 8/4 reductions."""

    parents, tokens = global_indices.shape
    if tokens % 8:
        raise ValueError("the token count must be divisible by 8")
    dim = source.shape[1]
    gathered = source.index_select(0, global_indices.reshape(-1)).reshape(
        parents, tokens // 8, 8, dim
    )
    coarse = gathered.float().mean(2).to(source.dtype)
    fine = (
        gathered.reshape(parents, tokens // 8, 2, 4, dim)
        .float()
        .mean(3)
        .reshape(parents, tokens // 4, dim)
        .to(source.dtype)
    )
    return coarse, fine


def _torch_interval_means(
    representatives: torch.Tensor,
    landmarks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    parents, groups, _ = representatives.shape
    boundaries = [index * groups // landmarks for index in range(landmarks + 1)]
    centers = torch.stack(
        [
            representatives[:, boundaries[i] : boundaries[i + 1]]
            .float()
            .mean(1)
            for i in range(landmarks)
        ],
        dim=1,
    ).to(representatives.dtype)
    weights = torch.tensor(
        [boundaries[i + 1] - boundaries[i] for i in range(landmarks)],
        device=representatives.device,
        dtype=torch.int32,
    ).expand(parents, -1)
    return centers, weights


def _normalize_landmark_mode(mode: str) -> str:
    if mode not in ("mean", "midpoint"):
        raise ValueError("landmark_mode must be mean or midpoint")
    return mode


def _torch_interval_landmarks(representatives, landmarks, mode):
    if mode == "mean":
        return _torch_interval_means(representatives, landmarks)
    from .landmark_tree_clustering import _midpoint_positions
    parents, groups, _ = representatives.shape
    positions = _midpoint_positions(groups, landmarks, representatives.device)
    boundaries = torch.arange(landmarks + 1, device=representatives.device) * groups // landmarks
    weights = (boundaries[1:] - boundaries[:-1]).to(torch.int32).expand(parents, -1)
    return representatives.index_select(1, positions).contiguous(), weights


def _block_mean_landmarks(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    group_size: int,
    *,
    optimized_means: bool,
    landmark_mode: str,
    landmark_count: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if optimized_means and source.is_cuda:
        from .landmark_tree_v2_triton import (
            contiguous_interval_means,
            indexed_group_mean,
        )

        representatives = indexed_group_mean(source, global_indices, group_size)
    else:
        representatives = _torch_indexed_group_mean(
            source, global_indices, group_size
        )
    landmarks = (32 if representatives.shape[1] >= 32 else 16) if landmark_count == 32 else min(landmark_count, representatives.shape[1])
    if optimized_means and representatives.is_cuda:
        centers, weights = contiguous_interval_means(representatives, landmarks, midpoint=landmark_mode == "midpoint")
    else:
        centers, weights = _torch_interval_landmarks(representatives, landmarks, landmark_mode)
    return representatives, centers, weights


def _root_block_mean_landmarks_and_atoms(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    optimized_means: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if optimized_means and source.is_cuda:
        from .landmark_tree_v2_triton import (
            contiguous_interval_means,
            indexed_group_means_8_and_4,
        )

        representatives, atoms = indexed_group_means_8_and_4(
            source, global_indices
        )
        landmarks = 32 if representatives.shape[1] >= 32 else 16
        centers, weights = contiguous_interval_means(representatives, landmarks)
    else:
        representatives, atoms = _torch_indexed_group_means_8_and_4(
            source, global_indices
        )
        landmarks = 32 if representatives.shape[1] >= 32 else 16
        centers, weights = _torch_interval_means(representatives, landmarks)
    return representatives, centers, weights, atoms


@torch.no_grad()
def _normalize_children(children):
    """Freeze a per-level fanout schedule; a scalar applies at every level."""
    if type(children) is int and children in (2, 4, 8, 16, 32):
        return children
    if isinstance(children, (list, tuple)) and children:
        values = tuple(children)
        if all(type(value) is int and value in (2, 4, 8, 16, 32) for value in values):
            return values
    raise ValueError("children must be 2, 4, 8, 16 or 32, or a nonempty list/tuple of those integers")


def _normalize_fanout(max_children, fanout):
    if fanout is not None:
        if max_children != 16 and _normalize_children(max_children) != _normalize_children(fanout):
            raise ValueError("use fanout or legacy max_children, not conflicting values")
        max_children = fanout
    return _normalize_children(max_children)


def _normalize_order_mode(order_mode):
    """Use the canonical name while accepting historical launch configurations."""
    if order_mode == "preserve_parent_order":
        return "parent_order"
    if order_mode not in ("parent_order", "scalar_order"):
        raise ValueError("order_mode must be parent_order or scalar_order")
    return order_mode


def _normalize_group_size(group_size):
    """Freeze the token-group schedule; repeat its last value at deeper levels."""
    if type(group_size) is int and group_size in (1, 2, 4, 8):
        return group_size
    if isinstance(group_size, (list, tuple)) and group_size:
        values = tuple(group_size)
        if all(type(value) is int and value in (1, 2, 4, 8) for value in values):
            return values
    raise ValueError("group_size must be 1, 2, 4 or 8, or a nonempty list/tuple of those integers")


def _recursive_landmark_tree_v2(
    samples: torch.Tensor,
    *,
    grid_shape: tuple[int, int, int],
    initial_indices: torch.Tensor,
    validate: bool,
    optimized_means: bool,
    reuse_group4: bool,
    distance: str = "cosine",
    order_mode: str = "parent_order",
    group_size: int | list[int] | tuple[int, ...] = 1,
    fitting_samples: torch.Tensor | None = None,
    max_children: int | list[int] | tuple[int, ...] = 16,
    fanout_mode: str = "power_of_two_fanout",
    fanout: int | list[int] | tuple[int, ...] | None = None,
    final_fanout: int | list[int] | tuple[int, ...] | None = None,
    root_fanout: int | None = None,
    landmark_mode: str = "midpoint",
    landmark_count: int = 32,
    aggregation: str = "linear",
    minimum_frames: int | None = None,
) -> LandmarkTreeV2Result:
    max_children = _normalize_fanout(max_children, fanout)
    fanout_mode = normalize_fanout_mode(fanout_mode)
    landmark_mode = _normalize_landmark_mode(landmark_mode)
    if landmark_count not in (32, 128, 256):
        raise ValueError("landmark_count must be 32, 128, or 256")
    reuse_group4 = reuse_group4 and landmark_mode == "mean" and landmark_count == 32
    children_schedule = (max_children,) if isinstance(max_children, int) else max_children
    if aggregation not in ("linear", "max") or (aggregation == "max" and distance != "cosine"):
        raise ValueError("max aggregation requires cosine distance")
    group_size = _normalize_group_size(group_size)
    group_sizes = (group_size,) if isinstance(group_size, int) else group_size
    order_mode = _normalize_order_mode(order_mode)
    if distance not in ("euclidean", "cosine"):
        raise ValueError("v2 distance must be euclidean or cosine")
    if samples.ndim < 2 or not samples.is_floating_point():
        raise ValueError("samples must be floating point with layout [...,N,D]")
    leading = samples.shape[:-2]
    tokens, dim = samples.shape[-2:]
    if tokens < 64 or dim <= 0:
        raise ValueError("landmark-tree-v2 requires at least 64 non-empty rows")
    if samples.is_cuda and samples.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA landmark-tree-v2 requires FP16 or BF16 input")
    flat = samples.reshape(-1, tokens, dim).contiguous()
    batch = flat.shape[0]
    active_root, excluded = _initial_order_partition(
        flat, grid_shape, initial_indices, validate_order=validate
    )
    num_excluded = tokens % 64
    active_tokens = tokens - num_excluded
    active_leaves = active_tokens // 64
    permutation = torch.empty((batch, tokens), device=samples.device, dtype=torch.long)
    blocks = permutation[:, :active_tokens].view(batch, active_leaves, 64)
    rows = torch.arange(batch, device=samples.device, dtype=torch.long)
    frontier = [
        _NodeGroup(
            indices=active_root,
            rows=rows,
            leaf_starts=torch.zeros_like(rows),
            leaf_budget=active_leaves,
            features=(
                torch.arange(
                    batch * (active_tokens // 4),
                    device=samples.device,
                    dtype=torch.long,
                ).reshape(batch, active_tokens // 4)
                if reuse_group4 and active_leaves > 1
                else None
            ),
        )
    ]
    hierarchy = build_reblock_hierarchy(tokens, tuple(children_schedule), grid_shape=tuple(grid_shape),
        minimum_frames=int(os.environ.get("H3_TEMPORAL_MIN_FRAMES", "0")) if minimum_frames is None else minimum_frames,
        final_fanout=final_fanout, root_fanout=root_fanout, fanout_mode=fanout_mode)
    if hierarchy.minimum_frames:
        if num_excluded or reuse_group4 or any(g != 1 for g in group_sizes):
            raise ValueError("Temporal roots require complete video blocks and group size 1")
        frontier = []
        for start, end in hierarchy.roots:
            if end-start == 1:
                blocks[:,start] = active_root[:,start*64:end*64]
            else:
                frontier.append(_NodeGroup(
                    indices=active_root[:,start*64:end*64].contiguous(), rows=rows,
                    leaf_starts=torch.full_like(rows,start), leaf_budget=end-start,
                    features=None))
    if active_leaves == 1:
        blocks[:, 0] = active_root
        frontier = []
    stats: list[LandmarkTreeV2SplitStats] = []
    level = 0
    if fitting_samples is not None:
        if fitting_samples.shape != samples.shape or fitting_samples.dtype != samples.dtype:
            raise ValueError("fitting samples must match raw transformed shape/dtype")
        source = fitting_samples.reshape(batch * tokens, dim).contiguous()
    else:
        source = flat.reshape(batch * tokens, dim)
    atom_source: torch.Tensor | None = None
    fp8_source: torch.Tensor | None = None
    if (FP8_FEATURES_ENABLED and optimized_means and source.is_cuda and dim == 128
            and source.dtype == torch.bfloat16 and distance == "cosine"
            and aggregation == "linear" and order_mode == "parent_order"
            and all(value == 1 for value in group_sizes) and not validate
            and _use_group1_fastpath(source, 1, distance, aggregation)):
        from .landmark_tree_v2_triton import fp8_feature_table
        fp8_source = fp8_feature_table(source)

    while frontier:
        grouped: dict[tuple[int, int], list[_NodeGroup]] = {}
        for fragment in frontier:
            grouped.setdefault(
                (fragment.indices.shape[1], fragment.leaf_budget), []
            ).append(fragment)
        next_frontier: list[_NodeGroup] = []
        for (node_tokens, leaf_budget), fragments in grouped.items():
            if len(fragments) == 1:
                group = fragments[0]
            else:
                group = _NodeGroup(
                    indices=torch.cat([item.indices for item in fragments]),
                    rows=torch.cat([item.rows for item in fragments]),
                    leaf_starts=torch.cat([item.leaf_starts for item in fragments]),
                    leaf_budget=leaf_budget,
                    features=(
                        torch.cat(
                            [item.features for item in fragments if item.features is not None]
                        )
                        if all(item.features is not None for item in fragments)
                        else None
                    ),
                )
            node_count = group.indices.shape[0]
            group_size = group_sizes[min(level, len(group_sizes) - 1)]
            if node_tokens % group_size:
                raise RuntimeError("v2 stage group size does not divide node tokens")
            global_indices = (
                group.indices + group.rows[:, None] * tokens
            ).contiguous()
            child_budgets = hierarchy.budgets(level, leaf_budget)
            children = len(child_budgets)
            child_token_capacities = tuple(64 * value for value in child_budgets)
            child_group_capacities = tuple(
                value // group_size for value in child_token_capacities
            )
            direct_partition = (group_size == 1 and order_mode == "parent_order"
                                and optimized_means and samples.is_cuda and not validate)
            indexed_group1 = optimized_means and _use_group1_fastpath(
                source, group_size, distance, aggregation)
            node_landmarks = min(landmark_count, node_tokens // group_size)
            final_round = all(budget == 1 for budget in child_budgets)
            split_mode = ("arbitrary_fanout" if final_round and fanout_mode != "power_of_two_fanout"
                          else fanout_mode)
            fused_node = (node_landmarks <= 128 and (node_landmarks & (node_landmarks - 1)) == 0 and indexed_group1 and direct_partition and distance == "cosine"
                          and aggregation == "linear"
                          and _use_fused_node(node_tokens, children, dim))
            if fused_node:
                # One program per node: means, proxy tree, scores, exact route
                # and stable partition, reading the node features twice.
                from .landmark_v2_fused_node import fused_node_split
                from .landmark_v2_cosine import FAST_PRECISION
                mapped = fused_node_split(
                    source if fp8_source is None else fp8_source,
                    global_indices, group.rows, tokens, child_group_capacities,
                    mode=FAST_PRECISION, fp8=fp8_source is not None,
                    midpoint=landmark_mode == "midpoint", landmarks=node_landmarks)
                landmarks_used = node_landmarks
            elif (split_mode == "arbitrary_fanout" and indexed_group1
                  and distance == "cosine" and aggregation == "linear"):
                from .landmark_tree_v2_triton import indexed_interval_means
                from .landmark_v2_terminal import partition_scores
                node_source = source if fp8_source is None else fp8_source
                centers, weights = indexed_interval_means(
                    node_source, global_indices, node_landmarks,
                    fp8=fp8_source is not None, center_dtype=source.dtype,
                    midpoint=landmark_mode == "midpoint")
                scores = _group1_indexed_scores(
                    source, global_indices, centers, weights, child_group_capacities,
                    distance, aggregation, fp8_source=fp8_source)
                if order_mode == "parent_order":
                    mapped = partition_scores(
                        scores, group.indices, child_group_capacities,
                        max_original_index=tokens - 1, validate=validate)
                else:
                    from .landmark_v2_order import scalar_tree_order
                    order = scalar_tree_order(scores, group.indices, child_group_capacities)
                    mapped = group.indices.gather(1, order)
                landmarks_used = centers.shape[1]
            elif split_mode == "arbitrary_fanout" and (final_round or children & (children-1)):
                from .landmark_v2_terminal import node_split_reference
                representatives, centers, weights = _block_mean_landmarks(
                    source, global_indices, group_size, optimized_means=optimized_means,
                    landmark_mode=landmark_mode, landmark_count=landmark_count)
                original_groups = group.indices.reshape(node_count,-1,group_size)[:,:,0]
                ordered_ids = node_split_reference(
                    representatives, original_groups, centers, weights, child_group_capacities,
                    distance=distance, aggregation=aggregation, order_mode=order_mode)
                sorted_ids, positions = original_groups.sort(dim=1)
                group_order = positions.gather(1,torch.searchsorted(sorted_ids.contiguous(),ordered_ids.contiguous()))
                mapped = group.indices.reshape(node_count,-1,group_size).gather(
                    1,group_order[:,:,None].expand(-1,-1,group_size)).reshape(node_count,node_tokens)
                landmarks_used = centers.shape[1]
            else:
                if indexed_group1:
                    from .landmark_tree_v2_triton import indexed_interval_means
                    landmarks = node_landmarks
                    if fp8_source is not None:
                        centers, weights = indexed_interval_means(
                            fp8_source, global_indices, landmarks, fp8=True,
                            center_dtype=source.dtype, midpoint=landmark_mode == "midpoint")
                    else:
                        centers, weights = indexed_interval_means(source, global_indices, landmarks, midpoint=landmark_mode == "midpoint")
                elif reuse_group4 and level == 0:
                    representatives, centers, weights, atoms = (
                        _root_block_mean_landmarks_and_atoms(
                            source,
                            global_indices,
                            optimized_means=optimized_means,
                        )
                    )
                    atom_source = atoms.reshape(-1, dim)
                elif reuse_group4:
                    if atom_source is None or group.features is None:
                        raise RuntimeError("missing cached group-4 representatives")
                    representatives, centers, weights = _block_mean_landmarks(
                        atom_source,
                        group.features.contiguous(),
                        1,
                        optimized_means=optimized_means,
                        landmark_mode=landmark_mode, landmark_count=landmark_count,
                    )
                else:
                    representatives, centers, weights = _block_mean_landmarks(
                        source,
                        global_indices,
                        group_size,
                        optimized_means=optimized_means,
                        landmark_mode=landmark_mode, landmark_count=landmark_count,
                    )
                if indexed_group1:
                    scores = _group1_indexed_scores(
                        source, global_indices, centers, weights, child_group_capacities,
                        distance, aggregation, fp8_source=fp8_source)
                elif aggregation == "max":
                    from .landmark_v2_max import cosine_max_proxy_scores, cosine_max_proxy_scores_reference
                    score_fn = cosine_max_proxy_scores if optimized_means else cosine_max_proxy_scores_reference
                    scores = score_fn(representatives, centers, weights, child_group_capacities)
                elif distance == "cosine":
                    from .landmark_v2_cosine import cosine_proxy_scores, cosine_proxy_scores_reference
                    score_fn = cosine_proxy_scores if optimized_means else cosine_proxy_scores_reference
                    scores = score_fn(representatives, centers, weights, child_group_capacities)
                elif optimized_means and centers.is_cuda:
                    from .landmark_v2_euclidean import euclidean_proxy_scores
                    scores = euclidean_proxy_scores(
                        representatives, centers, weights, child_group_capacities)
                else:
                    alpha, bias = _build_proxy_tree(
                        centers, weights, child_group_capacities, proxy_iterations=2
                    )
                    scores = _tree_score(representatives, centers, alpha, bias)
                token_groups = group.indices.reshape(
                    node_count, node_tokens // group_size, group_size
                )
                tie_indices = token_groups[:, :, 0].contiguous()
                if order_mode == "parent_order":
                    labels = _exact_tree_route(
                        scores,
                        tie_indices,
                        child_group_capacities,
                        max_original_index=tokens - 1,
                        validate=validate,
                        compact_selection=True,
                    )
                    group_order = _stable_counting_partition(
                        labels,
                        child_group_capacities,
                        validate=validate,
                        source_indices=group.indices if direct_partition else None,
                    )
                else:
                    from .landmark_v2_order import scalar_tree_order
                    group_order = scalar_tree_order(scores, tie_indices, child_group_capacities)
                assert isinstance(group_order, torch.Tensor)
                if direct_partition:
                    # The production partition kernel can emit original token IDs;
                    # avoid writing local positions and gathering those IDs again.
                    mapped = group_order
                else:
                    reordered_groups = torch.gather(
                        token_groups, 1,
                        group_order[:, :, None].expand(-1, -1, group_size),
                    )
                    mapped = reordered_groups.reshape(node_count, node_tokens)
                landmarks_used = centers.shape[1]
            mapped_features = None
            if reuse_group4:
                if group.features is None:
                    raise RuntimeError("missing group-4 IDs during partition")
                atoms_per_group = group_size // 4
                atom_groups = group.features.reshape(
                    node_count, representatives.shape[1], atoms_per_group
                )
                mapped_features = torch.gather(
                    atom_groups,
                    1,
                    group_order[:, :, None].expand(-1, -1, atoms_per_group),
                ).reshape(node_count, node_tokens // 4)
            stats.append(
                LandmarkTreeV2SplitStats(
                    level=level,
                    node_tokens=node_tokens,
                    node_count=node_count,
                    target_leaves=leaf_budget,
                    group_size=group_size,
                    representatives=node_tokens // group_size,
                    children=children,
                    landmarks=landmarks_used,
                )
            )
            token_offset = 0
            atom_offset = 0
            leaf_offset = 0
            for capacity, budget in zip(child_token_capacities, child_budgets):
                child_indices = mapped[:, token_offset : token_offset + capacity]
                child_starts = group.leaf_starts + leaf_offset
                if budget == 1:
                    blocks[group.rows, child_starts] = child_indices
                else:
                    next_frontier.append(
                        _NodeGroup(
                            indices=child_indices,
                            rows=group.rows,
                            leaf_starts=child_starts,
                            leaf_budget=budget,
                            features=(
                                mapped_features[
                                    :, atom_offset : atom_offset + capacity // 4
                                ]
                                if mapped_features is not None
                                else None
                            ),
                        )
                    )
                token_offset += capacity
                atom_offset += capacity // 4
                leaf_offset += budget
        frontier = next_frontier
        level += 1

    if num_excluded:
        permutation[:, active_tokens:].copy_(excluded)
    inverse = torch.empty_like(permutation)
    inverse.scatter_(
        1,
        permutation,
        torch.arange(tokens, device=samples.device).expand(batch, -1),
    )
    if validate:
        expected = torch.arange(tokens, device=samples.device).expand(batch, -1)
        if not torch.equal(permutation.sort(1).values, expected):
            raise RuntimeError("landmark-tree-v2 result is not a permutation")
    token_shape = (*leading, tokens)
    return LandmarkTreeV2Result(
        block_indices=blocks.reshape(*leading, active_leaves, 64),
        excluded_indices=excluded.reshape(*leading, num_excluded),
        permutation=permutation.reshape(token_shape),
        inverse_permutation=inverse.reshape(token_shape),
        active_tokens=active_tokens,
        num_excluded=num_excluded,
        split_stats=tuple(stats),
        hierarchy=hierarchy,
    )


def recursive_landmark_tree_v2_reference(
    samples: torch.Tensor,
    *,
    grid_shape: tuple[int, int, int],
    initial_order: InitialOrder = "flat",
    validate: bool = True,
    distance: str = "cosine",
    order_mode: str = "parent_order",
    group_size: int | list[int] | tuple[int, ...] = 1,
    fitting_samples: torch.Tensor | None = None,
    max_children: int | list[int] | tuple[int, ...] = 16,
    fanout_mode: str = "power_of_two_fanout",
    fanout: int | list[int] | tuple[int, ...] | None = None,
    final_fanout: int | list[int] | tuple[int, ...] | None = None,
    root_fanout: int | None = None,
    landmark_mode: str = "midpoint",
    landmark_count: int = 32,
    aggregation: str = "linear",
    minimum_frames: int | None = None,
) -> LandmarkTreeV2Result:
    """Direct PyTorch mean implementation used as the v2 correctness oracle."""

    return _recursive_landmark_tree_v2(
        samples,
        grid_shape=grid_shape,
        initial_indices=initial_token_indices(grid_shape, initial_order, device=samples.device),
        validate=validate,
        optimized_means=False,
        reuse_group4=False,
        group_size=group_size,
        fitting_samples=fitting_samples,
        max_children=max_children,
        fanout_mode=fanout_mode, fanout=fanout, final_fanout=final_fanout, root_fanout=root_fanout,
        landmark_mode=landmark_mode, landmark_count=landmark_count,
        aggregation=aggregation,
        minimum_frames=minimum_frames,
        distance=distance,
        order_mode=order_mode,
    )


def recursive_landmark_tree_v2_blocks(
    samples: torch.Tensor,
    *,
    grid_shape: tuple[int, int, int],
    initial_order: InitialOrder = "flat",
    validate: bool = True,
    distance: str = "cosine",
    order_mode: str = "parent_order",
    group_size: int | list[int] | tuple[int, ...] = 1,
    fitting_samples: torch.Tensor | None = None,
    max_children: int | list[int] | tuple[int, ...] = 16,
    fanout_mode: str = "power_of_two_fanout",
    fanout: int | list[int] | tuple[int, ...] | None = None,
    final_fanout: int | list[int] | tuple[int, ...] | None = None,
    root_fanout: int | None = None,
    landmark_mode: str = "midpoint",
    landmark_count: int = 32,
    aggregation: str = "linear",
    minimum_frames: int | None = None,
) -> LandmarkTreeV2Result:
    """Return strict 64-token leaves; defaults are single tokens, eight children and midpoint landmarks.

    group_size (1/2/4/8) and max_children (2/4/8/16/32) each accept a scalar or
    per-level list/tuple. Defaults are group_size=1, max_children=16 and landmark_mode="midpoint"; levels
    beyond a sequence's length reuse its last value. The schedules are independent.
    fanout_mode="power_of_two_fanout" selects maximum-first power-of-two
    scheduling for all rounds. "arbitrary_fanout" selects
    the minimum-depth balanced scheduler and early-stopping arbitrary-count
    splitter. "power_of_two_arbitrary_final" keeps power-of-two nonfinal
    rounds but finishes small nodes with arbitrary fanout.
    fanout (legacy alias: max_children) bounds nonfinal rounds;
    root_fanout bounds the first nonfinal round; final_fanout bounds the
    final round in the arbitrary modes. Both default to None, inheriting fanout.
    Strict power-of-two requires final_fanout to inherit or equal fanout.
    In the arbitrary modes a one-round tree uses final_fanout. For example,
    fanout=8, final_fanout=16 permits up to 16 final children.
    fanout=8, final_fanout=8 uses a strict eight-child bound throughout.
    parent_order preserves the relative order inherited from each parent.
    distance defaults to cosine; squared Euclidean remains explicitly selectable.
    initial_order selects root indices into original raster THW samples;
    it never physically reorders the samples tensor.
    """

    group_size = _normalize_group_size(group_size)
    return _recursive_landmark_tree_v2(
        samples,
        grid_shape=grid_shape,
        initial_indices=initial_token_indices(grid_shape, initial_order, device=samples.device),
        validate=validate,
        optimized_means=True,
        reuse_group4=(group_size == (8, 4)),
        group_size=group_size,
        fitting_samples=fitting_samples,
        max_children=max_children,
        fanout_mode=fanout_mode, fanout=fanout, final_fanout=final_fanout, root_fanout=root_fanout,
        landmark_mode=landmark_mode, landmark_count=landmark_count,
        aggregation=aggregation,
        minimum_frames=minimum_frames,
        distance=distance,
        order_mode=order_mode,
    )


class PreparedLandmarkTreeV2Permutation:
    """Reusable CUDA-graph plan for a fixed v2 paired input shape."""

    def __init__(
        self,
        *,
        batch: int,
        tokens: int,
        dim: int,
        grid_shape: tuple[int, int, int],
        device: torch.device,
        initial_order: InitialOrder = "flat",
        distance: str = "cosine",
        order_mode: str = "parent_order",
        group_size: int | list[int] | tuple[int, ...] = 1,
        input_unit_means: bool = False,
        metric_unit_means: bool = False,
        max_children: int | list[int] | tuple[int, ...] = 16,
        fanout_mode: str = "power_of_two_fanout",
        fanout: int | list[int] | tuple[int, ...] | None = None,
        final_fanout: int | list[int] | tuple[int, ...] | None = None,
        root_fanout: int | None = None,
        landmark_mode: str = "midpoint",
    landmark_count: int = 32,
        aggregation: str = "linear",
        minimum_frames: int | None = None,
        chunk_frames: int | None = None,
    ) -> None:
        if input_unit_means and metric_unit_means:
            raise ValueError("select only one unit-mean space")
        self.minimum_frames = int(os.environ.get("H3_TEMPORAL_MIN_FRAMES", "0")) if minimum_frames is None else minimum_frames
        if type(self.minimum_frames) is not int or self.minimum_frames < 0:
            raise ValueError("minimum_frames must be a nonnegative integer")
        self.chunk_frames = int(os.environ.get("H3_TEMPORAL_CHUNK_FRAMES", "0")) if chunk_frames is None else chunk_frames
        if type(self.chunk_frames) is not int or self.chunk_frames < 0:
            raise ValueError("chunk_frames must be a nonnegative integer")
        if self.chunk_frames and self.minimum_frames:
            raise ValueError("chunk and minimum temporal splitting are mutually exclusive")
        self.max_children = _normalize_fanout(max_children, fanout)
        self.fanout = self.max_children
        self.fanout_mode = normalize_fanout_mode(fanout_mode)
        self.root_fanout = resolve_root_fanout(self.max_children, root_fanout)
        self.final_fanout = resolve_final_fanout(self.max_children, final_fanout, fanout_mode=self.fanout_mode)
        self.landmark_mode = _normalize_landmark_mode(landmark_mode)
        if landmark_count not in (32, 128, 256):
            raise ValueError("landmark_count must be 32, 128, or 256")
        self.landmark_count = landmark_count
        self.aggregation = aggregation
        self.metric_unit_means = metric_unit_means
        self.input_unit_means = input_unit_means
        self._static_inverse_norms: torch.Tensor | None = None
        self.group_size = _normalize_group_size(group_size)
        self.order_mode = _normalize_order_mode(order_mode)
        if distance not in ("euclidean", "cosine"):
            raise ValueError("v2 distance must be euclidean or cosine")
        self.distance = distance
        self.batch = int(batch)
        self.tokens = int(tokens)
        self.dim = int(dim)
        self.grid_shape = grid_shape
        self.initial_order = normalize_initial_order(initial_order)
        self.device = torch.device(device)
        self._initial_indices = initial_token_indices(grid_shape, self.initial_order, device=self.device)
        if self._initial_indices.numel() != self.tokens:
            raise ValueError("grid_shape product must equal the token count")
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_failed = False
        self._calls = 0
        self._static_input: torch.Tensor | None = None
        self._permutation: torch.Tensor | None = None
        self._inverse: torch.Tensor | None = None
        self.split_count = 0
        self.hierarchy = None

    def _compute(self, samples: torch.Tensor, inverse_norms=None) -> tuple[torch.Tensor, torch.Tensor]:
        fitting = None
        if self.input_unit_means:
            if inverse_norms is None:
                raise ValueError("input-unit means require original-token inverse norms")
            fitting = (samples.float() * inverse_norms).to(samples.dtype)
        elif self.metric_unit_means:
            x = samples.float()
            fitting = (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(samples.dtype)
        chunk_frames = self.chunk_frames
        if chunk_frames:
            if self.minimum_frames:
                raise ValueError("chunk and minimum temporal splitting are mutually exclusive")
            if self.initial_order != "flat":
                raise ValueError("chunk splitting requires flat initial order")
            from .landmark_chunk import chunk_permutation
            permutation, inverse, stats, metadata = chunk_permutation(
                samples, chunk_frames=chunk_frames, grid_shape=self.grid_shape,
                fitting_samples=fitting, validate=False, optimized_means=True,
                reuse_group4=False, distance=self.distance, order_mode=self.order_mode,
                group_size=self.group_size, max_children=self.max_children,
                fanout_mode=self.fanout_mode, final_fanout=self.final_fanout, root_fanout=self.root_fanout,
                landmark_mode=self.landmark_mode, landmark_count=self.landmark_count,
                aggregation=self.aggregation, minimum_frames=0)
            self.split_count = len(stats)
            self.hierarchy = None
            self.chunk_metadata = metadata
            return permutation, inverse
        result = _recursive_landmark_tree_v2(
            samples,
            grid_shape=self.grid_shape,
            initial_indices=self._initial_indices,
            validate=False,
            optimized_means=True,
            reuse_group4=(self.group_size == (8, 4)),
            distance=self.distance,
            order_mode=self.order_mode,
            group_size=self.group_size,
            fitting_samples=fitting,
            max_children=self.max_children,
            fanout_mode=self.fanout_mode, final_fanout=self.final_fanout, root_fanout=self.root_fanout,
            landmark_mode=self.landmark_mode, landmark_count=self.landmark_count,
            aggregation=self.aggregation,
            minimum_frames=self.minimum_frames,
        )
        self.split_count = len(result.split_stats)
        self.hierarchy = result.hierarchy
        return result.permutation, result.inverse_permutation

    @property
    def graph_active(self) -> bool:
        return self._graph is not None

    @property
    def graph_input(self) -> torch.Tensor | None:
        return self._static_input if self._graph is not None else None

    @property
    def graph_inverse_norms(self) -> torch.Tensor | None:
        return self._static_inverse_norms if self._graph is not None else None

    def replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._graph is None:
            raise RuntimeError("v2 replay requires an active CUDA graph")
        self._graph.replay()
        assert self._permutation is not None and self._inverse is not None
        return self._permutation, self._inverse

    @torch.no_grad()
    def run(self, samples: torch.Tensor, inverse_norms=None) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.batch, self.tokens, self.dim)
        if samples.shape != expected or samples.dtype != torch.bfloat16:
            raise ValueError(
                f"prepared v2 input must be BF16 {expected}, got "
                f"{tuple(samples.shape)} {samples.dtype}"
            )
        if self.input_unit_means:
            if inverse_norms is None or inverse_norms.shape != (self.batch, self.tokens, 1) or inverse_norms.dtype != torch.float32:
                raise ValueError("input-unit means require FP32 [batch,tokens,1] inverse norms")
        elif inverse_norms is not None:
            raise ValueError("raw means do not accept inverse norms")
        if self._graph is not None:
            assert self._static_input is not None
            if self.input_unit_means:
                self._static_inverse_norms.copy_(inverse_norms)
            self._static_input.copy_(samples)
            return self.replay()
        if self._graph_failed or not samples.is_cuda:
            return self._compute(samples, inverse_norms)
        self._calls += 1
        if self._calls == 1:
            return self._compute(samples, inverse_norms)
        try:
            torch.cuda.synchronize()
            self._static_input = samples.detach().clone()
            self._static_inverse_norms = inverse_norms.detach().clone() if inverse_norms is not None else None
            warmup_stream = torch.cuda.Stream(device=samples.device)
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                self._compute(self._static_input, self._static_inverse_norms)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._permutation, self._inverse = self._compute(self._static_input, self._static_inverse_norms)
            self._graph = graph
        except Exception:
            self._graph = None
            self._graph_failed = True
            self._static_input = None
            self._static_inverse_norms = None
            self._permutation = None
            self._inverse = None
            torch.cuda.synchronize()
        if self._graph is not None:
            return self.replay()
        return self._compute(samples, inverse_norms)


__all__ = [
    "LandmarkTreeV2Result",
    "LandmarkTreeV2SplitStats",
    "PreparedLandmarkTreeV2Permutation",
    "recursive_landmark_tree_v2_blocks",
    "recursive_landmark_tree_v2_reference",
]
