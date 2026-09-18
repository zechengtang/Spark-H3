"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math


from dataclasses import dataclass


import torch


from .mahalanobis_kmeans import hilbert_order_3d


_PROXY_STATIC_CACHE: dict[
    tuple[torch.device, int, tuple[int, ...]],
    tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]],
] = {}


_PARTITION_OFFSET_CACHE: dict[
    tuple[torch.device, tuple[int, ...]], torch.Tensor
] = {}


_ROUTE_GROUP_CACHE: dict[
    tuple[torch.device, tuple[int, ...]],
    tuple[tuple[tuple[int, torch.Tensor, torch.Tensor], ...], ...],
] = {}


_COMPACT_ROUTE_CACHE: dict[
    tuple[torch.device, tuple[int, ...]],
    tuple[tuple[torch.Tensor | None, tuple[tuple[int, int, int, int, int], ...]], ...],
] = {}


@dataclass
class _NodeGroup:
    indices: torch.Tensor
    rows: torch.Tensor
    leaf_starts: torch.Tensor
    leaf_budget: int
    features: torch.Tensor | None = None


def _midpoint_positions(tokens: int, landmarks: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(landmarks, device=device, dtype=torch.long)
    start = torch.div(index * tokens, landmarks, rounding_mode="floor")
    end = torch.div((index + 1) * tokens, landmarks, rounding_mode="floor")
    return start + torch.div(end - start, 2, rounding_mode="floor")


def _largest_norm_remainder(
    samples: torch.Tensor, count: int, *, validate: bool = True
) -> torch.Tensor:
    """Select largest squared norms; equal norms use the lower original index."""

    batch, tokens, _ = samples.shape
    if count == 0:
        return torch.zeros((batch, tokens), device=samples.device, dtype=torch.bool)
    if samples.is_cuda:
        from .landmark_tree_triton import squared_norm

        norm_sq = squared_norm(samples)
    else:
        norm_sq = samples.float().square().sum(-1)
    threshold = norm_sq.topk(count, dim=1, largest=True, sorted=False).values.amin(
        dim=1, keepdim=True
    )
    greater = norm_sq > threshold
    equal = norm_sq == threshold
    needed = count - greater.sum(1, keepdim=True)
    # Input columns are original_sample_index order, so this prefix implements
    # the required secondary key without relying on topk's tie behavior.
    equal_rank = equal.to(torch.int32).cumsum(1)
    selected = greater | (equal & (equal_rank <= needed))
    if validate and not bool((selected.sum(1) == count).all()):
        raise RuntimeError("remainder norm selection failed")
    return selected


def _initial_hilbert_partition(
    samples: torch.Tensor,
    grid_shape: tuple[int, int, int],
    hilbert_order: torch.Tensor | None = None,
    *,
    validate_order: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active and largest-norm remainder ids, both Hilbert-stable."""

    batch, tokens, _ = samples.shape
    if math.prod(grid_shape) != tokens:
        raise ValueError("grid_shape product must equal the token count")
    remainder = tokens % 64
    remainder_mask = _largest_norm_remainder(
        samples, remainder, validate=validate_order
    )
    hilbert = (
        hilbert_order_3d(grid_shape, device=samples.device)
        if hilbert_order is None
        else hilbert_order.to(device=samples.device, dtype=torch.long)
    )
    if hilbert.shape != (tokens,):
        raise ValueError("hilbert_order must be a permutation of all token ids")
    if validate_order and not torch.equal(
        hilbert.sort().values, torch.arange(tokens, device=samples.device)
    ):
        raise ValueError("hilbert_order must be a permutation of all token ids")
    # This is the one outer stable Hilbert partition. Recursive routing below
    # moves only index metadata and performs one stable child grouping/node.
    active_tokens = tokens - remainder
    if samples.is_cuda:
        from .landmark_tree_triton import stable_binary_mask_partition

        partitioned = stable_binary_mask_partition(
            remainder_mask.contiguous(),
            hilbert.contiguous(),
            left_count=active_tokens,
        )
    else:
        order = hilbert.unsqueeze(0).expand(batch, -1)
        ordered_is_remainder = remainder_mask.gather(1, order)
        stable = ordered_is_remainder.to(torch.int8).argsort(dim=1, stable=True)
        partitioned = order.gather(1, stable)
    return partitioned[:, :active_tokens], partitioned[:, active_tokens:]


def _build_proxy_tree(
    centers: torch.Tensor,
    weights: torch.Tensor,
    child_capacities: tuple[int, ...],
    proxy_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the proxy tree levelwise, batching all nodes at the same depth."""

    batch, landmarks, dim = centers.shape
    children = len(child_capacities)
    internal_nodes = children - 1
    if centers.is_cuda and landmarks in (16, 32):
        from .landmark_tree_triton import build_weighted_proxy_tree

        return build_weighted_proxy_tree(
            centers, weights, child_capacities, proxy_iterations
        )
    fp32 = centers.float()
    alpha = torch.empty(
        (batch, landmarks, internal_nodes), device=centers.device, dtype=torch.float32
    )
    bias = torch.empty(
        (batch, internal_nodes), device=centers.device, dtype=torch.float32
    )
    norm = fp32.square().sum(-1)
    distance = norm[:, :, None] + norm[:, None, :] - 2.0 * torch.bmm(
        fp32, fp32.transpose(1, 2)
    )
    cache_key = (centers.device, landmarks, child_capacities)
    cached = _PROXY_STATIC_CACHE.get(cache_key)
    if cached is None:
        upper = torch.triu(
            torch.ones(
                (landmarks, landmarks), device=centers.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        capacity_levels: list[tuple[torch.Tensor, torch.Tensor]] = []
        static_ranges = [(0, children)]
        for _ in range(int(math.log2(children))):
            left_values = [
                sum(child_capacities[s : (s + e) // 2])
                for s, e in static_ranges
            ]
            right_values = [
                sum(child_capacities[(s + e) // 2 : e])
                for s, e in static_ranges
            ]
            capacity_levels.append(
                (
                    torch.tensor(left_values, device=centers.device, dtype=torch.long),
                    torch.tensor(right_values, device=centers.device, dtype=torch.long),
                )
            )
            next_static_ranges: list[tuple[int, int]] = []
            for start, end in static_ranges:
                middle = (start + end) // 2
                next_static_ranges.extend(((start, middle), (middle, end)))
            static_ranges = next_static_ranges
        cached = (upper, tuple(capacity_levels))
        _PROXY_STATIC_CACHE[cache_key] = cached
    upper, capacity_levels = cached
    active = weights.to(torch.long).unsqueeze(1)
    ranges = [(0, children)]
    node_offset = 0
    for depth in range(int(math.log2(children))):
        nodes = len(ranges)
        active_mask = active > 0
        valid = (
            active_mask[:, :, :, None]
            & active_mask[:, :, None, :]
            & upper[None, None, :, :]
        )
        pair = distance[:, None].masked_fill(~valid, -float("inf")).flatten(2).argmax(2)
        first = torch.div(pair, landmarks, rounding_mode="floor")
        second = pair % landmarks
        degenerate = active_mask.sum(2) < 2
        only = active_mask.to(torch.int32).argmax(2).long()
        first = torch.where(degenerate, only, first)
        second = torch.where(degenerate, only, second)
        expanded = fp32[:, None].expand(-1, nodes, -1, -1)
        left_center = expanded.gather(
            2, first[:, :, None, None].expand(-1, -1, 1, dim)
        ).squeeze(2)
        right_center = expanded.gather(
            2, second[:, :, None, None].expand(-1, -1, 1, dim)
        ).squeeze(2)
        left_capacity, right_capacity = capacity_levels[depth]
        left_weight = right_weight = None
        for _ in range(proxy_iterations):
            direction = 2.0 * (right_center - left_center)
            node_bias = left_center.square().sum(-1) - right_center.square().sum(-1)
            delta = torch.bmm(fp32, direction.transpose(1, 2)).transpose(1, 2)
            delta = delta + node_bias[:, :, None]
            order = delta.argsort(dim=2, stable=True)
            ordered_weight = active.gather(2, order)
            prefix_before = ordered_weight.cumsum(2) - ordered_weight
            take = (left_capacity[None, :, None] - prefix_before).clamp_min(0)
            take = torch.minimum(take, ordered_weight)
            left_weight = torch.zeros_like(active).scatter(2, order, take)
            right_weight = active - left_weight
            pair_weight = torch.stack((left_weight, right_weight), dim=2).reshape(
                batch, nodes * 2, landmarks
            )
            pair_center = torch.bmm(pair_weight.float(), fp32).reshape(
                batch, nodes, 2, dim
            )
            left_center = pair_center[:, :, 0] / left_capacity[None, :, None]
            right_center = pair_center[:, :, 1] / right_capacity[None, :, None]
        assert left_weight is not None and right_weight is not None
        coefficient = right_weight.float() / right_capacity[None, :, None]
        coefficient -= left_weight.float() / left_capacity[None, :, None]
        node_bias = left_center.square().sum(-1) - right_center.square().sum(-1)
        alpha[:, :, node_offset : node_offset + nodes] = coefficient.transpose(1, 2)
        bias[:, node_offset : node_offset + nodes] = node_bias
        node_offset += nodes
        active = torch.stack((left_weight, right_weight), dim=2).reshape(
            batch, nodes * 2, landmarks
        )
        next_ranges: list[tuple[int, int]] = []
        for start, end in ranges:
            middle = (start + end) // 2
            next_ranges.extend(((start, middle), (middle, end)))
        ranges = next_ranges
    return alpha.contiguous(), bias.contiguous()


def _tree_score(
    samples: torch.Tensor,
    centers: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if samples.is_cuda:
        from .landmark_tree_triton import tile_tree_score

        return tile_tree_score(samples, centers, alpha, bias)
    wide = torch.bmm(samples.float(), centers.float().transpose(1, 2))
    return 2.0 * torch.bmm(wide, alpha) + bias[:, None, :]


def _exact_tree_route(
    internal_score: torch.Tensor,
    original_indices: torch.Tensor,
    child_capacities: tuple[int, ...],
    *,
    max_original_index: int,
    validate: bool,
    compact_selection: bool = False,
) -> torch.Tensor:
    """Route exact capacities using kth values and deterministic tie cutoffs."""

    batch, tokens, internal_nodes = internal_score.shape
    children = len(child_capacities)
    if internal_nodes != children - 1:
        raise ValueError("score/tree shape mismatch")
    if internal_score.is_cuda:
        from .landmark_tree_triton import (
            active_score_index_keys,
            route_active_depth_,
        )

        active_node = torch.zeros(
            (batch, tokens), device=internal_score.device, dtype=torch.int32
        )
        cache_key = (internal_score.device, child_capacities)
        if compact_selection:
            from .landmark_tree_triton import stable_label_partition

            compact_levels = _COMPACT_ROUTE_CACHE.get(cache_key)
            if compact_levels is None:
                levels = []
                ranges = [(0, children)]
                while ranges:
                    totals: list[int] = []
                    targets: list[int] = []
                    next_ranges: list[tuple[int, int]] = []
                    for start, end in ranges:
                        middle = (start + end) // 2
                        totals.append(sum(child_capacities[start:end]))
                        targets.append(sum(child_capacities[start:middle]))
                        if middle - start > 1:
                            next_ranges.append((start, middle))
                        if end - middle > 1:
                            next_ranges.append((middle, end))
                    offsets_values = [0]
                    for total in totals[:-1]:
                        offsets_values.append(offsets_values[-1] + total)
                    offsets = (
                        torch.tensor(
                            offsets_values,
                            device=internal_score.device,
                            dtype=torch.long,
                        )
                        if len(totals) > 1
                        else None
                    )
                    runs: list[tuple[int, int, int, int, int]] = []
                    run_start = 0
                    token_start = 0
                    while run_start < len(totals):
                        run_end = run_start + 1
                        pair = (totals[run_start], targets[run_start])
                        while run_end < len(totals) and (
                            totals[run_end], targets[run_end]
                        ) == pair:
                            run_end += 1
                        runs.append(
                            (
                                pair[1],
                                run_start,
                                run_end,
                                token_start,
                                pair[0],
                            )
                        )
                        token_start += (run_end - run_start) * pair[0]
                        run_start = run_end
                    levels.append((offsets, tuple(runs)))
                    ranges = next_ranges
                compact_levels = tuple(levels)
                _COMPACT_ROUTE_CACHE[cache_key] = compact_levels
            for depth, (offsets, runs) in enumerate(compact_levels):
                node_offset = (1 << depth) - 1
                nodes = 1 << depth
                keys = active_score_index_keys(
                    internal_score, original_indices, active_node
                )
                if nodes == 1:
                    target, _, _, _, _ = runs[0]
                    thresholds = keys.kthvalue(target, dim=1).values[:, None]
                else:
                    assert offsets is not None
                    local_node = (active_node - node_offset).contiguous()
                    compact_keys = stable_label_partition(
                        local_node, offsets, keys
                    )
                    thresholds = torch.empty(
                        (batch, nodes),
                        device=internal_score.device,
                        dtype=torch.long,
                    )
                    for target, run_start, run_end, token_start, capacity in runs:
                        run_nodes = run_end - run_start
                        segment = compact_keys[
                            :, token_start : token_start + run_nodes * capacity
                        ].view(batch, run_nodes, capacity)
                        selected = segment.kthvalue(target, dim=2).values
                        thresholds[:, run_start:run_end].copy_(selected)
                route_active_depth_(
                    active_node,
                    keys,
                    thresholds.contiguous(),
                    node_offset=node_offset,
                )
            return active_node - (children - 1)

        grouped_levels = _ROUTE_GROUP_CACHE.get(cache_key)
        if grouped_levels is None:
            levels: list[tuple[tuple[int, torch.Tensor, torch.Tensor], ...]] = []
            ranges = [(0, children)]
            while ranges:
                targets: list[int] = []
                next_ranges: list[tuple[int, int]] = []
                for start, end in ranges:
                    middle = (start + end) // 2
                    targets.append(sum(child_capacities[start:middle]))
                    if middle - start > 1:
                        next_ranges.append((start, middle))
                    if end - middle > 1:
                        next_ranges.append((middle, end))
                groups: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                for target in sorted(set(targets), reverse=True):
                    local = [index for index, value in enumerate(targets) if value == target]
                    local_tensor = torch.tensor(
                        local, device=internal_score.device, dtype=torch.long
                    )
                    groups.append(
                        (
                            target,
                            local_tensor,
                            local_tensor.unsqueeze(0).unsqueeze(-1),
                        )
                    )
                levels.append(tuple(groups))
                ranges = next_ranges
            grouped_levels = tuple(levels)
            _ROUTE_GROUP_CACHE[cache_key] = grouped_levels
        for depth, groups in enumerate(grouped_levels):
            node_offset = (1 << depth) - 1
            nodes = 1 << depth
            keys = active_score_index_keys(
                internal_score, original_indices, active_node
            )
            thresholds = torch.empty(
                (batch, nodes), device=internal_score.device, dtype=torch.long
            )
            for target, local_nodes, local_view in groups:
                global_view = local_view + node_offset
                member = active_node[:, None, :] == global_view
                candidates = keys[:, None, :].masked_fill(
                    ~member, torch.iinfo(torch.int64).max
                )
                selected = candidates.kthvalue(target, dim=2).values
                thresholds.index_copy_(1, local_nodes, selected)
            route_active_depth_(
                active_node,
                keys,
                thresholds,
                node_offset=node_offset,
            )
        return active_node - (children - 1)

    # CPU reference keeps the explicit two-key implementation below.
    lexicographic_key = None
    active_node = torch.zeros(
        (batch, tokens), device=internal_score.device, dtype=torch.long
    )
    frontier: list[tuple[int, int, int]] = [(0, 0, children)]
    while frontier:
        node, start, end = frontier.pop(0)
        middle = (start + end) // 2
        left_capacity = sum(child_capacities[start:middle])
        member = active_node == node
        if lexicographic_key is not None:
            key = lexicographic_key[:, :, node]
            threshold = key.masked_fill(~member, torch.iinfo(torch.int64).max).kthvalue(
                left_capacity, dim=1
            ).values[:, None]
            left = member & (key <= threshold)
        else:
            score = internal_score[:, :, node]
            threshold = score.masked_fill(~member, float("inf")).kthvalue(
                left_capacity, dim=1
            ).values[:, None]
            lower = member & (score < threshold)
            equal = member & (score == threshold)
            needed = left_capacity - lower.sum(1)
            low = torch.full_like(needed, -1)
            high = torch.full_like(needed, max_original_index)
            for _ in range(max(1, (max_original_index + 1).bit_length())):
                middle_index = torch.div(low + high, 2, rounding_mode="floor")
                count = (
                    equal & (original_indices <= middle_index[:, None])
                ).sum(1)
                enough = count >= needed
                high = torch.where(enough, middle_index, high)
                low = torch.where(enough, low, middle_index)
            left = lower | (equal & (original_indices <= high[:, None]))
        if validate and not bool((left.sum(1) == left_capacity).all()):
            raise RuntimeError("exact tree routing failed its left capacity")
        active_node = torch.where(
            member & left,
            torch.full_like(active_node, 2 * node + 1),
            torch.where(
                member,
                torch.full_like(active_node, 2 * node + 2),
                active_node,
            ),
        )
        if middle - start > 1:
            frontier.append((2 * node + 1, start, middle))
        if end - middle > 1:
            frontier.append((2 * node + 2, middle, end))
    return active_node - (children - 1)


def _stable_counting_partition(
    labels: torch.Tensor,
    child_capacities: tuple[int, ...],
    *,
    validate: bool,
    source_indices: torch.Tensor | None = None,
    samples: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return stable grouped local positions or directly grouped source ids."""

    batch, tokens = labels.shape
    if labels.is_cuda and not validate:
        from .landmark_tree_triton import (
            stable_label_partition,
            stable_label_partition_with_features,
        )

        cache_key = (labels.device, child_capacities)
        offsets = _PARTITION_OFFSET_CACHE.get(cache_key)
        if offsets is None:
            values = [0]
            for capacity in child_capacities[:-1]:
                values.append(values[-1] + capacity)
            offsets = torch.tensor(values, device=labels.device, dtype=torch.long)
            _PARTITION_OFFSET_CACHE[cache_key] = offsets
        if samples is not None:
            if source_indices is None:
                raise ValueError("feature partition requires source_indices")
            return stable_label_partition_with_features(
                labels.contiguous(), offsets, source_indices, samples
            )
        return stable_label_partition(labels.contiguous(), offsets, source_indices)
    local = torch.arange(tokens, device=labels.device).expand(batch, -1)
    order = torch.zeros_like(local)
    offset = 0
    for label, capacity in enumerate(child_capacities):
        member = labels == label
        if validate and not bool((member.sum(1) == capacity).all()):
            raise RuntimeError("fine labels do not match requested capacities")
        rank = member.to(torch.long).cumsum(1) - 1
        destination = torch.where(member, rank + offset, torch.zeros_like(rank))
        source = torch.where(member, local, torch.zeros_like(local))
        # Every valid destination has one writer; masked rows contribute zero.
        order.scatter_add_(1, destination, source)
        offset += capacity
    mapped = order if source_indices is None else source_indices.gather(1, order)
    if samples is None:
        return mapped
    gather_index = order.unsqueeze(-1).expand(-1, -1, samples.shape[2])
    return mapped, samples.gather(1, gather_index)

