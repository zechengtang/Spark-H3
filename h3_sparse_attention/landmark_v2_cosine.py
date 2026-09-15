"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import os


import torch


_CAPACITY_CACHE = {}


_legacy_selected = any(name in os.environ for name in
                       ("H3_LMV2_COS_PARITY", "H3_LMV2_COS_FUSED_UNIT"))


FAST_ENABLED = os.environ.get("H3_LMV2_COS_FAST", "0" if _legacy_selected else "1") == "1"


FAST_PRECISION = os.environ.get("H3_LMV2_COS_PRECISION", "fp16")


PARITY_ENABLED = os.environ.get("H3_LMV2_COS_PARITY", "0") == "1"


FUSED_UNIT_ENABLED = os.environ.get("H3_LMV2_COS_FUSED_UNIT", "0") == "1"


def unit(x):
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def cosine_split_scores(samples, left, right):
    """Positive prefers right; d_cos(x,left)-d_cos(x,right)."""
    return torch.bmm(unit(samples), (unit(right) - unit(left)).transpose(1, 2))


def cosine_proxy_scores_reference(samples, centers, weights, child_capacities):
    """Build capacity-weighted binary proxy tree then score actual mean tokens."""
    x = centers.float()
    batch, landmarks, dim = x.shape
    normalized = unit(x)
    distance = 1 - torch.bmm(normalized, normalized.transpose(1, 2))
    upper = torch.ones((landmarks, landmarks), device=x.device, dtype=torch.bool).triu(1)
    active = weights.long().unsqueeze(1)
    ranges = [(0, len(child_capacities))]
    all_scores = []
    while ranges:
        nodes = len(ranges)
        valid = (active > 0)
        pairs = valid[..., :, None] & valid[..., None, :] & upper
        pair = distance[:, None].masked_fill(~pairs, -float("inf")).flatten(2).argmax(2)
        first, second = pair // landmarks, pair % landmarks
        only = valid.int().argmax(2)
        first = torch.where(valid.sum(2) < 2, only, first)
        second = torch.where(valid.sum(2) < 2, only, second)
        expanded = x[:, None].expand(-1, nodes, -1, -1)
        left = expanded.gather(2, first[..., None, None].expand(-1, -1, 1, dim)).squeeze(2)
        right = expanded.gather(2, second[..., None, None].expand(-1, -1, 1, dim)).squeeze(2)
        cache_key = (x.device, child_capacities, tuple(ranges))
        capacities = _CAPACITY_CACHE.get(cache_key)
        if capacities is None:
            capacities = (
                torch.tensor([sum(child_capacities[s:(s+e)//2]) for s,e in ranges], device=x.device),
                torch.tensor([sum(child_capacities[(s+e)//2:e]) for s,e in ranges], device=x.device),
            )
            _CAPACITY_CACHE[cache_key] = capacities
        lc, rc = capacities
        for _ in range(2):
            scores = cosine_split_scores(x, left, right).transpose(1, 2)
            order = scores.argsort(dim=2, stable=True)
            ordered = active.gather(2, order)
            before = ordered.cumsum(2) - ordered
            take = torch.minimum((lc[None, :, None] - before).clamp_min(0), ordered)
            lw = torch.zeros_like(active).scatter(2, order, take)
            rw = active - lw
            left = torch.bmm(lw.float(), x) / lc[None, :, None]
            right = torch.bmm(rw.float(), x) / rc[None, :, None]
        all_scores.append(cosine_split_scores(samples, left, right))
        active = torch.stack((lw, rw), dim=2).reshape(batch, nodes * 2, landmarks)
        ranges = [(a,b) for s,e in ranges for a,b in ((s,(s+e)//2),((s+e)//2,e)) if b-a > 1]
    return torch.cat(all_scores, dim=-1).contiguous()


def cosine_proxy_scores(samples, centers, weights, child_capacities):
    """Fast COS default; explicit legacy controls remain available at startup."""
    if FAST_ENABLED:
        return cosine_proxy_scores_fast(samples, centers, weights, child_capacities)
    if PARITY_ENABLED:
        return cosine_proxy_scores_parity(samples, centers, weights, child_capacities)
    function = cosine_proxy_scores_fused_unit if FUSED_UNIT_ENABLED else cosine_proxy_scores_v1
    return function(samples, centers, weights, child_capacities)


def cosine_proxy_scores_fused_unit(samples, centers, weights, child_capacities):
    """Bitwise-tested normalization-fusion candidate; raw centroid updates unchanged."""
    if not samples.is_cuda:
        return cosine_proxy_scores_reference(samples, centers, weights, child_capacities)
    from .landmark_v2_cosine_triton import fused_unit128
    return _cosine_proxy_scores_cuda(samples, centers, weights, child_capacities, fused_unit128)


def cosine_proxy_scores_parity(samples, centers, weights, child_capacities):
    """Explicit candidate: exact fused normalization and proxy directions."""
    if not samples.is_cuda:
        return cosine_proxy_scores_reference(samples, centers, weights, child_capacities)
    from .landmark_v2_cosine_triton import fused_unit128, fused_direction128
    return _cosine_proxy_scores_cuda(samples, centers, weights, child_capacities,
                                     fused_unit128, fused_direction128, fused_means=True)


def cosine_proxy_scores_fast(samples, centers, weights, child_capacities, mode=None, fused_proxy=True):
    """Algorithm-equivalent fused scoring; precision explicitly configurable."""
    if not samples.is_cuda:
        return cosine_proxy_scores_reference(samples, centers, weights, child_capacities)
    from .landmark_v2_cosine_triton import fused_unit128, fused_direction128
    from .landmark_v2_cosine_fast import fused_cosine_scores, build_cosine_directions
    if fused_proxy and centers.shape[-1] == 128:
        directions = build_cosine_directions(centers, weights, child_capacities)
        return fused_cosine_scores(samples, directions, mode or FAST_PRECISION)
    score = lambda x, d: fused_cosine_scores(x, d, mode or FAST_PRECISION)
    return _cosine_proxy_scores_cuda(samples, centers, weights, child_capacities,
        fused_unit128, fused_direction128, fused_means=True, score_samples=score)


def cosine_proxy_scores_v1(samples, centers, weights, child_capacities):
    """Accepted pre-fusion CUDA implementation, retained for independent timing."""
    if not samples.is_cuda:
        return cosine_proxy_scores_reference(samples, centers, weights, child_capacities)
    return _cosine_proxy_scores_cuda(samples, centers, weights, child_capacities, unit)


def _cosine_proxy_scores_cuda(samples, centers, weights, child_capacities, normalize, direction=None, fused_means=False, score_samples=None):
    from .landmark_v2_cosine_triton import proxy_seeds, partition_weights
    if direction is None:
        direction = lambda left, right: normalize(right) - normalize(left)
    x = centers.float().contiguous()
    batch, landmarks, dim = x.shape
    normalized = normalize(x)
    distance = 1 - torch.bmm(normalized, normalized.transpose(1, 2))
    active = weights.long().unsqueeze(1).contiguous()
    ranges = [(0, len(child_capacities))]
    directions = []
    while ranges:
        nodes = len(ranges)
        left, right = proxy_seeds(x, distance, active)
        cache_key = (x.device, child_capacities, tuple(ranges))
        capacities = _CAPACITY_CACHE.get(cache_key)
        if capacities is None:
            capacities = (
                torch.tensor([sum(child_capacities[s:(s+e)//2]) for s,e in ranges], device=x.device),
                torch.tensor([sum(child_capacities[(s+e)//2:e]) for s,e in ranges], device=x.device),
            )
            _CAPACITY_CACHE[cache_key] = capacities
        lc, rc = capacities
        current_direction = direction(left, right) if fused_means else None
        for _ in range(2):
            scores = torch.bmm(normalized, (current_direction if fused_means else direction(left, right)).transpose(1,2)).transpose(1,2)
            order = scores.argsort(dim=2, stable=True)
            if fused_means:
                lw, rw, lf, rf = partition_weights(active, order, lc, float_output=True)
            else:
                lw, rw = partition_weights(active, order, lc)
            # Preserve the reference GEMM shapes and raw mean arithmetic.
            if fused_means:
                left_sum = torch.bmm(lf, x)
                right_sum = torch.bmm(rf, x)
                current_direction = direction(left_sum, right_sum, lc, rc)
            else:
                left = torch.bmm(lw.float(), x) / lc[None,:,None]
                right = torch.bmm(rw.float(), x) / rc[None,:,None]
        directions.append(current_direction if fused_means else direction(left, right))
        active = torch.stack((lw,rw), dim=2).reshape(batch,nodes*2,landmarks)
        ranges = [(a,b) for s,e in ranges for a,b in ((s,(s+e)//2),((s+e)//2,e)) if b-a>1]
    if score_samples is not None:
        return score_samples(samples, directions)
    # Normalize the large representative matrix once, not once per depth.
    normalized_samples = normalize(samples)
    # Keep the original 1/2/4-column GEMM shapes: a merged 7-column GEMM can
    # change FP32 rounding enough to move exact-capacity boundaries at group1.
    return torch.cat([torch.bmm(normalized_samples,d.transpose(1,2)) for d in directions],dim=-1).contiguous()

