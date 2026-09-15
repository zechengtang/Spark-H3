"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


import triton


import triton.language as tl


from triton.language.extra.cuda import libdevice


@triton.jit
def _max_score(X, CENTERS, ACTIVE, OUT, INDICES, N: tl.constexpr, K: tl.constexpr,
               F: tl.constexpr, BM: tl.constexpr, INDIRECT: tl.constexpr):
    batch = tl.program_id(1)
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    dims = tl.arange(0, 128)
    cols = tl.arange(0, K)
    if INDIRECT:
        source_rows = tl.load(INDICES + batch*N + rows, rows < N, 0).to(tl.int64)
    else:
        source_rows = batch*N + rows
    x = tl.load(X + source_rows[:, None]*128+dims[None, :], rows[:, None] < N, 0).to(tl.float32)
    c = tl.load(CENTERS + (batch*K+cols[:, None])*128+dims[None, :])
    dot = tl.dot(x, tl.trans(c), input_precision="tf32x3")
    norm = tl.maximum(libdevice.sqrt_rn(tl.sum(x*x, 1)), 1.e-12)
    for node in tl.static_range(F-1):
        lw = tl.load(ACTIVE + (batch*(2*F-1)+2*node+1)*K+cols)
        rw = tl.load(ACTIVE + (batch*(2*F-1)+2*node+2)*K+cols)
        left = tl.max(tl.where(lw[None, :] > 0, dot, -float("inf")), 1)
        right = tl.max(tl.where(rw[None, :] > 0, dot, -float("inf")), 1)
        tl.store(OUT+(batch*N+rows)*(F-1)+node, (right-left)/norm, rows < N)


def max_support_scores(samples, normalized_centers, active):
    """All K distance columns stay on chip; only F-1 margins are written."""
    batch, rows, dim = samples.shape
    landmarks = normalized_centers.shape[1]
    children = (active.shape[1]+1)//2
    if not (samples.is_cuda and dim == 128 and samples.is_contiguous()
            and landmarks in (16, 32)):
        return max_support_scores_reference(samples, normalized_centers, active)
    out = torch.empty((batch, rows, children-1), device=samples.device, dtype=torch.float32)
    _max_score[(triton.cdiv(rows, 64), batch)](
        samples, normalized_centers.contiguous(), active.contiguous(), out, samples,
        N=rows, K=landmarks, F=children, BM=64, INDIRECT=False, num_warps=4)
    return out


def max_support_scores_indexed(source, indices, normalized_centers, active):
    """Read group-one token features directly, preserving max-score arithmetic."""
    batch, rows = indices.shape
    landmarks = normalized_centers.shape[1]
    children = (active.shape[1]+1)//2
    out = torch.empty((batch, rows, children-1), device=source.device, dtype=torch.float32)
    _max_score[(triton.cdiv(rows, 64), batch)](
        source, normalized_centers, active, out, indices,
        N=rows, K=landmarks, F=children, BM=64, INDIRECT=True, num_warps=4,
    )
    return out


def max_support_scores_reference(samples, normalized_centers, active):
    x = samples.float()
    z = torch.bmm(x, normalized_centers.float().transpose(1, 2))
    z = z / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    children = (active.shape[1]+1)//2
    return torch.stack([
        z.masked_fill(active[:, 2*n+2, None, :] <= 0, -float("inf")).amax(-1)
        - z.masked_fill(active[:, 2*n+1, None, :] <= 0, -float("inf")).amax(-1)
        for n in range(children-1)
    ], dim=-1)


def reference_proxy_partition(centers, weights, child_capacities):
    """Independent PyTorch stable weighted proxy fit, retaining all tree nodes."""
    from .landmark_v2_cosine import unit
    x = centers.float()
    b, k, d = x.shape
    f = len(child_capacities)
    normalized = unit(x)
    distances = 1-torch.bmm(normalized, normalized.transpose(1, 2))
    upper = torch.ones((k, k), device=x.device, dtype=torch.bool).triu(1)
    active = torch.zeros((b, 2*f-1, k), device=x.device, dtype=torch.int32)
    active[:, 0] = weights
    ranges = [(0, f)]
    offset = 0
    while ranges:
        nodes = len(ranges)
        w = active[:, offset:offset+nodes].long()
        valid = w > 0
        pairs = valid[..., :, None] & valid[..., None, :] & upper
        pair = distances[:, None].masked_fill(~pairs, -float("inf")).flatten(2).argmax(2)
        first, second = pair//k, pair%k
        only = valid.int().argmax(2)
        first = torch.where(valid.sum(2) < 2, only, first)
        second = torch.where(valid.sum(2) < 2, only, second)
        expanded = x[:, None].expand(-1, nodes, -1, -1)
        left = expanded.gather(2, first[..., None, None].expand(-1, -1, 1, d)).squeeze(2)
        right = expanded.gather(2, second[..., None, None].expand(-1, -1, 1, d)).squeeze(2)
        lc = torch.tensor([sum(child_capacities[a:(a+e)//2]) for a,e in ranges], device=x.device)
        rc = torch.tensor([sum(child_capacities[(a+e)//2:e]) for a,e in ranges], device=x.device)
        for _ in range(2):
            scores = torch.bmm(normalized, (unit(right)-unit(left)).transpose(1,2)).transpose(1,2)
            order = scores.argsort(dim=2, stable=True)
            ordered = w.gather(2, order)
            before = ordered.cumsum(2)-ordered
            take = torch.minimum((lc[None,:,None]-before).clamp_min(0), ordered)
            lw = torch.zeros_like(w).scatter(2, order, take)
            rw = w-lw
            left = torch.bmm(lw.float(), x)/lc[None,:,None]
            right = torch.bmm(rw.float(), x)/rc[None,:,None]
        for n in range(nodes):
            active[:, 2*(offset+n)+1] = lw[:, n]
            active[:, 2*(offset+n)+2] = rw[:, n]
        offset += nodes
        ranges = [(a,e) for s,t in ranges for a,e in ((s,(s+t)//2),((s+t)//2,t)) if e-a > 1]
    return normalized, active


def cosine_max_proxy_scores_reference(samples, centers, weights, child_capacities):
    normalized, active = reference_proxy_partition(centers, weights, child_capacities)
    return max_support_scores_reference(samples, normalized, active)


def cosine_max_proxy_scores(samples, centers, weights, child_capacities):
    if not (samples.is_cuda and centers.shape[-1] == 128):
        return cosine_max_proxy_scores_reference(samples, centers, weights, child_capacities)
    from .landmark_v2_cosine_fast import build_cosine_directions
    _, normalized, active = build_cosine_directions(
        centers, weights, child_capacities, return_partition=True)
    return max_support_scores(samples, normalized, active)

