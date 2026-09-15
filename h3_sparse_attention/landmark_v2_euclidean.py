"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import os


import torch


import triton


import triton.language as tl


from .landmark_tree_triton import build_weighted_proxy_tree


DIRECT_ENABLED = os.environ.get("H3_LMV2_EUCLIDEAN_DIRECT", "1") == "1"


@triton.jit
def _score(X, INDICES, DIRECTION, BIAS, OUT,
           N: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
           BM: tl.constexpr, BD: tl.constexpr, INDIRECT: tl.constexpr):
    batch = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    nodes = tl.arange(0, 16)
    if INDIRECT:
        source_rows = tl.load(INDICES + batch * N + rows, rows < N, 0).to(tl.int64)
    else:
        source_rows = batch * N + rows
    x = tl.load(X + source_rows[:, None] * D + dims[None, :],
                (rows[:, None] < N) & (dims[None, :] < D), 0).to(tl.float32)
    direction = tl.load(
        DIRECTION + (batch * S + nodes[:, None]) * D + dims[None, :],
        (nodes[:, None] < S) & (dims[None, :] < D), 0)
    bias = tl.load(BIAS + batch * S + nodes, nodes < S, 0)
    margin = tl.dot(x, tl.trans(direction), input_precision="tf32x3") + bias[None, :]
    tl.store(OUT + (batch * N + rows[:, None]) * S + nodes[None, :], margin,
             (rows[:, None] < N) & (nodes[None, :] < S))


def build_euclidean_directions(centers, weights, capacities):
    return build_weighted_proxy_tree(
        centers, weights, capacities, proxy_iterations=2, direct_output=True)


def fused_euclidean_scores(samples, directions, bias, *, indices=None):
    """Return x @ (2*(right-left)).T + ||left||² - ||right||².

    Samples are contiguous [B,N,D], or [total_tokens,D] with int64 [B,N]
    global indices. Directions and bias stay FP32; no cosine normalization.
    """
    if not (samples.is_cuda and samples.is_contiguous()
            and samples.dtype in (torch.float16, torch.bfloat16)):
        raise ValueError("samples must be contiguous CUDA FP16/BF16")
    if indices is None:
        if samples.ndim != 3:
            raise ValueError("contiguous samples must have shape [B,N,D]")
        batch, tokens, dim = samples.shape
    else:
        if (samples.ndim != 2 or indices.ndim != 2 or indices.dtype != torch.int64
                or not indices.is_contiguous() or indices.device != samples.device):
            raise ValueError("indexed samples require [T,D] and contiguous int64 [B,N] indices")
        batch, tokens = indices.shape
        dim = samples.shape[-1]
    if directions.ndim != 3:
        raise ValueError("directions must have shape [B,S,D]")
    nodes = directions.shape[1]
    if (not 1 <= nodes <= 15 or not 1 <= dim <= 256
            or directions.shape != (batch, nodes, dim) or bias.shape != (batch, nodes)
            or any(t.dtype != torch.float32 or t.device != samples.device
                   or not t.is_contiguous() for t in (directions, bias))):
        raise ValueError("expected contiguous FP32 directions/bias, 1..15 nodes, D<=256")
    out = torch.empty((batch, tokens, nodes), device=samples.device, dtype=torch.float32)
    _score[(triton.cdiv(tokens, 64), batch)](
        samples, indices if indices is not None else samples, directions, bias, out,
        N=tokens, D=dim, S=nodes, BM=64, BD=max(32, triton.next_power_of_2(dim)),
        INDIRECT=indices is not None, num_warps=4)
    return out


def use_direct(centers):
    return (DIRECT_ENABLED and centers.is_cuda and centers.is_contiguous()
            and centers.dtype in (torch.float16, torch.bfloat16)
            and centers.shape[1] in (16, 32) and 0 < centers.shape[-1] <= 256)


def euclidean_proxy_scores(samples, centers, weights, capacities, *, indices=None):
    if use_direct(centers):
        directions, bias = build_euclidean_directions(centers, weights, capacities)
        return fused_euclidean_scores(samples, directions, bias, indices=indices)
    from .landmark_tree_clustering import _build_proxy_tree, _tree_score
    alpha, bias = _build_proxy_tree(centers, weights, capacities, proxy_iterations=2)
    if indices is not None:
        from .landmark_tree_triton import tile_tree_score_indexed
        return tile_tree_score_indexed(samples, indices, centers, alpha, bias)
    return _tree_score(samples, centers, alpha, bias)

