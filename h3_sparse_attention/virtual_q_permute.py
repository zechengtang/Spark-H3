"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


import triton


import triton.language as tl


@triton.jit
def _permute_anchor(Q, IDX, RANGES, OUT, A, PART,
                    T: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
                    P: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr):
    group, bh = tl.program_id(0), tl.program_id(1)
    b, h = (bh // H).to(tl.int64), bh % H
    if SPLIT:
        start = group * 64
        end = tl.minimum(start + 64, T)
    else:
        start = tl.load(RANGES + group * 2)
        end = tl.load(RANGES + group * 2 + 1)
    ii = tl.arange(0, 64)
    dd = tl.arange(0, 128)
    acc = tl.zeros((128,), tl.float32)
    for offset in range(start, end, 64):
        rows = offset + ii
        src = tl.load(IDX + (b * H + h) * V + rows,
                      (rows < end) & (rows < V), 0).to(tl.int64)
        src = tl.where(rows < V, src, rows)
        value = tl.load(Q + ((b * T + src[:, None]) * H + h) * 128 + dd[None, :],
                        (rows < end)[:, None], 0)
        tl.store(OUT + ((b * T + rows[:, None]) * H + h) * 128 + dd[None, :],
                 value, (rows < end)[:, None])
        acc += tl.sum(value.to(tl.float32), 0)
    if SPLIT:
        tl.store(PART + ((b * N + group) * H + h) * 128 + dd, acc)
    else:
        tl.store(A + ((b * P + group) * H + h) * 128 + dd, acc / (end - start))


@triton.jit
def _reduce_parts(PART, RANGES, A, H: tl.constexpr, N: tl.constexpr, P: tl.constexpr):
    parent, bh = tl.program_id(0), tl.program_id(1)
    b, h = (bh // H).to(tl.int64), bh % H
    start = tl.load(RANGES + parent * 2)
    end = tl.load(RANGES + parent * 2 + 1)
    dd = tl.arange(0, 128)
    acc = tl.zeros((128,), tl.float32)
    for offset in range(start, end, 64):
        acc += tl.load(PART + ((b * N + offset // 64) * H + h) * 128 + dd)
    tl.store(A + ((b * P + parent) * H + h) * 128 + dd, acc / (end - start))


def permute_with_virtual_anchors(q, permutation, virtual_ranges, *, video_tokens, split=False):
    """Return reordered Q and parent anchors; inputs use validated LMv2 topology."""
    if q.ndim != 4 or q.shape[-1] != 128 or not q.is_cuda or not q.is_contiguous() or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('requires contiguous CUDA BTH128 BF16/FP16 Q')
    b, t, h, d = q.shape
    if not 0 < video_tokens <= t or permutation.shape != (b, h, video_tokens):
        raise ValueError('invalid video token count or permutation shape')
    if virtual_ranges.ndim != 2 or virtual_ranges.shape[1] != 2 or not virtual_ranges.shape[0]:
        raise ValueError('requires nonempty validated ranges[P,2]')
    for value in (permutation, virtual_ranges):
        if value.device != q.device or value.dtype != torch.int64 or not value.is_contiguous():
            raise ValueError('permutation and ranges require contiguous CUDA int64')
    p, n = virtual_ranges.shape[0], triton.cdiv(t, 64)
    out = torch.empty_like(q)
    anchors = torch.empty((b, p, h, d), device=q.device, dtype=q.dtype)
    partial = torch.empty((b, n, h, d), device=q.device, dtype=torch.float32) if split else anchors
    _permute_anchor[(n if split else p, b * h)](
        q, permutation, virtual_ranges, out, anchors, partial, t, h, video_tokens, p, n, split,
        num_warps=4)
    if split:
        _reduce_parts[(p, b * h)](partial, virtual_ranges, anchors, h, n, p, num_warps=4)
    return out, anchors

