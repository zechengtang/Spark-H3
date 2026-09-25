"""Layout-aware full-token projection used by ComfyUI reblock planning."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bthd_projection_kernel(
    source, factor, output, score_directions, score_output,
    tokens: tl.constexpr, heads: tl.constexpr,
    block_m: tl.constexpr, store_scores: tl.constexpr,
    factor_stride_batch: tl.constexpr,
    factor_stride_k: tl.constexpr,
    factor_stride_n: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head - batch * heads
    rows = tile * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, 128)
    reduction = tl.arange(0, 128)
    source_offsets = (
        ((batch * tokens + rows[:, None]) * heads + head) * 128
        + reduction[None, :]
    )
    factor_offsets = (
        batch_head * factor_stride_batch
        + reduction[:, None] * factor_stride_k
        + cols[None, :] * factor_stride_n
    )
    values = tl.load(
        source + source_offsets, mask=rows[:, None] < tokens, other=0.0
    )
    transform = tl.load(factor + factor_offsets)
    result = tl.dot(values, transform)
    output_offsets = (
        batch_head * tokens * 128 + rows[:, None] * 128 + cols[None, :]
    )
    mask = rows[:, None] < tokens
    tl.store(output + output_offsets, result, mask=mask)
    if store_scores:
        score_cols = tl.arange(0, 16)
        direction = tl.load(
            score_directions
            + (batch_head * 15 + score_cols[:, None]) * 128
            + reduction[None, :],
            mask=score_cols[:, None] < 15,
            other=0.0,
        ).to(tl.float32)
        projected = result.to(tl.bfloat16).to(tl.float32)
        norm = tl.maximum(
            tl.sqrt(tl.sum(projected * projected, axis=1)), 1.0e-12
        )
        score = tl.dot(
            (projected / norm[:, None]).to(tl.float16),
            tl.trans(direction.to(tl.float16)),
        )
        score_offsets = (
            (batch_head * tokens + rows[:, None]) * 15 + score_cols[None, :]
        )
        tl.store(
            score_output + score_offsets, score,
            mask=(rows[:, None] < tokens) & (score_cols[None, :] < 15),
        )


def project_bthd(
    source: torch.Tensor,
    factor: torch.Tensor,
    *,
    out: torch.Tensor,
    block_m: int = 64,
    score_directions: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute each BTHD head's ``[T,128] @ [128,128]`` without transposing."""
    if (
        source.ndim != 4 or source.shape[-1] != 128
        or source.dtype != torch.bfloat16 or not source.is_cuda
        or not source.is_contiguous()
    ):
        raise ValueError("source must be contiguous CUDA BF16 BTHD with dim=128")
    batch, tokens, heads, _ = source.shape
    if factor.shape != (batch * heads, 128, 128) or factor.dtype != torch.bfloat16:
        raise ValueError("factor must be BF16 [B*H,128,128]")
    if (
        out.shape != (batch * heads, tokens, 128)
        or out.dtype != torch.bfloat16 or not out.is_contiguous()
        or out.device != source.device
    ):
        raise ValueError("out must be contiguous CUDA BF16 [B*H,T,128]")
    if (score_directions is None) != (score_out is None):
        raise ValueError("score_directions and score_out must be provided together")
    if score_directions is not None and (
        score_directions.shape != (batch * heads, 15, 128)
        or score_directions.dtype != torch.float32
        or not score_directions.is_contiguous()
        or score_directions.device != source.device
        or score_out.shape != (batch * heads, tokens, 15)
        or score_out.dtype != torch.float32
        or not score_out.is_contiguous()
        or score_out.device != source.device
    ):
        raise ValueError("root score buffers have invalid shape, dtype, or layout")
    if block_m not in (32, 64, 128):
        raise ValueError("block_m must be 32, 64, or 128")
    _bthd_projection_kernel[(triton.cdiv(tokens, block_m), batch * heads)](
        source, factor, out,
        score_directions if score_directions is not None else out,
        score_out if score_out is not None else out,
        tokens=tokens, heads=heads, block_m=block_m,
        store_scores=score_directions is not None,
        factor_stride_batch=factor.stride(0),
        factor_stride_k=factor.stride(1),
        factor_stride_n=factor.stride(2),
        num_warps=8 if block_m >= 64 else 4,
        num_stages=3,
    )
    return out


__all__ = ["project_bthd"]
