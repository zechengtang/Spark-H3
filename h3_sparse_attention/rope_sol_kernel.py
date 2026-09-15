"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math


import torch


import triton


import triton.language as tl


from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK_SIZE = 64


ROUTE_GROUP_SIZE = 32


_CUTE_COMPILED = {}


_CUTE_THRESHOLD_COMPILED = {}


def rope_sol_backend(device: torch.device | str | int | None = None) -> str:
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability == (12, 0):
        try:
            from sol_attn.sm120 import make_kernel

            make_kernel(external_route=True)
        except (ImportError, TypeError):
            pass
        else:
            return "cute_sm120_explicit_route"
    return "triton_explicit_route"


def sol_topk_threshold_backend(
    device: torch.device | str | int | None = None,
) -> str | None:
    """Return the stock-threshold backend used by fixed-ratio SOL.

    The cutoff path deliberately targets the stock SM120 CuTe mainloop. Other
    architectures retain the existing explicit-route implementation.
    """

    if device is None:
        device = torch.cuda.current_device()
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        return None
    try:
        from sol_attn.sm120 import make_kernel

        make_kernel(external_route=False, hybrid_route=True, force_local_blocks=False)
    except (ImportError, TypeError):
        return None
    return "cute_sm120_topk_threshold"


def _rope_sol_attn_cute(
    q,
    k,
    v,
    key_centroids,
    value_sums,
    route_mask,
    *,
    scale,
    sink_start_block,
    sink_end_block,
):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from sol_attn.common import to_cute_tensor
    from sol_attn.sm120 import make_kernel

    batch, tokens, heads, _ = q.shape
    output = torch.empty_like(v)
    threshold = torch.zeros(
        (batch, math.ceil(tokens / BLOCK_SIZE), heads),
        device=q.device,
        dtype=torch.float32,
    )
    lse = torch.empty(
        (batch, tokens, heads), device=q.device, dtype=torch.float32
    )
    tensors = [
        q,
        k,
        v,
        output,
        key_centroids,
        value_sums,
        threshold,
        route_mask,
        lse,
    ]
    args = [to_cute_tensor(tensor) for tensor in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (q.device.index, batch, tokens, heads)
    compiled = _CUTE_COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            make_kernel(external_route=True),
            *args,
            scale,
            sink_start_block,
            sink_end_block,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _CUTE_COMPILED[key] = compiled
    compiled(
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
    )
    return output


def _sol_topk_threshold_attn_cute(
    q,
    k,
    v,
    key_centroids,
    value_sums,
    threshold,
    route_mask,
    *,
    scale,
    sink_start_block,
    sink_end_block,
    force_local_blocks=False,
):
    """Invoke the stock SM120 CuTe route selector with caller cutoffs."""

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from sol_attn.common import to_cute_tensor
    from sol_attn.sm120 import make_kernel

    batch, tokens, heads, _ = q.shape
    output = torch.empty_like(v)
    lse = torch.empty(
        (batch, tokens, heads), device=q.device, dtype=torch.float32
    )
    hybrid_route = route_mask is not None
    # The local SM120 compatibility patch gives stock, hybrid, and
    # explicit-route kernels the same ABI. Pure stock mode never dereferences
    # this placeholder; hybrid mode reads only rows whose cutoff is NaN.
    route_argument = (
        torch.empty((1,), device=q.device, dtype=torch.uint8)
        if route_mask is None
        else route_mask
    )
    tensors = [
        q,
        k,
        v,
        output,
        key_centroids,
        value_sums,
        threshold,
        route_argument,
        lse,
    ]
    args = [to_cute_tensor(tensor) for tensor in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (q.device.index, batch, tokens, heads, hybrid_route, force_local_blocks)
    compiled = _CUTE_THRESHOLD_COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            make_kernel(external_route=False, hybrid_route=hybrid_route, force_local_blocks=force_local_blocks),
            *args,
            scale,
            sink_start_block,
            sink_end_block,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        _CUTE_THRESHOLD_COMPILED[key] = compiled
    compiled(
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
    )
    return output


def sol_topk_threshold_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    threshold: torch.Tensor,
    route_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    sink_start: int | None = None,
    sink_tokens: int = 0,
    force_local_blocks: bool = False,
) -> torch.Tensor:
    """Run fixed-ratio SOL through the stock SM120 threshold selector."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must share shape [B, T, H, D]")
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise TypeError(
            "sol_topk_threshold_attn requires BF16 inputs with head dimension 128"
        )
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous BTHD tensors")
    batch, tokens, heads, head_dim = q.shape
    blocks = math.ceil(tokens / BLOCK_SIZE)
    expected_summaries = (batch, blocks, heads, head_dim)
    if key_centroids.shape != expected_summaries or value_sums.shape != expected_summaries:
        raise ValueError(
            "Kc/Vc shape mismatch: "
            f"expected {expected_summaries}, got {key_centroids.shape}/{value_sums.shape}"
        )
    expected_threshold = (batch, blocks, heads)
    if threshold.shape != expected_threshold:
        raise ValueError(
            f"threshold must have shape {expected_threshold}, got {threshold.shape}"
        )
    if threshold.dtype != torch.float32 or not threshold.is_contiguous():
        raise TypeError("threshold must be a contiguous float32 tensor")
    if threshold.device != q.device:
        raise ValueError("threshold must be on the Q/K/V device")
    if route_mask is not None:
        expected_route = (batch, blocks, heads, blocks)
        if route_mask.shape != expected_route:
            raise ValueError(
                f"route_mask must have shape {expected_route}, got {route_mask.shape}"
            )
        if (
            route_mask.dtype not in (torch.bool, torch.uint8)
            or not route_mask.is_contiguous()
        ):
            raise TypeError("route_mask must be a contiguous bool/uint8 tensor")
        if route_mask.device != q.device:
            raise ValueError("route_mask must be on the Q/K/V device")
    if sink_start is None:
        sink_start = tokens - sink_tokens
    if not 0 <= sink_start <= tokens or not 0 <= sink_tokens <= tokens - sink_start:
        raise ValueError("invalid sink range")
    if sol_topk_threshold_backend(q.device) is None:
        raise RuntimeError("stock CuTe Top-K threshold routing requires SM120")

    sink_start_block = sink_start // BLOCK_SIZE
    sink_end_block = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
    scale = head_dim**-0.5 if scale is None else float(scale)
    return _sol_topk_threshold_attn_cute(
        q,
        k,
        v,
        key_centroids,
        value_sums,
        threshold,
        route_mask,
        scale=scale,
        sink_start_block=sink_start_block,
        sink_end_block=sink_end_block,
        force_local_blocks=force_local_blocks,
    )


@triton.jit
def _forward_tma(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
    vc_desc,
    key_moments,
    route_mask,
    output_desc,
    scale,
    rms_approximation_beta,
    tokens,
    sink_start_block,
    sink_end_block,
    has_sink: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    blocks: tl.constexpr,
    candidate_blocks: tl.constexpr,
    rms_approximation: tl.constexpr,
    value_tile: tl.constexpr,
    block_size: tl.constexpr,
    route_group_size: tl.constexpr,
):
    value_tile_id, query_block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // heads, batch_head % heads
    group_offsets = tl.max_contiguous(
        tl.arange(0, route_group_size), route_group_size
    )
    token_offsets = tl.max_contiguous(tl.arange(0, block_size), block_size)
    query_start = query_block * block_size
    query = q_desc.load([batch, query_start, head, 0]).reshape(
        [block_size, head_dim]
    )

    output = tl.zeros([block_size, value_tile], dtype=tl.float32)
    row_sum = tl.zeros((block_size,), dtype=tl.float32)
    row_max = tl.full((block_size,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634

    for group_start in range(0, blocks, route_group_size):
        block_indices = group_start + group_offsets
        valid = block_indices < blocks
        key_centroid = kc_desc.load(
            [batch, group_start, head, 0]
        ).reshape([route_group_size, head_dim])
        value_sum = vc_desc.load(
            [batch, group_start, head, value_tile_id * value_tile]
        ).reshape([route_group_size, value_tile])
        scores = tl.dot(query, key_centroid.T).to(tl.float32) * scale_log2

        if rms_approximation:
            dims = tl.arange(0, head_dim)
            moment_offsets = (
                (
                    (batch * heads + head) * candidate_blocks
                    + block_indices[:, None]
                )
                * (2 * head_dim)
                + dims[None, :]
            )
            moment_valid = (
                (block_indices[:, None] < candidate_blocks)
                & (dims[None, :] < head_dim)
            )
            moment_pos = tl.load(
                key_moments + moment_offsets, mask=moment_valid, other=0.0
            )
            moment_neg = tl.load(
                key_moments + moment_offsets + head_dim,
                mask=moment_valid,
                other=0.0,
            )
            query_fp32 = query.to(tl.float32)
            query_pos = tl.maximum(query_fp32, 0.0)
            query_neg = tl.maximum(-query_fp32, 0.0)
            query_pos_sq = query_pos * query_pos
            query_neg_sq = query_neg * query_neg
            # BF16 tensor-core products match the storage/compute precision of
            # the surrounding Sol approximation and avoid an emulated IEEE
            # FP32 matrix product in every route group.
            directional_variance = tl.dot(
                query_pos_sq.to(query.dtype), moment_pos.to(query.dtype).T
            ).to(tl.float32) + tl.dot(
                query_neg_sq.to(query.dtype), moment_neg.to(query.dtype).T
            ).to(tl.float32)
            # Gaussian/Jensen log-mass correction in base-2 score units:
            # log E exp(alpha X) ~= 0.5 * alpha^2 * Var[X].
            scores += (
                0.5
                * rms_approximation_beta
                * scale
                * scale
                * 1.4426950408889634
                * tl.maximum(directional_variance, 0.0)
            )

        route_offsets = (
            ((batch * blocks + query_block) * heads + head) * blocks
            + block_indices
        )
        exact = tl.load(route_mask + route_offsets, mask=valid, other=0) != 0
        if has_sink:
            exact = exact | (
                (block_indices >= sink_start_block)
                & (block_indices < sink_end_block)
            )
        exact = exact & valid

        approximate = valid & ~exact
        has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
        approximate_scores = tl.where(
            approximate[None, :], scores, -float("inf")
        )
        safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
        candidate_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
        new_max = tl.where(has_approximate, candidate_max, row_max)
        alpha = tl.math.exp2(
            tl.where(has_approximate, row_max - new_max, 0.0)
        )
        probability = tl.math.exp2(
            safe_scores
            - tl.where(has_approximate, new_max, 0.0)[:, None]
        )
        probability = tl.where(
            has_approximate & approximate[None, :], probability, 0.0
        )
        output = output * alpha[:, None] + tl.dot(
            probability.to(value_sum.dtype), value_sum
        )
        block_lengths = tl.minimum(
            block_size,
            tl.maximum(0, tokens - block_indices * block_size),
        ).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(
            probability * block_lengths[None, :], axis=1
        )
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, route_group_size)
        exact_count = tl.sum(exact.to(tl.int32), axis=0)
        for _ in range(exact_count):
            offset = tl.min(exact_offsets)
            key_block = group_start + offset
            exact_offsets = tl.where(
                group_offsets == offset, route_group_size, exact_offsets
            )
            key_start = key_block * block_size
            key = k_desc.load([batch, key_start, head, 0]).reshape(
                [block_size, head_dim]
            )
            exact_scores = tl.dot(query, key.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(
                (key_start + token_offsets)[None, :] < tokens,
                0.0,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            value = v_desc.load(
                [batch, key_start, head, value_tile_id * value_tile]
            ).reshape([block_size, value_tile])
            output = output * alpha[:, None] + tl.dot(
                exact_probability.to(value.dtype), value
            )
            row_max = new_max

    output_desc.store(
        [batch, query_start, head, value_tile_id * value_tile],
        (output / row_sum[:, None]).to(tl.bfloat16)[None, :, None, :],
    )


def rope_sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    route_mask: torch.Tensor,
    *,
    key_moments: torch.Tensor | None = None,
    rms_approximation_beta: float = 0.0,
    scale: float | None = None,
    sink_start: int | None = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    """Run Sol's mixed approximate/exact attention with an explicit route."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must share shape [B, T, H, D]")
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise TypeError("rope_sol_attn requires BF16 inputs with head dimension 128")
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous BTHD tensors")
    batch, tokens, heads, head_dim = q.shape
    blocks = math.ceil(tokens / BLOCK_SIZE)
    expected_summaries = (batch, blocks, heads, head_dim)
    if key_centroids.shape != expected_summaries or value_sums.shape != expected_summaries:
        raise ValueError(
            "Kc/Vc shape mismatch: "
            f"expected {expected_summaries}, got {key_centroids.shape}/{value_sums.shape}"
        )
    expected_route = (batch, blocks, heads, blocks)
    if route_mask.shape != expected_route:
        raise ValueError(
            f"route_mask must have shape {expected_route}, got {route_mask.shape}"
        )
    if route_mask.dtype not in (torch.bool, torch.uint8) or not route_mask.is_contiguous():
        raise TypeError("route_mask must be a contiguous bool/uint8 tensor")
    if sink_start is None:
        sink_start = tokens - sink_tokens
    if not 0 <= sink_start <= tokens or not 0 <= sink_tokens <= tokens - sink_start:
        raise ValueError("invalid sink range")
    sink_start_block = sink_start // BLOCK_SIZE
    sink_end_block = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
    scale = head_dim**-0.5 if scale is None else float(scale)
    rms_approximation = key_moments is not None
    if rms_approximation:
        candidate_blocks = key_moments.shape[2]
        expected_moments = (batch, heads, candidate_blocks, 2 * head_dim)
        if (
            key_moments.shape != expected_moments
            or key_moments.dtype != torch.float32
            or key_moments.device != q.device
            or not key_moments.is_contiguous()
        ):
            raise TypeError(
                "key_moments must be contiguous CUDA FP32 [B,H,K,2D]"
            )
        if not math.isfinite(rms_approximation_beta) or rms_approximation_beta < 0:
            raise ValueError("rms_approximation_beta must be finite and non-negative")
    else:
        candidate_blocks = 0

    if not rms_approximation and rope_sol_backend(q.device) == "cute_sm120_explicit_route":
        return _rope_sol_attn_cute(
            q,
            k,
            v,
            key_centroids,
            value_sums,
            route_mask,
            scale=scale,
            sink_start_block=sink_start_block,
            sink_end_block=sink_end_block,
        )

    output = torch.empty_like(v)
    qkv_shape = [1, BLOCK_SIZE, 1, head_dim]
    summary_shape = [1, ROUTE_GROUP_SIZE, 1, head_dim]
    grid = (1, blocks, batch * heads)
    _forward_tma[grid](
        TensorDescriptor.from_tensor(q, qkv_shape),
        TensorDescriptor.from_tensor(k, qkv_shape),
        TensorDescriptor.from_tensor(v, qkv_shape),
        TensorDescriptor.from_tensor(key_centroids, summary_shape),
        TensorDescriptor.from_tensor(value_sums, summary_shape),
        key_centroids if key_moments is None else key_moments,
        route_mask,
        TensorDescriptor.from_tensor(output, qkv_shape),
        scale,
        rms_approximation_beta,
        tokens,
        sink_start_block,
        sink_end_block,
        sink_tokens > 0,
        heads,
        head_dim,
        blocks,
        candidate_blocks,
        rms_approximation,
        head_dim,
        BLOCK_SIZE,
        ROUTE_GROUP_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return output

