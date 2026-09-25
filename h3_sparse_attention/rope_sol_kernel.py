"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import inspect


import math
import operator


import torch


BLOCK_SIZE = 64


_CUTE_THRESHOLD_COMPILED = {}


def sol_topk_threshold_backend(
    device: torch.device | str | int | None = None,
) -> str | None:
    """Return the stock-threshold backend used by fixed-ratio SOL.

    The cutoff path targets the CuTe mainloops with hybrid-route support
    (SM90/SM100/SM120). Other architectures return None and the caller falls
    back to explicit-route attention via sol_vaware_compensation.exact_attention.
    """

    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability == (12, 0):
        try:
            from sol_attn.sm120 import make_kernel

            make_kernel(external_route=False, hybrid_route=True, force_local_blocks=False)
        except (ImportError, TypeError):
            return None
        return "cute_sm120_topk_threshold"
    if capability in ((9, 0), (10, 0)):
        try:
            if capability == (9, 0):
                from sol_attn.sm90 import make_kernel
            else:
                from sol_attn.sm100 import make_kernel
        except ImportError:
            return None
        parameters = inspect.signature(make_kernel).parameters
        if "hybrid_route" not in parameters or "force_local_blocks" not in parameters:
            return None
        return f"cute_sm{capability[0]}{capability[1]}_topk_threshold"
    return None


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
    """Invoke the stock CuTe route selector with caller cutoffs."""

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from sol_attn.common import to_cute_tensor

    capability = tuple(torch.cuda.get_device_capability(q.device))
    batch, query_tokens, heads, _ = q.shape
    tokens = k.shape[1]
    output = torch.empty_like(q)
    lse = torch.empty(
        (batch, query_tokens, heads), device=q.device, dtype=torch.float32
    )
    hybrid_route = route_mask is not None
    # The route-mask ABI is uniform across backends. Pure stock mode never
    # dereferences this placeholder; hybrid mode reads only rows whose
    # cutoff is NaN.
    route_argument = (
        torch.empty((1,), device=q.device, dtype=torch.uint8)
        if route_mask is None
        else route_mask
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        q.device.index, capability, batch, tokens, query_tokens, heads,
        hybrid_route, force_local_blocks,
    )
    compiled = _CUTE_THRESHOLD_COMPILED.get(key)
    if capability == (9, 0):
        from sol_attn.sm90 import make_kernel

        # SM90 packs the sink block range into one Int32; an empty range
        # (start == end) is the disabled sentinel.
        sink_range = (
            0
            if sink_start_block == sink_end_block
            else sink_start_block | (sink_end_block << 16)
        )
        tensors = [q, k, v, output, key_centroids, value_sums, threshold, lse]
        args = [to_cute_tensor(tensor) for tensor in tensors]
        route_arg = to_cute_tensor(route_argument)
        if compiled is None:
            operator = make_kernel(
                tokens,
                1,
                external_route=False,
                hybrid_route=hybrid_route,
                force_local_blocks=force_local_blocks,
            )
            compiled = cute.compile(
                operator,
                *args,
                scale,
                sink_range,
                stream=stream,
                mRouteMask=route_arg,
                options="--enable-tvm-ffi",
            )
            _CUTE_THRESHOLD_COMPILED[key] = compiled
        compiled(*args, scale, sink_range, stream=stream, mRouteMask=route_arg)
        return output
    if capability == (10, 0):
        from sol_attn.sm100 import make_kernel
    else:
        from sol_attn.sm120 import make_kernel
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
    _query_tokens: int | None = None,
) -> torch.Tensor:
    """Run fixed-ratio SOL through the stock CuTe threshold selector."""

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
    query_tokens = tokens if _query_tokens is None else operator.index(_query_tokens)
    if not 0 < query_tokens <= tokens or (query_tokens != tokens and query_tokens % BLOCK_SIZE):
        raise ValueError("_query_tokens must end at a physical query-block boundary")
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
        raise RuntimeError(
            "stock CuTe Top-K threshold routing requires an SM90/SM100/SM120 backend"
        )

    sink_start_block = sink_start // BLOCK_SIZE
    sink_end_block = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
    scale = head_dim**-0.5 if scale is None else float(scale)
    q_kernel = q if query_tokens == tokens else q[:, :query_tokens].contiguous()
    output_kernel = _sol_topk_threshold_attn_cute(
        q_kernel,
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
    if query_tokens == tokens:
        return output_kernel
    output = torch.empty_like(q)
    output[:, :query_tokens].copy_(output_kernel)
    return output
