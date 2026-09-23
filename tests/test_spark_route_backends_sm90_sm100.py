"""SM90/SM100 route-mask backend tests.

The wiring tests run everywhere (backend probing is signature-only and never
constructs a kernel). The numerical tests are gated on SM90/SM100 hardware and
compare the CuTe route-mask paths against a torch-native reference that
reimplements Sol's mixed approximate/exact semantics.
"""
from __future__ import annotations

import math

import pytest
import torch

BLOCK = 64

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
requires_route_backend = pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(torch.cuda.get_device_capability(0)) not in ((9, 0), (10, 0)),
    reason="route-mask CuTe backends require SM90/SM100 hardware",
)


def _sink_blocks(tokens, sink_start, sink_tokens):
    sink_start = tokens - sink_tokens if sink_start is None else sink_start
    return sink_start // BLOCK, math.ceil((sink_start + sink_tokens) / BLOCK)


def _route_decision(
    col_mean, threshold, route_row, q_block, kv_blocks, sink_range, force_local_blocks
):
    """Replicate the kernel's per-column exact decision for one query block."""
    if route_row is not None and (threshold is None or threshold != threshold):
        exact = route_row.bool().clone()
        if force_local_blocks and threshold is not None:
            exact |= (kv_blocks - q_block).abs() <= 1
    else:
        exact = col_mean > threshold
        if force_local_blocks:
            exact |= (kv_blocks - q_block).abs() <= 1
    sink_start_block, sink_end_block = sink_range
    exact |= (kv_blocks >= sink_start_block) & (kv_blocks < sink_end_block)
    return exact


def _torch_mixed_reference(
    q, k, v, kc, vs, threshold, route, *, sink_start, sink_tokens, force_local_blocks
):
    """Torch reference for Sol's mixed approximate/exact attention.

    Approximate blocks contribute exp2(q.kc) * value_sum with a block-length
    mass multiplier; exact blocks contribute per-token exp2(q.k) * v.
    """
    b, t, h, d = q.shape
    n = kc.shape[1]
    scale = d**-0.5 * math.log2(math.e)
    sink_range = _sink_blocks(t, sink_start, sink_tokens)
    kv_blocks = torch.arange(n, device=q.device)
    block_lens = torch.clamp(t - kv_blocks * BLOCK, max=BLOCK).float()
    out = torch.zeros(b, t, h, d, dtype=torch.float32, device=q.device)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    kcf = kc.float()
    vsf = vs.float()
    for bi in range(b):
        for hi in range(h):
            for qb in range(math.ceil(t / BLOCK)):
                rows = torch.arange(qb * BLOCK, min(t, (qb + 1) * BLOCK), device=q.device)
                qrows = qf[bi, rows, hi]
                col_score = (qrows @ kcf[bi, :, hi].T) * scale
                col_mean = col_score.mean(0)
                thr = None if threshold is None else float(threshold[bi, qb, hi])
                route_row = None if route is None else route[bi, qb, hi]
                exact = _route_decision(
                    col_mean, thr, route_row, qb, kv_blocks, sink_range, force_local_blocks
                )
                numerator = torch.zeros(len(rows), d, dtype=torch.float32, device=q.device)
                denominator = torch.zeros(len(rows), dtype=torch.float32, device=q.device)
                for kvb in range(n):
                    if exact[kvb]:
                        token_idx = torch.arange(kvb * BLOCK, min(t, (kvb + 1) * BLOCK), device=q.device)
                        scores = (qrows @ kf[bi, token_idx, hi].T) * scale
                        probs = torch.exp2(scores)
                        numerator += probs @ vf[bi, token_idx, hi]
                        denominator += probs.sum(-1)
                    else:
                        probs = torch.exp2(col_score[:, kvb])
                        numerator += probs[:, None] * vsf[bi, kvb, hi][None, :]
                        denominator += probs * block_lens[kvb]
                out[bi, rows, hi] = numerator / denominator[:, None].clamp_min(1e-30)
                out[bi, rows, hi] = torch.where(
                    (denominator > 0)[:, None], out[bi, rows, hi], torch.zeros_like(out[bi, rows, hi])
                )
    return out


def _torch_exact_reference(q, k, v, route, *, sink_start, sink_tokens):
    """Exact-only attention over route-selected blocks plus sink blocks."""
    b, t, h, d = q.shape
    scale = d**-0.5
    sink_start_block, sink_end_block = _sink_blocks(t, sink_start, sink_tokens)
    out = torch.zeros(b, t, h, d, dtype=torch.float32, device=q.device)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    for bi in range(b):
        for hi in range(h):
            block_mask = route[bi, :, hi].bool().clone()
            if sink_tokens:
                block_mask[:, sink_start_block:sink_end_block] = True
            for qb in range(math.ceil(t / BLOCK)):
                rows = torch.arange(qb * BLOCK, min(t, (qb + 1) * BLOCK), device=q.device)
                selected = block_mask[qb]
                token_idx = torch.cat(
                    [
                        torch.arange(kvb * BLOCK, min(t, (kvb + 1) * BLOCK), device=q.device)
                        for kvb in torch.nonzero(selected).flatten().tolist()
                    ]
                ) if selected.any() else torch.zeros(0, dtype=torch.long, device=q.device)
                if token_idx.numel() == 0:
                    continue
                scores = (qf[bi, rows, hi] @ kf[bi, token_idx, hi].T) * scale
                probs = torch.softmax(scores, dim=-1)
                out[bi, rows, hi] = probs @ vf[bi, token_idx, hi]
    return out


def _inputs(b=2, t=256, h=4, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (
        torch.randn(b, t, h, d, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        for _ in range(3)
    )
    from sol_attn.preprocess import _reduce_kv

    kc, vs = _reduce_kv(k, v)
    return q, k, v, kc, vs


# --- wiring tests (run on any arch) ---------------------------------------


@requires_cuda
def test_threshold_backend_probe_sm90_sm100(monkeypatch):
    from h3_sparse_attention import rope_sol_kernel

    real = torch.cuda.get_device_capability
    for cap, expected in (
        ((9, 0), "cute_sm90_topk_threshold"),
        ((10, 0), "cute_sm100_topk_threshold"),
        ((12, 0), "cute_sm120_topk_threshold"),
        ((8, 0), None),
    ):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, _c=cap, **k: _c)
        assert rope_sol_kernel.sol_topk_threshold_backend(0) == expected
    monkeypatch.setattr(torch.cuda, "get_device_capability", real)


@requires_cuda
def test_vaware_cute_capability_set():
    from h3_sparse_attention.sol_vaware_compensation import _CUTE_CAPS

    assert (9, 0) in _CUTE_CAPS and (10, 0) in _CUTE_CAPS and (12, 0) in _CUTE_CAPS


# --- numerical tests (SM90/SM100 hardware only) ----------------------------


@requires_route_backend
@pytest.mark.parametrize("force_local_blocks", [True, False])
def test_topk_threshold_matches_torch_reference(force_local_blocks):
    from h3_sparse_attention.rope_sol_kernel import (
        sol_topk_threshold_attn,
        sol_topk_threshold_backend,
    )

    assert sol_topk_threshold_backend(0) is not None
    q, k, v, kc, vs = _inputs()
    b, t, h, _ = q.shape
    n = kc.shape[1]
    g = torch.Generator(device="cuda").manual_seed(1)
    threshold = torch.randn(b, n, h, generator=g, device="cuda", dtype=torch.float32) * 0.5
    sink_tokens = 64
    out = sol_topk_threshold_attn(
        q, k, v, kc, vs, threshold,
        sink_tokens=sink_tokens, force_local_blocks=force_local_blocks,
    )
    ref = _torch_mixed_reference(
        q, k, v, kc, vs, threshold, None,
        sink_start=None, sink_tokens=sink_tokens,
        force_local_blocks=force_local_blocks,
    )
    torch.testing.assert_close(out.float(), ref, atol=0.008, rtol=0.015)


@requires_route_backend
def test_topk_threshold_hybrid_nan_rows():
    from h3_sparse_attention.rope_sol_kernel import sol_topk_threshold_attn

    q, k, v, kc, vs = _inputs()
    b, t, h, _ = q.shape
    n = kc.shape[1]
    g = torch.Generator(device="cuda").manual_seed(2)
    threshold = torch.randn(b, n, h, generator=g, device="cuda", dtype=torch.float32) * 0.5
    route = (
        torch.rand(b, n, h, n, generator=g, device="cuda") > 0.5
    ).to(torch.uint8)
    # NaN cutoffs mark numerically unsafe rows; those rows follow the mask.
    threshold[:, 1, :] = float("nan")
    out = sol_topk_threshold_attn(
        q, k, v, kc, vs, threshold, route,
        sink_tokens=0, force_local_blocks=False,
    )
    ref = _torch_mixed_reference(
        q, k, v, kc, vs, threshold, route,
        sink_start=None, sink_tokens=0, force_local_blocks=False,
    )
    torch.testing.assert_close(out.float(), ref, atol=0.008, rtol=0.015)


@requires_route_backend
def test_vaware_external_route_matches_torch_reference():
    from h3_sparse_attention.sol_vaware_compensation import exact_attention

    q, k, v, kc, vs = _inputs()
    b, t, h, _ = q.shape
    n = kc.shape[1]
    g = torch.Generator(device="cuda").manual_seed(3)
    route = (
        torch.rand(b, n, h, n, generator=g, device="cuda") > 0.5
    ).to(torch.uint8)
    route[:, 0, :, 0] = 1  # avoid empty selections
    out, _, actual = exact_attention(
        q, k, v, kc, vs, route=route, sink_tokens=64,
    )
    ref = _torch_exact_reference(q, k, v, route, sink_start=None, sink_tokens=64)
    torch.testing.assert_close(out.float(), ref.to(out.dtype).float(), atol=0.008, rtol=0.015)
    expected = route.clone()
    sink_start_block, sink_end_block = _sink_blocks(t, None, 64)
    expected[:, :, :, sink_start_block:sink_end_block] = 1
    assert torch.equal(actual, expected)


@requires_route_backend
@pytest.mark.parametrize("force_local_blocks", [True, False])
def test_vaware_export_route_recovers_threshold_decision(force_local_blocks):
    from h3_sparse_attention.sol_vaware_compensation import exact_attention

    q, k, v, kc, vs = _inputs()
    b, t, h, _ = q.shape
    n = kc.shape[1]
    g = torch.Generator(device="cuda").manual_seed(4)
    threshold = torch.randn(b, n, h, generator=g, device="cuda", dtype=torch.float32) * 0.5
    sink_tokens = 64
    out, _, actual = exact_attention(
        q, k, v, kc, vs, threshold=threshold, sink_tokens=sink_tokens,
        force_local_blocks=force_local_blocks,
    )
    ref = _torch_mixed_reference(
        q, k, v, kc, vs, threshold, None,
        sink_start=None, sink_tokens=sink_tokens,
        force_local_blocks=force_local_blocks,
    )
    torch.testing.assert_close(out.float(), ref, atol=0.008, rtol=0.015)

    scale_log2e = q.shape[-1] ** -0.5 * math.log2(math.e)
    sink_range = _sink_blocks(t, None, sink_tokens)
    kv_blocks = torch.arange(n, device=q.device)
    expected = torch.zeros(b, n, h, n, dtype=torch.uint8, device=q.device)
    for qb in range(n):
        rows = torch.arange(qb * BLOCK, min(t, (qb + 1) * BLOCK), device=q.device)
        col_mean = (
            (q.float()[:, rows].permute(0, 2, 1, 3) @ kc.float().permute(0, 2, 3, 1))
            * scale_log2e
        ).mean(2)  # [B, H, N]
        exact = col_mean > threshold[:, qb, :, None]
        if force_local_blocks:
            exact |= (kv_blocks - qb).abs() <= 1
        exact[:, :, (kv_blocks >= sink_range[0]) & (kv_blocks < sink_range[1])] = True
        expected[:, qb] = exact.to(torch.uint8)
    assert torch.equal(actual, expected)


@requires_route_backend
def test_official_force_local_blocks_false_matches_triton():
    import sol_attn.interface as interface
    from sol_attn import sol_attn

    q, k, v = (
        torch.randn(2, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    cute_out = sol_attn(q, k, v, tau=0.5, force_local_blocks=False)
    original = interface._cute_runtime_available
    interface._cute_runtime_available.cache_clear()
    try:
        interface._cute_runtime_available = lambda: False
        triton_out = sol_attn(q, k, v, tau=0.5, force_local_blocks=False)
    finally:
        interface._cute_runtime_available = original
    torch.testing.assert_close(cute_out.float(), triton_out.float(), atol=0.008, rtol=0.015)
