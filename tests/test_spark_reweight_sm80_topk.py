"""Correctness coverage for SM80 CTA-local fused Top-K routing."""

import pytest
import torch


requires_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(torch.cuda.get_device_capability(0)) != (8, 0),
    reason="SM80 fused Top-K requires Ampere SM80",
)


@requires_sm80
@pytest.mark.parametrize("ratio,tied", [(0.1, False), (0.25, False), (0.5, False), (0.25, True)])
def test_sm80_cta_local_topk_matches_explicit_route(monkeypatch, ratio, tied):
    from sol_attn.preprocess import _reduce_kv
    from h3_sparse_attention.sol_numerator_virtual_q import (
        validate_virtual_layout,
        virtual_q_attention,
    )

    generator = torch.Generator(device="cuda").manual_seed(321 + int(ratio * 100))
    batch, tokens, heads, dim = 1, 576, 2, 128
    query_tokens, candidate_blocks = 512, 8
    if tied:
        q = torch.zeros(
            batch, tokens, heads, dim, device="cuda", dtype=torch.bfloat16
        )
    else:
        q = torch.randn(
            batch, tokens, heads, dim, generator=generator,
            device="cuda", dtype=torch.float32,
        ).to(torch.bfloat16)
    k, v = (
        torch.randn(
            batch, tokens, heads, dim, generator=generator,
            device="cuda", dtype=torch.float32,
        ).to(torch.bfloat16)
        for _ in range(2)
    )
    kc, vs = _reduce_kv(k, v)
    blocks = kc.shape[1]
    ranges_host, mapping_host = validate_virtual_layout(
        [[0, tokens]], [0] * blocks, tokens
    )
    ranges = torch.tensor(ranges_host, device="cuda", dtype=torch.int64)
    mapping = torch.tensor(mapping_host, device="cuda", dtype=torch.int64)

    qmean = q.reshape(batch, blocks, 64, heads, dim).float().mean(2)
    qmean = qmean.to(torch.bfloat16).float()
    scores = torch.einsum("bqhd,bkhd->bqhk", qmean, kc.float()) * dim**-0.5
    target = max(
        1,
        (candidate_blocks * round(ratio * 10000) + 5000) // 10000,
    )
    # stable=True matches the kernel's lower-block-index tie break.
    selected = torch.argsort(
        scores[..., :candidate_blocks], dim=-1, descending=True, stable=True
    )[..., :target]
    route = torch.zeros(
        batch, blocks, heads, blocks, device="cuda", dtype=torch.uint8
    )
    route.scatter_(-1, selected, 1)

    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    fused = virtual_q_attention(
        q, k, v,
        virtual_ranges=ranges,
        leaf_to_virtual=mapping,
        key_centroids=kc,
        value_sums=vs,
        sink_start=query_tokens,
        sink_tokens=64,
        force_local_blocks=False,
        fused_topk_ratio=ratio,
        _query_tokens=query_tokens,
    )
    torch.cuda.synchronize()

    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "0")
    reference = virtual_q_attention(
        q, k, v,
        virtual_ranges=ranges,
        leaf_to_virtual=mapping,
        key_centroids=kc,
        value_sums=vs,
        route=route,
        sink_start=query_tokens,
        sink_tokens=64,
        force_local_blocks=False,
        _query_tokens=query_tokens,
    )
    torch.cuda.synchronize()

    # Only the video prefix is launched; the context suffix is handled by the
    # caller's dense suffix and intentionally remains uninitialized here.
    assert torch.isfinite(fused[:, :query_tokens]).all()
    torch.testing.assert_close(
        fused[:, :query_tokens].float(),
        reference[:, :query_tokens].float(),
        atol=0.002,
        rtol=0.01,
    )


def test_sm80_fused_topk_ratio_validation():
    from h3_sparse_attention.spark_reweight_sm80 import SparkReweightForwardSm80

    with pytest.raises(ValueError, match="fused_topk_ratio"):
        SparkReweightForwardSm80(fused_topk_ratio=1.01)


@requires_sm80
def test_sm80_self_exact_fused_fastpath_matches_explicit_union(monkeypatch):
    from sol_attn.preprocess import _reduce_kv
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_attention

    generator = torch.Generator(device="cuda").manual_seed(906)
    batch, tokens, heads, dim = 1, 320, 1, 128
    query_tokens, candidate_blocks, ratio = 256, 4, 0.25
    q, k, v = (
        torch.randn(
            batch, tokens, heads, dim, generator=generator,
            device="cuda", dtype=torch.float32,
        ).to(torch.bfloat16)
        for _ in range(3)
    )
    kc, vs = _reduce_kv(k, v)
    blocks = kc.shape[1]
    ranges = torch.tensor([[0, tokens]], device="cuda", dtype=torch.int64)
    mapping = torch.zeros(blocks, device="cuda", dtype=torch.int64)

    qmean = q.reshape(batch, blocks, 64, heads, dim).float().mean(2)
    scores = torch.einsum(
        "bqhd,bkhd->bqhk", qmean.to(torch.bfloat16).float(), kc.float()
    ) * dim**-0.5
    selected = torch.argsort(
        scores[..., :candidate_blocks], dim=-1, descending=True, stable=True
    )[..., :1]
    route = torch.zeros(
        batch, blocks, heads, blocks, device="cuda", dtype=torch.uint8
    )
    route.scatter_(-1, selected, 1)
    route[:, :candidate_blocks, :, :candidate_blocks] |= torch.eye(
        candidate_blocks, device="cuda", dtype=torch.uint8
    )[None, :, None, :]

    common = dict(
        virtual_ranges=ranges,
        leaf_to_virtual=mapping,
        key_centroids=kc,
        value_sums=vs,
        sink_start=query_tokens,
        sink_tokens=64,
        _query_tokens=query_tokens,
    )
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    fused = virtual_q_attention(
        q, k, v, force_local_blocks=0, fused_topk_ratio=ratio, **common
    )
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "0")
    reference = virtual_q_attention(
        q, k, v, route=route, force_local_blocks=False, **common
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        fused[:, :query_tokens].float(),
        reference[:, :query_tokens].float(),
        atol=0.002,
        rtol=0.01,
    )


@requires_sm80
def test_sm80_direct_summary_layout_matches_legacy_copy(monkeypatch):
    from sol_attn.preprocess import _reduce_kv
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_attention

    generator = torch.Generator(device="cuda").manual_seed(812)
    batch, tokens, heads, dim = 1, 576, 3, 128
    q, k, v = (
        torch.randn(
            batch, tokens, heads, dim, generator=generator,
            device="cuda", dtype=torch.float32,
        ).to(torch.bfloat16)
        for _ in range(3)
    )
    kc, vs = _reduce_kv(k, v)
    blocks = kc.shape[1]
    ranges = torch.tensor([[0, tokens]], device="cuda", dtype=torch.int64)
    mapping = torch.zeros(blocks, device="cuda", dtype=torch.int64)
    kwargs = dict(
        virtual_ranges=ranges,
        leaf_to_virtual=mapping,
        key_centroids=kc,
        value_sums=vs,
        sink_start=512,
        sink_tokens=64,
        force_local_blocks=False,
        fused_topk_ratio=0.1,
        _query_tokens=512,
    )
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    monkeypatch.setenv("H3_SM80_DIRECT_SUMMARIES", "0")
    copied = virtual_q_attention(q, k, v, **kwargs)
    monkeypatch.setenv("H3_SM80_DIRECT_SUMMARIES", "1")
    direct = virtual_q_attention(q, k, v, **kwargs)
    torch.cuda.synchronize()

    assert torch.equal(direct[:, :512], copied[:, :512])


@requires_sm80
def test_sm80_speculative_summary_prefetch_all_exact(monkeypatch):
    """An unused speculative summary copy must drain before exact K/V reuse."""
    from sol_attn.preprocess import _reduce_kv
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_attention

    generator = torch.Generator(device="cuda").manual_seed(913)
    batch, tokens, heads, dim = 1, 192, 2, 128
    q, k, v = (
        torch.randn(
            batch, tokens, heads, dim, generator=generator,
            device="cuda", dtype=torch.float32,
        ).to(torch.bfloat16)
        for _ in range(3)
    )
    kc, vs = _reduce_kv(k, v)
    blocks = kc.shape[1]
    kwargs = dict(
        virtual_ranges=torch.tensor([[0, tokens]], device="cuda", dtype=torch.int64),
        leaf_to_virtual=torch.zeros(blocks, device="cuda", dtype=torch.int64),
        key_centroids=kc,
        value_sums=vs,
        sink_start=128,
        sink_tokens=64,
        force_local_blocks=False,
        fused_topk_ratio=1.0,
        _query_tokens=128,
    )
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    monkeypatch.setenv("H3_SM80_PREFETCH_SUMMARY", "0")
    monkeypatch.setenv("H3_SM80_SKIP_FINAL_TILE_BARRIER", "0")
    demand_loaded = virtual_q_attention(q, k, v, **kwargs).clone()
    monkeypatch.setenv("H3_SM80_PREFETCH_SUMMARY", "1")
    monkeypatch.setenv("H3_SM80_SKIP_FINAL_TILE_BARRIER", "1")
    prefetched = virtual_q_attention(q, k, v, **kwargs).clone()
    torch.cuda.synchronize()

    assert torch.equal(prefetched[:, :128], demand_loaded[:, :128])
