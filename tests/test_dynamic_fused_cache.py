"""Fused SM80/SM120 callables reuse dynamic text extents and Top-K ratios."""

import pytest
import torch

from h3_sparse_attention.sol_numerator_virtual_q import (
    _dynamic_tensor_cache_signature,
)


def test_dynamic_signature_ignores_non_singleton_extents_and_compact_strides():
    first = torch.empty((1, 73560, 56, 128), dtype=torch.bfloat16)
    second = torch.empty((1, 73583, 56, 128), dtype=torch.bfloat16)

    assert _dynamic_tensor_cache_signature(first) == _dynamic_tensor_cache_signature(second)


def test_dynamic_signature_preserves_static_abi_properties():
    base = torch.empty((1, 128, 4, 128), dtype=torch.bfloat16)
    different_dtype = torch.empty((1, 128, 4, 128), dtype=torch.float32)
    different_order = torch.empty((1, 4, 128, 128), dtype=torch.bfloat16).permute(0, 2, 1, 3)
    different_singletons = torch.empty((1, 128, 1, 128), dtype=torch.bfloat16)

    signature = _dynamic_tensor_cache_signature(base)
    assert signature != _dynamic_tensor_cache_signature(different_dtype)
    assert signature != _dynamic_tensor_cache_signature(different_order)
    assert signature != _dynamic_tensor_cache_signature(different_singletons)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "execution",
    ("threshold", "fused", "packed_external_no_route_qk"),
)
def test_sm120_compiled_callable_reuses_different_sink_lengths(execution):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    import h3_sparse_attention.sol_numerator_virtual_q as fused

    torch.manual_seed(813)
    video_tokens = 8192

    def inputs(total_tokens):
        heads = 2
        blocks = (total_tokens + 63) // 64
        q, k, v = [
            torch.randn(
                1,
                total_tokens,
                heads,
                128,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        ]
        ranges = torch.tensor(
            [[0, video_tokens], [video_tokens, total_tokens]],
            device="cuda",
            dtype=torch.int64,
        )
        mapping = torch.tensor(
            [0] * (video_tokens // 64)
            + [1] * (blocks - video_tokens // 64),
            device="cuda",
            dtype=torch.int64,
        )
        anchors = fused.build_virtual_anchors(q, ranges)
        centroids = fused.reduce_virtual_key_centroids(k)
        threshold = (
            torch.zeros((1, blocks, heads), device="cuda", dtype=torch.float32)
            if execution == "threshold"
            else None
        )
        route = None
        if execution == "packed_external_no_route_qk":
            route = torch.full(
                (1, video_tokens // 64, heads, (blocks + 31) // 32),
                -1,
                device="cuda",
                dtype=torch.int32,
            )
        return q, k, v, anchors, ranges, mapping, centroids, threshold, route

    options = {
        "force_local_blocks": False,
        "_query_tokens": video_tokens,
        "fused_topk_ratio": 0.1 if execution == "fused" else 0.0,
        "skip_external_route_qk": execution == "packed_external_no_route_qk",
    }

    first = inputs(video_tokens + 64)
    second = inputs(video_tokens + 81)
    fused._FUSED_COMPILED.clear()
    fused._FUSED_COMPILE_CALLS = 0
    fused._FUSED_COMPILE_SECONDS = 0.0

    fused._fused_virtual(
        *first,
        video_tokens,
        64,
        **options,
    )
    reused = fused._fused_virtual(
        *second,
        video_tokens,
        81,
        **options,
    )[:, :video_tokens].clone()
    reused_stats = fused.fused_compile_cache_stats()

    assert reused_stats["entries"] == 1
    assert reused_stats["compile_calls"] == 1
    assert torch.isfinite(reused).all()

    fused._FUSED_COMPILED.clear()
    fresh = fused._fused_virtual(
        *second,
        video_tokens,
        81,
        **options,
    )[:, :video_tokens]
    torch.cuda.synchronize()
    assert torch.equal(reused, fresh)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sm120_compiled_callable_reuses_different_fused_topk_ratios(monkeypatch):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    import h3_sparse_attention.sol_numerator_virtual_q as fused

    monkeypatch.setenv("H3_SM120_RUNTIME_TOPK_RATIO", "1")

    torch.manual_seed(814)
    video_tokens = 8192
    sink_tokens = 64
    total_tokens = video_tokens + sink_tokens
    heads = 2
    blocks = (total_tokens + 63) // 64
    q, k, v = [
        torch.randn(
            1, total_tokens, heads, 128, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(3)
    ]
    ranges = torch.tensor(
        [[0, video_tokens], [video_tokens, total_tokens]],
        device="cuda",
        dtype=torch.int64,
    )
    mapping = torch.tensor(
        [0] * (video_tokens // 64) + [1] * (blocks - video_tokens // 64),
        device="cuda",
        dtype=torch.int64,
    )
    anchors = fused.build_virtual_anchors(q, ranges)
    centroids = fused.reduce_virtual_key_centroids(k)

    def run(ratio):
        return fused._fused_virtual(
            q, k, v, anchors, ranges, mapping, centroids, None, None,
            video_tokens, sink_tokens,
            force_local_blocks=False,
            _query_tokens=video_tokens,
            fused_topk_ratio=ratio,
        )[:, :video_tokens].clone()

    fused._FUSED_COMPILED.clear()
    fused._FUSED_COMPILE_CALLS = 0
    fused._FUSED_COMPILE_SECONDS = 0.0
    first = run(0.1)
    reused = run(0.2)
    reused_stats = fused.fused_compile_cache_stats()
    assert reused_stats["entries"] == 1
    assert reused_stats["compile_calls"] == 1
    assert not torch.equal(first, reused)

    fused._FUSED_COMPILED.clear()
    fresh = run(0.2)
    torch.cuda.synchronize()
    assert torch.equal(reused, fresh)

    monkeypatch.setenv("H3_SM120_RUNTIME_TOPK_RATIO", "0")
    fused._FUSED_COMPILED.clear()
    static = run(0.2)
    torch.cuda.synchronize()
    assert torch.equal(reused, static)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sm80_compiled_callable_reuses_text_lengths_and_topk_ratios(monkeypatch):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 required")

    import h3_sparse_attention.sol_numerator_virtual_q as fused

    monkeypatch.setenv("H3_SM80_RUNTIME_TOPK_RATIO", "1")
    torch.manual_seed(915)
    video_tokens = 512

    def inputs(sink_tokens):
        total_tokens = video_tokens + sink_tokens
        blocks = (total_tokens + 63) // 64
        q, k, v = [
            torch.randn(1, total_tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        ranges = torch.tensor(
            [[0, video_tokens], [video_tokens, total_tokens]],
            device="cuda", dtype=torch.int64,
        )
        mapping = torch.tensor(
            [0] * (video_tokens // 64) + [1] * (blocks - video_tokens // 64),
            device="cuda", dtype=torch.int64,
        )
        return (
            q, k, v, fused.build_virtual_anchors(q, ranges), ranges, mapping,
            fused.reduce_virtual_key_centroids(k),
        )

    def run(data, sink_tokens, ratio):
        return fused._fused_virtual(
            *data, None, None, video_tokens, sink_tokens,
            force_local_blocks=False, _query_tokens=video_tokens,
            fused_topk_ratio=ratio,
        )[:, :video_tokens].clone()

    shorter = inputs(64)
    longer = inputs(81)
    fused._FUSED_COMPILED.clear()
    fused._FUSED_COMPILE_CALLS = 0
    run(shorter, 64, 0.1)
    first_ratio = run(longer, 81, 0.1)
    second_ratio = run(longer, 81, 0.2)
    assert fused.fused_compile_cache_stats()["entries"] == 1
    assert fused.fused_compile_cache_stats()["compile_calls"] == 1
    assert not torch.equal(first_ratio, second_ratio)
    with pytest.raises(ValueError, match="fused_topk_ratio"):
        run(longer, 81, 1.01)

    fused._FUSED_COMPILED.clear()
    fresh = run(longer, 81, 0.2)
    monkeypatch.setenv("H3_SM80_RUNTIME_TOPK_RATIO", "0")
    fused._FUSED_COMPILED.clear()
    static = run(longer, 81, 0.2)
    torch.cuda.synchronize()
    assert torch.equal(second_ratio, fresh)
    assert torch.equal(second_ratio, static)
