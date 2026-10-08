"""The route-QK-free packed-external path is the Spark default."""

import os
from types import SimpleNamespace

import pytest
import torch

from h3_sparse_attention import H3SparseAttentionConfig
from h3_sparse_attention.spark_reweight_sm120 import SparkReweightForwardSm120
from h3_sparse_attention.spark_reweight_sm80 import SparkReweightForwardSm80


def test_route_qk_free_selector_is_default():
    baseline = H3SparseAttentionConfig.spark(20)
    optimized = H3SparseAttentionConfig.spark(20, sol_route_topk_execution="threshold")
    assert baseline.sol_route_topk_execution == "packed_external_no_route_qk"
    assert optimized.sol_route_topk_execution == "threshold"
    assert optimized.landmark_tree_v2_children == baseline.landmark_tree_v2_children == 16
    with pytest.raises(ValueError, match="bitwise-equivalent but slower"):
        H3SparseAttentionConfig.spark(
            20, sol_route_topk_execution="packed_external"
        )


def test_route_qk_free_default_falls_back_outside_supported_gpus():
    from h3_sparse_attention.spark_integration import _effective_topk_execution

    requested = "packed_external_no_route_qk"
    assert _effective_topk_execution(requested, (12, 0)) == requested
    assert _effective_topk_execution(requested, (8, 0)) == requested
    assert _effective_topk_execution(requested, (9, 0)) == "threshold"
    assert _effective_topk_execution("threshold", (12, 0)) == "threshold"


def test_route_qk_free_kernel_disables_centroid_handoff():
    optimized = SparkReweightForwardSm120()
    assert optimized.external_route
    assert optimized.packed_external_route
    assert optimized.skip_external_route_qk
    assert not optimized.prefetch_next_route_k
    with pytest.raises(ValueError, match="packed external"):
        SparkReweightForwardSm120(packed_external_route=False)
    with pytest.raises(ValueError, match="route-QK-free"):
        SparkReweightForwardSm120(skip_external_route_qk=False)


def test_direct_packed_route_rejects_removed_route_qk_path():
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_attention

    q = torch.empty((1, 64, 1, 128), dtype=torch.bfloat16)
    route = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="route-QK-free"):
        virtual_q_attention(
            q,
            q,
            q,
            virtual_ranges=torch.tensor([[0, 64]], dtype=torch.int64),
            leaf_to_virtual=torch.zeros(1, dtype=torch.int64),
            route=route,
        )


def test_sm80_route_qk_free_kernel_is_packed_external_only():
    optimized = SparkReweightForwardSm80()
    assert optimized.external_route
    assert optimized.packed_external_route
    assert optimized.skip_external_route_qk
    with pytest.raises(ValueError, match="packed external"):
        SparkReweightForwardSm80(packed_external_route=False)
    with pytest.raises(NotImplementedError, match="only route-QK-free"):
        SparkReweightForwardSm80(
            packed_external_route=False,
            skip_external_route_qk=False,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(torch.cuda.get_device_capability(0)) != (8, 0),
    reason="SM80 packed external route requires Ampere SM80",
)
def test_sm80_route_qk_free_matches_fused_topk(monkeypatch):
    from sol_attn.preprocess import _reduce_kv
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_attention
    from h3_sparse_attention.sol_topk_cutoff import gemm_topk_packed_route

    generator = torch.Generator(device="cuda").manual_seed(1208)
    batch, tokens, heads, dim = 1, 576, 2, 128
    query_tokens, video_tokens, ratio = 512, 512, 0.25
    q, k, v = (
        torch.randn(
            batch, tokens, heads, dim, generator=generator,
            device="cuda", dtype=torch.float32,
        ).to(torch.bfloat16)
        for _ in range(3)
    )
    kc, vs = _reduce_kv(k, v)
    blocks = kc.shape[1]
    route, _ = gemm_topk_packed_route(
        q, kc, video_tokens=video_tokens, topk_ratio=ratio,
        query_tokens=query_tokens,
    )
    common = dict(
        virtual_ranges=torch.tensor(
            [[0, tokens]], device="cuda", dtype=torch.int64
        ),
        leaf_to_virtual=torch.zeros(
            blocks, device="cuda", dtype=torch.int64
        ),
        key_centroids=kc,
        value_sums=vs,
        sink_start=video_tokens,
        sink_tokens=tokens - video_tokens,
        force_local_blocks=False,
        _query_tokens=query_tokens,
    )
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    fused = virtual_q_attention(
        q, k, v, fused_topk_ratio=ratio, **common
    ).clone()
    external = virtual_q_attention(
        q, k, v, route=route, skip_external_route_qk=True, **common
    ).clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        external[:, :query_tokens].float(),
        fused[:, :query_tokens].float(),
        atol=0.002,
        rtol=0.01,
    )


@pytest.mark.skipif(
    os.environ.get("H3_RUN_KERNEL_PARITY") != "1" or not torch.cuda.is_available(),
    reason="run explicitly on an idle SM120 GPU with H3_RUN_KERNEL_PARITY=1",
)
def test_sm120_route_qk_free_repeats_bitwise():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 specialization")
    from h3_sparse_attention.processor import _Controller
    from h3_sparse_attention.spark_integration import _spark_topk_attention

    generator = torch.Generator(device="cuda").manual_seed(42)
    q, k, v = (
        torch.randn(1, 8256, 1, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    ranges = torch.tensor([[0, 8192], [8192, 8256]], device="cuda", dtype=torch.int64)
    mapping = torch.tensor([0] * 128 + [1], device="cuda", dtype=torch.int64)
    layout = SimpleNamespace(video_tokens=8192, sequence_length=8256)
    outputs = []
    for _ in range(2):
        execution = "packed_external_no_route_qk"
        controller = _Controller(H3SparseAttentionConfig.spark(
            20, sol_route_topk_execution=execution, sol_log_density=False,
        ))
        outputs.append(_spark_topk_attention(
            controller, q, k, v, layout,
            virtual_query_data=(ranges, mapping, None), _query_tokens=8192,
        ))
        assert controller.counts["sol_topk_packed_route_calls"] == 1
    assert torch.equal(outputs[0], outputs[1])
