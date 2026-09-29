"""The route-QK-free SM120 path must remain opt-in."""

import os
from types import SimpleNamespace

import pytest
import torch

from h3_sparse_attention import H3SparseAttentionConfig
from h3_sparse_attention.spark_reweight_sm120 import SparkReweightForwardSm120


def test_route_qk_free_selector_is_opt_in():
    baseline = H3SparseAttentionConfig.spark(20, sol_route_topk_execution="packed_external")
    optimized = H3SparseAttentionConfig.spark(
        20, sol_route_topk_execution="packed_external_no_route_qk"
    )
    assert baseline.sol_route_topk_execution == "packed_external"
    assert optimized.sol_route_topk_execution == "packed_external_no_route_qk"
    assert optimized.landmark_tree_v2_children == baseline.landmark_tree_v2_children == 16


def test_route_qk_free_kernel_disables_centroid_handoff():
    baseline = SparkReweightForwardSm120(packed_external_route=True)
    optimized = SparkReweightForwardSm120(
        packed_external_route=True, skip_external_route_qk=True
    )
    assert not baseline.skip_external_route_qk
    assert baseline.prefetch_next_route_k
    assert optimized.skip_external_route_qk
    assert not optimized.prefetch_next_route_k
    with pytest.raises(ValueError, match="packed external"):
        SparkReweightForwardSm120(skip_external_route_qk=True)


@pytest.mark.skipif(
    os.environ.get("H3_RUN_KERNEL_PARITY") != "1" or not torch.cuda.is_available(),
    reason="run explicitly on an idle SM120 GPU with H3_RUN_KERNEL_PARITY=1",
)
def test_route_qk_free_matches_existing_packed_external():
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
    for execution in ("packed_external", "packed_external_no_route_qk"):
        controller = _Controller(H3SparseAttentionConfig.spark(
            20, sol_route_topk_execution=execution, sol_log_density=False,
        ))
        outputs.append(_spark_topk_attention(
            controller, q, k, v, layout,
            virtual_query_data=(ranges, mapping, None), _query_tokens=8192,
        ))
        assert controller.counts["sol_topk_packed_route_calls"] == 1
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
