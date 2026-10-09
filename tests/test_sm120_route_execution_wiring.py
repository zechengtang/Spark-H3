"""Hardware-independent wiring checks for fused route executions."""
from types import SimpleNamespace

import pytest
import torch

from h3_sparse_attention import H3SparseAttentionConfig
from h3_sparse_attention.processor import _Controller


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor fixture required")
@pytest.mark.parametrize(
    ("execution", "expected_ratio", "expected_skip", "expects_route"),
    [
        ("fused", 0.25, False, False),
        ("packed_external_no_route_qk", 0.0, True, True),
    ],
)
@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (12, 0)])
def test_sm120_route_execution_reaches_fused_kernel_contract(
    monkeypatch, execution, expected_ratio, expected_skip, expects_route, capability
):
    """The production dispatcher must preserve each supported route contract."""
    from h3_sparse_attention import rope_sol_kernel, sol_numerator_virtual_q
    from h3_sparse_attention import sol_topk_cutoff, spark_integration

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(
        rope_sol_kernel, "sol_topk_threshold_backend",
        lambda device: f"cute_sm{capability[0]}{capability[1]}_topk_threshold",
    )

    q = torch.zeros((1, 128, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    kc = torch.zeros((1, 2, 1, 128), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(sol_numerator_virtual_q, "virtual_q_backend",
                        lambda tensor: f"sm{capability[0]}{capability[1]}_fused_virtual_query")
    monkeypatch.setattr(sol_numerator_virtual_q, "reduce_virtual_key_centroids",
                        lambda tensor: kc)

    packed_route = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.int32)
    route_calls = []

    def packed(*args, **kwargs):
        route_calls.append((args, kwargs))
        return packed_route, {
            "candidate_video_blocks": 1,
            "query_video_blocks": 1,
            "target_topk_blocks_per_query": 1,
        }

    monkeypatch.setattr(sol_topk_cutoff, "gemm_topk_packed_route", packed)
    kernel_calls = []

    def virtual(*args, **kwargs):
        kernel_calls.append((args, kwargs))
        return torch.zeros_like(q)

    monkeypatch.setattr(sol_numerator_virtual_q, "virtual_q_attention", virtual)
    config = H3SparseAttentionConfig.spark(
        20,
        sol_route_topk_execution=execution,
        sol_route_topk_ratio=0.25,
        sol_log_density=False,
    )
    controller = _Controller(config)
    ranges = torch.tensor([[0, 64], [64, 128]], device="cuda", dtype=torch.int64)
    mapping = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    layout = SimpleNamespace(video_tokens=64, sequence_length=128)

    output = spark_integration._spark_topk_attention(
        controller, q, k, v, layout,
        virtual_query_data=(ranges, mapping, None), _query_tokens=64,
    )

    assert output.shape == q.shape
    assert len(kernel_calls) == 1
    kwargs = kernel_calls[0][1]
    assert kwargs["fused_topk_ratio"] == expected_ratio
    assert kwargs["skip_external_route_qk"] is expected_skip
    assert (kwargs["route"] is packed_route) is expects_route
    assert len(route_calls) == int(expects_route)
    if execution == "fused":
        assert controller.counts["sol_topk_fused_route_calls"] == 1
    else:
        assert controller.counts["sol_topk_packed_route_calls"] == 1
