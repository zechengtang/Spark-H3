"""Diffusers Spark's optional query-block-shared approximate branch."""

import pytest
import torch
from types import SimpleNamespace

from h3_sparse_attention.processor import H3SparseAttentionConfig


def test_block_tail_configuration_is_opt_in():
    assert H3SparseAttentionConfig.spark(20).sol_tail_granularity == "query"
    assert H3SparseAttentionConfig.spark(
        20, sol_tail_granularity="block"
    ).sol_tail_granularity == "block"
    with pytest.raises(ValueError, match="sol_tail_granularity"):
        H3SparseAttentionConfig.spark(20, sol_tail_granularity="invalid")
    with pytest.raises(ValueError, match="virtual query summaries"):
        H3SparseAttentionConfig.sol(20, sol_tail_granularity="block")
    with pytest.raises(ValueError, match="threshold or packed_external_no_route_qk"):
        H3SparseAttentionConfig.spark(
            20, sol_tail_granularity="block", sol_route_topk_execution="fused"
        )
    assert H3SparseAttentionConfig.spark(
        20, sol_tail_granularity="block8x8"
    ).sol_tail_granularity == "block8x8"
    with pytest.raises(ValueError, match="threshold or packed_external_no_route_qk"):
        H3SparseAttentionConfig.spark(
            20, sol_tail_granularity="block8x8", sol_route_topk_execution="fused"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block8x8_refines_only_skipped_tail():
    from h3_sparse_attention.sol_numerator_virtual_q import (
        _summaries_block8, _skip_merge_chunk_block8,
    )

    gen = torch.Generator(device="cuda").manual_seed(420)
    q, k, v = (
        torch.randn(1, 128, 1, 128, generator=gen, device="cuda", dtype=torch.bfloat16) * .1
        for _ in range(3)
    )
    anchor = q.float().mean(dim=1, keepdim=True).to(torch.bfloat16).contiguous()
    ak = torch.empty(1, 1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    av = torch.empty_like(ak)
    lm = torch.empty(1, 1, 1, 16, device="cuda", dtype=torch.float32)
    _summaries_block8[(1, 16, 1)](
        anchor, k, v, ak, av, lm, 128, 1, 16, 1,
        anchor.stride(0), anchor.stride(1), 0, 1, 1, 0,
        False, True, True, num_warps=4,
    )
    route = torch.tensor([[[[1, 0]], [[0, 1]]]], device="cuda", dtype=torch.uint8)
    mapping = torch.zeros(2, device="cuda", dtype=torch.int64)
    exact = torch.randn(1, 128, 1, 128, generator=gen, device="cuda", dtype=torch.bfloat16)
    lse = torch.randn(1, 128, 1, generator=gen, device="cuda", dtype=torch.float32)
    actual = exact.clone()
    _skip_merge_chunk_block8[(2, 8, 1)](
        q, mapping, ak, av, lm, route, actual, lse,
        128, 1, 2, 16, 1, 0, 0, num_warps=4, num_stages=1,
    )
    expected = exact.float().clone()
    for qb in range(2):
        for micro in range(8):
            rows = slice(qb * 64 + micro * 8, qb * 64 + (micro + 1) * 8)
            keys = [key for key in range(16) if route[0, qb, 0, key // 8] == 0]
            qm = q[0, rows, 0].float().mean(0).to(torch.bfloat16).float()
            scores = (ak[0, 0, 0, keys].float() * qm).sum(-1) * (128 ** -.5) + lm[0, 0, 0, keys]
            weights = torch.softmax(scores, 0)
            tail = (weights[:, None] * av[0, 0, 0, keys].float()).sum(0)
            tail_lse = torch.logsumexp(scores, 0)
            ew = torch.exp(lse[0, rows, 0] - torch.logaddexp(lse[0, rows, 0], tail_lse))
            aw = torch.exp(tail_lse - torch.logaddexp(lse[0, rows, 0], tail_lse))
            expected[0, rows, 0] = ew[:, None] * exact[0, rows, 0].float() + aw[:, None] * tail
    torch.testing.assert_close(actual.float(), expected.to(torch.bfloat16).float(), atol=.02, rtol=.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_route_expands_query_prefix_without_changing_bits():
    from h3_sparse_attention.sol_numerator_virtual_q import unpack_packed_route

    packed = torch.tensor(
        [[[[0b101], [0b010]], [[0b110], [0b001]]]],
        device="cuda", dtype=torch.int32,
    )
    dense = unpack_packed_route(packed, 3)
    assert dense.shape == (1, 3, 2, 3)
    assert dense.cpu().tolist() == [[
        [[1, 0, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
    ]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_tail_shares_approximation_but_keeps_row_exact_output():
    from h3_sparse_attention.sol_numerator_virtual_q import _skip_merge_chunk_block

    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(1, 192, 1, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
    ak = torch.randn(1, 1, 1, 3, 128, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    av = torch.randn(1, 1, 1, 3, 128, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.1
    lm = torch.tensor([[[[0.1, -0.2, 0.3]]]], device="cuda", dtype=torch.float32)
    mapping = torch.zeros(3, device="cuda", dtype=torch.int64)
    route = torch.eye(3, device="cuda", dtype=torch.uint8).view(1, 3, 1, 3)
    exact = torch.randn(1, 192, 1, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
    exact_lse = torch.randn(1, 192, 1, generator=generator, device="cuda", dtype=torch.float32)
    result = exact.clone()
    _skip_merge_chunk_block[(3, 1)](
        q, mapping, ak, av, lm, route, result, exact_lse,
        192, 1, 3, 1, 0, 0, num_warps=4, num_stages=1,
    )

    expected = exact.float().clone()
    for block in range(3):
        rows = slice(block * 64, (block + 1) * 64)
        qmean = q[:, rows, 0].float().mean(dim=1).to(torch.bfloat16).float()
        scores = (qmean[:, None] * ak[:, 0, 0].float()).sum(-1) * (128 ** -0.5) + lm[:, 0, 0]
        scores[:, block] = -torch.inf
        approximate = torch.softmax(scores, dim=-1) @ av[0, 0, 0].float()
        approximate_lse = torch.logsumexp(scores, dim=-1)
        exact_row_lse = exact_lse[:, rows, 0]
        maximum = torch.maximum(exact_row_lse, approximate_lse[:, None])
        exact_weight = (exact_row_lse - maximum).exp()
        approx_weight = (approximate_lse[:, None] - maximum).exp()
        expected[:, rows, 0] = (
            exact_weight[:, :, None] * exact[:, rows, 0].float()
            + approx_weight[:, :, None] * approximate[:, None]
        ) / (exact_weight + approx_weight)[:, :, None]
    torch.testing.assert_close(result.float(), expected.to(torch.bfloat16).float(), atol=0.02, rtol=0.02)
    # All rows in a block share the approximate value, but retain distinct exact states.
    assert not torch.equal(result[:, 0], result[:, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_tail_runs_with_production_packed_topk_route():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("packed external route requires SM120")
    from h3_sparse_attention.processor import _Controller
    from h3_sparse_attention.spark_integration import _spark_topk_attention

    generator = torch.Generator(device="cuda").manual_seed(43)
    q, k, v = (
        torch.randn(1, 8256, 1, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    ranges = torch.tensor([[0, 8192], [8192, 8256]], device="cuda", dtype=torch.int64)
    mapping = torch.tensor([0] * 128 + [1], device="cuda", dtype=torch.int64)
    layout = SimpleNamespace(video_tokens=8192, sequence_length=8256)
    controller = _Controller(H3SparseAttentionConfig.spark(
        20, sol_tail_granularity="block", sol_log_density=False,
        sol_route_topk_execution="packed_external_no_route_qk",
    ))
    output = _spark_topk_attention(
        controller, q, k, v, layout,
        virtual_query_data=(ranges, mapping, None), _query_tokens=8192,
    )
    assert output.shape == q.shape
    assert torch.isfinite(output[:, :8192]).all()
    assert controller.counts["sol_topk_packed_route_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block8x8_runs_with_topk_threshold_route():
    from h3_sparse_attention.processor import _Controller
    from h3_sparse_attention.spark_integration import _spark_topk_attention

    gen = torch.Generator(device="cuda").manual_seed(44)
    q, k, v = (
        torch.randn(1, 8256, 1, 128, generator=gen, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    ranges = torch.tensor([[0, 8192], [8192, 8256]], device="cuda", dtype=torch.int64)
    mapping = torch.tensor([0] * 128 + [1], device="cuda", dtype=torch.int64)
    layout = SimpleNamespace(video_tokens=8192, sequence_length=8256)
    controller = _Controller(H3SparseAttentionConfig.spark(
        20, sol_tail_granularity="block8x8", sol_log_density=False,
    ))
    output = _spark_topk_attention(
        controller, q, k, v, layout,
        virtual_query_data=(ranges, mapping, None), _query_tokens=8192,
    )
    assert output.shape == q.shape
    assert torch.isfinite(output[:, :8192]).all()
    assert controller.counts["sol_topk_calls"] == 1
