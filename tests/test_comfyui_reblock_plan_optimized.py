"""CUDA checks for the ComfyUI-only reblock planning fast path."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_bthd_projection_respects_factor_layout_and_fuses_root_scores():
    from h3_sparse_attention.landmark_projection import project_bthd
    from h3_sparse_attention.landmark_v2_cosine_fast import fused_cosine_scores

    torch.manual_seed(42)
    tokens, heads, dim = 256, 4, 128
    source = torch.randn(
        1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    matrix = torch.randn(heads, dim, dim, device="cuda")
    factor = torch.linalg.cholesky(
        matrix @ matrix.transpose(-1, -2)
        + 0.1 * torch.eye(dim, device="cuda")
    ).to(torch.bfloat16)
    assert factor.stride(-2) == 1  # Cholesky's production column-major layout.

    directions = torch.randn(heads, 15, dim, device="cuda", dtype=torch.float32)
    output = torch.empty(
        heads, tokens, dim, device="cuda", dtype=torch.bfloat16
    )
    scores = torch.empty(
        heads, tokens, 15, device="cuda", dtype=torch.float32
    )
    project_bthd(
        source,
        factor,
        out=output,
        block_m=128,
        score_directions=directions,
        score_out=scores,
    )

    source_bhtd = source.permute(0, 2, 1, 3).reshape(heads, tokens, dim)
    reference_output = torch.bmm(source_bhtd, factor)
    reference_scores = fused_cosine_scores(
        reference_output, directions, "fp16", block_m=128
    )
    assert torch.equal(output, reference_output)
    assert torch.allclose(scores, reference_scores, atol=1e-3, rtol=1e-3)


def test_precomputed_root_scores_preserve_plan_and_graph_replay():
    from h3_sparse_attention.landmark_tree_v2 import (
        PreparedLandmarkTreeV2Permutation,
    )
    from h3_sparse_attention.landmark_v2_cosine_fast import fused_cosine_scores
    from h3_sparse_attention.landmark_v2_fused_node import (
        fused_midpoint_directions,
    )

    torch.manual_seed(7)
    batch, tokens, dim = 2, 32 * 64, 128
    samples = torch.randn(
        batch, tokens, dim, device="cuda", dtype=torch.bfloat16
    )
    common = dict(
        batch=batch,
        tokens=tokens,
        dim=dim,
        grid_shape=(1, 1, tokens),
        device=torch.device("cuda"),
        initial_order="flat",
        max_children=16,
        landmark_mode="midpoint",
        landmark_count=32,
        return_inverse=False,
        compact_direct_route=True,
    )
    baseline = PreparedLandmarkTreeV2Permutation(**common)
    expected, _ = baseline.run(samples)

    capacities = (128,) * 16
    global_indices = (
        torch.arange(tokens, device="cuda", dtype=torch.int32)[None]
        + torch.arange(batch, device="cuda", dtype=torch.int32)[:, None] * tokens
    )
    directions = fused_midpoint_directions(
        samples.reshape(-1, dim), global_indices, capacities
    )
    root_scores = fused_cosine_scores(
        samples, directions, "fp16", block_m=128
    )
    optimized = PreparedLandmarkTreeV2Permutation(**common)
    actual, _ = optimized.run(samples, root_scores=root_scores)
    assert torch.equal(actual, expected)
    optimized.run(samples, root_scores=root_scores)
    replayed, _ = optimized.run(samples, root_scores=root_scores)
    assert optimized.graph_active
    assert torch.equal(replayed, expected)


def test_quantized_route_keeps_exact_capacities_and_permutation():
    from h3_sparse_attention.landmark_v2_route import route_scores_cuda

    torch.manual_seed(11)
    capacities = (128,) * 16
    tokens = sum(capacities)
    scores = torch.randn(3, tokens, 15, device="cuda", dtype=torch.float32)
    original = torch.stack(
        [torch.randperm(tokens, device="cuda", dtype=torch.int64) for _ in range(3)]
    ).to(torch.int32)
    permutation = route_scores_cuda(
        scores,
        original,
        capacities,
        max_original_index=tokens - 1,
        partition=True,
        approximate_key32=True,
    )
    assert torch.equal(permutation.sort(1).values, original.sort(1).values)
    assert permutation.dtype == torch.int32
