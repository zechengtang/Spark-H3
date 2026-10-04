import dataclasses

import pytest
import torch

from h3_sparse_attention.landmark_direction import landmark_direction_factors
from h3_sparse_attention.mahalanobis_kmeans import (
    estimate_noncentered_second_moment,
    flat64_midpoint_sample_indices,
)
from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
from h3_sparse_attention.processor import H3SparseAttentionConfig, PackedLayout, _Controller
from h3_sparse_attention.spark_integration import (
    _landmark_tree_v2_combined_permutations,
    _landmark_tree_v2_qk_block_permutations,
)


def _direct_m2(x):
    x = x.float()
    return x.transpose(-1, -2) @ x / x.shape[-2]


@pytest.mark.parametrize(
    ("samples_per_block", "expected_first_block"),
    [(1, [32]), (2, [16, 48]), (4, [8, 24, 40, 56])],
)
def test_flat64_midpoint_indices_are_deterministic_real_unique(
    samples_per_block, expected_first_block
):
    indices, weights = flat64_midpoint_sample_indices(
        69, samples_per_block, device="cpu"
    )
    assert indices[:samples_per_block].tolist() == expected_first_block
    assert indices.unique().numel() == indices.numel()
    assert int(indices.min()) >= 0 and int(indices.max()) < 69
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    # The five-token tail follows the same formula and uses real token ids.
    tail_count = min(samples_per_block, 5)
    expected_tail = [64 + ((2 * j + 1) * 5) // (2 * tail_count)
                     for j in range(tail_count)]
    assert indices[-tail_count:].tolist() == expected_tail


def test_flat64_block_mean_matches_weighted_formula_with_tail():
    torch.manual_seed(1)
    x = torch.randn(2, 69, 5, dtype=torch.bfloat16)
    actual, diagnostics = estimate_noncentered_second_moment(
        x, "flat64_block_mean"
    )
    first = x[:, :64].float().mean(1)
    tail = x[:, 64:].float().mean(1)
    expected = (
        (64 / 69) * first.unsqueeze(2) @ first.unsqueeze(1)
        + (5 / 69) * tail.unsqueeze(2) @ tail.unsqueeze(1)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert diagnostics["flat64_blocks"] == 2


@pytest.mark.parametrize("count", [1, 2, 4])
def test_flat64_midpoint_moment_matches_explicit_weighted_formula(count):
    torch.manual_seed(2)
    x = torch.randn(2, 71, 6)
    estimator = f"flat64_midpoint_{count}"
    actual, diagnostics = estimate_noncentered_second_moment(x, estimator)
    indices, weights = flat64_midpoint_sample_indices(71, count, device="cpu")
    selected = x.index_select(1, indices).float()
    expected = selected.transpose(1, 2) @ (selected * weights[None, :, None])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert diagnostics["sample_indices"].tolist() == indices.tolist()


def test_flat64_mean_diag_has_exact_full_m2_diagonal():
    torch.manual_seed(3)
    x = torch.randn(3, 131, 7, dtype=torch.bfloat16)
    actual, diagnostics = estimate_noncentered_second_moment(x, "flat64_mean_diag")
    exact = _direct_m2(x)
    torch.testing.assert_close(
        actual.diagonal(dim1=-2, dim2=-1),
        exact.diagonal(dim1=-2, dim2=-1),
        rtol=2e-6,
        atol=2e-6,
    )
    assert diagnostics["flat64_blocks"] == 3


def test_full_chunked_matches_direct_fp32_reference():
    torch.manual_seed(4)
    x = torch.randn(2, 137, 9, dtype=torch.bfloat16)
    actual, diagnostics = estimate_noncentered_second_moment(
        x, "full", full_chunk_size=17
    )
    torch.testing.assert_close(actual, _direct_m2(x), rtol=2e-6, atol=2e-6)
    assert diagnostics["sample_count"] == 137
    assert diagnostics["full_chunk_size"] == 17


def test_hilbert_estimator_uses_supplied_indices():
    torch.manual_seed(5)
    x = torch.randn(2, 20, 4)
    indices = torch.tensor([1, 6, 12, 19])
    actual, diagnostics = estimate_noncentered_second_moment(
        x, "hilbert_midpoint", sample_indices=indices
    )
    torch.testing.assert_close(actual, _direct_m2(x[:, indices]))
    assert torch.equal(diagnostics["sample_indices"], indices)


def test_landmark_direction_default_path_is_unchanged_and_sides_are_identity():
    torch.manual_seed(6)
    q = torch.randn(2, 40, 8, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    indices = torch.tensor([1, 9, 17, 25, 33])
    legacy = landmark_direction_factors(q, k, indices)
    explicit = landmark_direction_factors(
        q, k, indices, m2_estimator="hilbert_midpoint", m2_side="both"
    )
    assert torch.equal(legacy[0], explicit[0])
    assert torch.equal(legacy[1], explicit[1])

    query_factor, key_factor = landmark_direction_factors(
        q, k, indices, m2_estimator="full", m2_side="query"
    )
    identity = torch.eye(8).expand(2, -1, -1)
    assert torch.equal(query_factor, identity)
    assert not torch.equal(key_factor, identity)


@pytest.mark.parametrize(
    ("side", "query_constructed", "key_constructed"),
    [
        ("none", False, False),
        ("query", False, True),
        ("key", True, False),
        ("both", True, True),
    ],
)
def test_m2_side_constructs_only_the_required_opposite_moment(
    side, query_constructed, key_constructed
):
    torch.manual_seed(7)
    q = torch.randn(2, 67, 6)
    k = torch.randn_like(q)
    indices = torch.tensor([5, 21, 42, 63])
    _, _, diagnostics = landmark_direction_factors(
        q,
        k,
        indices,
        m2_estimator="full",
        m2_side=side,
        return_diagnostics=True,
    )
    assert diagnostics["query"]["constructed"] is query_constructed
    assert diagnostics["key"]["constructed"] is key_constructed


def test_config_defaults_and_experiment_validation():
    baseline = H3SparseAttentionConfig.spark(20)
    assert baseline.landmark_tree_v2_m2_side == "both"
    assert baseline.landmark_tree_v2_m2_estimator == "hilbert_midpoint"
    assert baseline.landmark_tree_v2_proxy_iterations == 2
    assert baseline.landmark_tree_v2_proxy_seed_rule == "farthest_pair"
    assert baseline.landmark_tree_v2_proxy_update_rule == "mean"
    assert baseline.landmark_tree_v2_landmark_mode == "midpoint"
    assert not hasattr(baseline, "landmark_tree_v2_landmark_compression")
    assert baseline.landmark_tree_v2_layout_reuse == "independent"
    assert dataclasses.asdict(H3SparseAttentionConfig.spark(
        20,
        landmark_tree_v2_m2_side="none",
        landmark_tree_v2_m2_estimator="full",
        landmark_tree_v2_proxy_iterations=0,
        landmark_tree_v2_proxy_seed_rule="endpoint_order",
        landmark_tree_v2_proxy_update_rule="medoid",
        landmark_tree_v2_layout_reuse="q_from_k",
    ))

    with pytest.raises(ValueError, match="m2_estimator"):
        H3SparseAttentionConfig.spark(20, landmark_tree_v2_m2_estimator="bad")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_legacy_mean_group2_still_builds_reblock_plan():
    grid = (1, 32, 64)
    tokens = 2048
    ids = torch.arange(tokens, device="cuda")
    positions = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, device="cuda") for size in grid), indexing="ij"
        ), dim=-1,
    ).reshape(tokens, 3)
    layout = PackedLayout(ids, ids.clone(), grid, tokens, tokens, positions)
    torch.manual_seed(106)
    query = torch.randn(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    config = H3SparseAttentionConfig.spark(
        20, landmark_tree_v2_landmark_mode="mean",
        landmark_tree_v2_group_size=2,
    )
    plan, _, permutation, inverse = _landmark_tree_v2_combined_permutations(
        _Controller(config), query, key, layout
    )
    expected = ids.expand_as(permutation)
    assert plan.landmark_mode == "mean"
    assert torch.equal(permutation.sort(1).values, expected)
    assert torch.equal(inverse.gather(1, permutation), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout_reuse", ["q_from_k", "k_from_q"])
def test_shared_layout_aliases_qk_indices_without_duplicate_storage(layout_reuse):
    grid = (1, 32, 64)
    tokens = 2048
    ids = torch.arange(tokens, device="cuda")
    positions = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, device="cuda") for size in grid), indexing="ij"
        ), dim=-1,
    ).reshape(tokens, 3)
    layout = PackedLayout(ids, ids.clone(), grid, tokens, tokens, positions)
    torch.manual_seed(107)
    query = torch.randn(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    config = H3SparseAttentionConfig.spark(
        20, landmark_tree_v2_layout_reuse=layout_reuse
    )
    controller = _Controller(config)

    # The third call exercises the captured CUDA-graph path.  Record the input
    # passed into the second run to verify that capture consumes the producer's
    # fixed buffer directly rather than cloning another full transformed input.
    capture_input = {}
    for call in range(3):
        query_perm, query_inv, key_perm, key_inv, _, _ = (
            _landmark_tree_v2_qk_block_permutations(
                controller, query, key, layout
            )
        )
        if call == 0:
            plan = next(
                value
                for value in controller.rope_sol_key_clustering_static.values()
                if isinstance(value, PreparedLandmarkTreeV2Permutation)
            )
            original_run = plan.run

            def record_capture_input(samples, *args, **kwargs):
                capture_input["data_ptr"] = samples.data_ptr()
                return original_run(samples, *args, **kwargs)

            plan.run = record_capture_input

    assert plan.graph_active
    assert capture_input["data_ptr"] == plan.graph_input.data_ptr()
    assert query_perm.data_ptr() == key_perm.data_ptr()
    assert query_inv.data_ptr() == key_inv.data_ptr()
    assert query_perm.shape == (1, 1, tokens)
    assert query_inv.shape == (1, 1, tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_graph_input_capture_failure_falls_back_to_eager(monkeypatch):
    grid = (1, 16, 64)
    tokens = 1024
    ids = torch.arange(tokens, device="cuda")
    positions = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, device="cuda") for size in grid), indexing="ij"
        ), dim=-1,
    ).reshape(tokens, 3)
    layout = PackedLayout(ids, ids.clone(), grid, tokens, tokens, positions)
    torch.manual_seed(108)
    query = torch.randn(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    controller = _Controller(H3SparseAttentionConfig.spark(20))

    first = _landmark_tree_v2_qk_block_permutations(
        controller, query, key, layout
    )
    plan = next(
        value
        for value in controller.rope_sol_key_clustering_static.values()
        if isinstance(value, PreparedLandmarkTreeV2Permutation)
    )

    def fail_capture():
        raise RuntimeError("injected graph capture failure")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", fail_capture)
    second = _landmark_tree_v2_qk_block_permutations(
        controller, query, key, layout
    )

    assert plan._graph_failed
    assert not plan.graph_active
    assert plan._static_input is None
    for expected, actual in zip(first[:4], second[:4]):
        assert torch.equal(expected, actual)
