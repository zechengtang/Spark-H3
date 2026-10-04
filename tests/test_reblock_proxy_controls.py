import pytest
import torch

from h3_sparse_attention.landmark_tree_v2 import (
    PreparedLandmarkTreeV2Permutation,
    recursive_landmark_tree_v2_blocks,
    recursive_landmark_tree_v2_reference,
)
from h3_sparse_attention.landmark_v2_terminal import node_split_reference


def _assert_complete(result, tokens):
    expected = torch.arange(tokens, device=result.permutation.device).expand_as(
        result.permutation
    )
    assert torch.equal(result.permutation.sort(1).values, expected)
    assert torch.equal(
        result.inverse_permutation.gather(1, result.permutation), expected
    )


def test_proxy_default_is_explicit_two_pass_farthest_mean():
    torch.manual_seed(101)
    samples = torch.randn(2, 512, 9)
    common = dict(grid_shape=(1, 8, 64), max_children=4)
    implicit = recursive_landmark_tree_v2_reference(samples, **common)
    explicit = recursive_landmark_tree_v2_reference(
        samples,
        proxy_iterations=2,
        seed_rule="farthest_pair",
        update_rule="mean",
        **common,
    )
    assert torch.equal(implicit.permutation, explicit.permutation)
    assert all(stat.proxy_iterations == 2 for stat in explicit.split_stats)
    assert all(stat.seed_rule == "farthest_pair" for stat in explicit.split_stats)
    assert all(stat.update_rule == "mean" for stat in explicit.split_stats)


def test_zero_updates_scores_tokens_with_seed_direction():
    centers = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]
    )
    samples = torch.tensor(
        [[[0.8, 0.2], [-0.7, 0.1], [0.4, -0.9], [-0.2, -0.8]]]
    )
    original = torch.arange(4)[None]
    actual = node_split_reference(
        samples,
        original,
        centers,
        torch.ones(1, 4, dtype=torch.int64),
        (2, 2),
        proxy_iterations=0,
    )
    # Farthest-pair tie breaking picks landmarks 0 and 1.  The seed cosine
    # direction is (-1,0) - (1,0), and parent_order preserves IDs per side.
    unit_samples = samples / samples.norm(dim=-1, keepdim=True)
    score = unit_samples @ torch.tensor([-2.0, 0.0])
    left = score[0].argsort(stable=True)[:2]
    is_left = torch.zeros(4, dtype=torch.bool).scatter(0, left, True)
    expected = torch.cat((original[0, is_left], original[0, ~is_left]))[None]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("iterations", [0, 1, 2, 4])
def test_proxy_iteration_counts_make_complete_capacity_constrained_tree(iterations):
    torch.manual_seed(102)
    samples = torch.randn(2, 1024, 7)
    result = recursive_landmark_tree_v2_reference(
        samples,
        grid_shape=(2, 8, 64),
        max_children=4,
        proxy_iterations=iterations,
    )
    _assert_complete(result, 1024)
    assert len(result.split_stats) >= 2
    assert all(stat.proxy_iterations == iterations for stat in result.split_stats)
    assert all(stat.children <= 4 for stat in result.split_stats)


@pytest.mark.parametrize("seed_rule", ["farthest_pair", "endpoint_order"])
@pytest.mark.parametrize("update_rule", ["mean", "medoid"])
def test_proxy_seed_and_update_rules_are_independent(seed_rule, update_rule):
    torch.manual_seed(103)
    samples = torch.randn(1, 512, 6)
    result = recursive_landmark_tree_v2_reference(
        samples,
        grid_shape=(1, 8, 64),
        max_children=4,
        proxy_iterations=1,
        seed_rule=seed_rule,
        update_rule=update_rule,
    )
    _assert_complete(result, 512)
    assert all(stat.seed_rule == seed_rule for stat in result.split_stats)
    assert all(stat.update_rule == update_rule for stat in result.split_stats)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"proxy_iterations": 3}, "proxy_iterations"),
        ({"seed_rule": "random"}, "seed_rule"),
        ({"update_rule": "median"}, "update_rule"),
    ],
)
def test_proxy_controls_reject_undefined_semantics(kwargs, match):
    with pytest.raises(ValueError, match=match):
        recursive_landmark_tree_v2_reference(
            torch.randn(1, 128, 4), grid_shape=(1, 2, 64), **kwargs
        )


def test_frozen_excluded_indices_do_not_depend_on_features():
    # Input order is deliberately reversed: the emitted tail follows root order.
    frozen = torch.tensor([[77, 3]])
    first = torch.randn(1, 130, 5)
    second = torch.randn(1, 130, 5) * 1000
    common = dict(
        grid_shape=(1, 2, 65),
        initial_order="flat",
        excluded_indices=frozen,
    )
    a = recursive_landmark_tree_v2_reference(first, **common)
    b = recursive_landmark_tree_v2_reference(second, **common)
    assert a.excluded_indices.tolist() == [[3, 77]]
    assert b.excluded_indices.tolist() == [[3, 77]]
    assert a.permutation[:, -2:].tolist() == [[3, 77]]
    assert b.permutation[:, -2:].tolist() == [[3, 77]]
    _assert_complete(a, 130)
    _assert_complete(b, 130)


@pytest.mark.parametrize(
    "excluded,match",
    [
        (torch.tensor([[1]]), "shape"),
        (torch.tensor([[1, 1]]), "unique"),
        (torch.tensor([[1, 130]]), "range"),
    ],
)
def test_frozen_excluded_indices_are_strictly_validated(excluded, match):
    with pytest.raises(ValueError, match=match):
        recursive_landmark_tree_v2_reference(
            torch.randn(1, 130, 4),
            grid_shape=(1, 2, 65),
            excluded_indices=excluded,
        )


def test_prepared_cpu_plan_accepts_frozen_tail():
    samples = torch.randn(1, 130, 8, dtype=torch.bfloat16)
    plan = PreparedLandmarkTreeV2Permutation(
        batch=1,
        tokens=130,
        dim=8,
        grid_shape=(1, 2, 65),
        device="cpu",
        proxy_iterations=0,
    )
    permutation, inverse = plan.run(samples, excluded_indices=torch.tensor([[4, 91]]))
    assert permutation[:, -2:].tolist() == [[4, 91]]
    expected = torch.arange(130)[None]
    assert torch.equal(inverse.gather(1, permutation), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("iterations", [0, 1, 2, 4])
def test_experimental_cuda_path_matches_reference(iterations):
    torch.manual_seed(104)
    samples = torch.randn(1, 512, 128, device="cuda", dtype=torch.bfloat16)
    common = dict(
        grid_shape=(1, 8, 64),
        max_children=4,
        proxy_iterations=iterations,
        seed_rule="endpoint_order",
    )
    expected = recursive_landmark_tree_v2_reference(samples, **common)
    actual = recursive_landmark_tree_v2_blocks(samples, **common)
    assert torch.equal(actual.permutation, expected.permutation)
    _assert_complete(actual, 512)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_euclidean_indexed_scores_accept_compact_int32_indices():
    from h3_sparse_attention.landmark_v2_euclidean import (
        build_euclidean_directions,
        fused_euclidean_scores,
    )

    torch.manual_seed(106)
    source = torch.randn(80, 8, device="cuda", dtype=torch.bfloat16)
    centers = torch.randn(1, 16, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.full((1, 16), 4, device="cuda", dtype=torch.int32)
    directions, bias = build_euclidean_directions(centers, weights, (32, 32))
    indices64 = torch.randperm(80, device="cuda")[:64][None].contiguous()
    expected = fused_euclidean_scores(
        source, directions, bias, indices=indices64
    )
    actual = fused_euclidean_scores(
        source, directions, bias, indices=indices64.to(torch.int32)
    )
    assert torch.equal(actual, expected)
