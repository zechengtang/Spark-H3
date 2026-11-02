"""CUDA checks for the ComfyUI-only reblock planning fast path."""

import inspect
import math
import sys
import types

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.is_floating_point():
        assert torch.isfinite(actual).all()
        assert torch.isfinite(expected).all()
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


@pytest.fixture(scope="module")
def tree_before_score_release():
    """Isolate lifetime changes: all arithmetic stays identical to current code."""
    import h3_sparse_attention.landmark_tree_v2 as current

    source = inspect.getsource(current)
    release = "            scores = None\n"
    assert source.count(release) == 1
    module = types.ModuleType("h3_sparse_attention._test_before_score_release")
    module.__file__ = current.__file__
    module.__package__ = "h3_sparse_attention"
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.replace(release, ""), current.__file__, "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[module.__name__]


@pytest.mark.parametrize("batch,heads,tokens", [
    (1, 1, 65), (1, 3, 2048), (2, 3, 2080),
    (1, 8, 4097), (1, 56, 37296), (1, 56, 74592),
])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("seed", [11, 43])
@pytest.mark.parametrize("pattern", ["random", "zero", "repeated_rows"])
def test_projection_can_consume_head_major_key_without_changing_bits(
    batch, heads, tokens, column_major, seed, pattern,
):
    from h3_sparse_attention.landmark_projection import project_bthd

    torch.manual_seed(seed)
    key = torch.randn(batch * heads, tokens, 128, device="cuda", dtype=torch.bfloat16)
    if pattern == "zero":
        key.zero_()
    elif pattern == "repeated_rows":
        key[:, 1:].copy_(key[:, :1].clone())
    factor = torch.randn(batch * heads, 128, 128, device="cuda", dtype=torch.bfloat16)
    if column_major:
        factor = factor.transpose(-1, -2)
    expected = torch.bmm(key, factor)
    source = key.view(batch, heads, tokens, 128).transpose(1, 2)
    project_bthd(source, factor, out=key, block_m=128)
    _assert_bits(key, expected)


@pytest.mark.parametrize("grid", [(1, 32, 64), (1, 32, 65)])
@pytest.mark.parametrize("seed", [20261030, 20261102, 20261103])
def test_consuming_key_preserves_real_runtime_permutation(grid, seed):
    from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
    from comfyui_reblock_plan import build_comfy_reblock_permutations
    from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices

    torch.manual_seed(seed)
    video = grid[0] * grid[1] * grid[2]
    tokens, heads = video + 37, 3
    ids = torch.arange(tokens, device="cuda")
    layout = ComfyPackedLayout(ids, ids, grid, video, tokens,
                               torch.zeros(video, 3, device="cuda"))
    indices = hilbert_midpoint_sample_indices(grid, video // 64, device="cuda")
    q = torch.randn(1, indices.numel(), heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    compact = k[:, :video].transpose(1, 2).contiguous().transpose(1, 2)
    expected = build_comfy_reblock_permutations(
        ComfySparkController(ComfySparkConfig()), q, k, layout,
        query_sample_indices=indices)
    controller = ComfySparkController(ComfySparkConfig())
    actual = build_comfy_reblock_permutations(
        controller, q, compact, layout,
        query_sample_indices=indices, consume_key=True)
    assert controller.counts["reblock_plan_inplace_key"] == 1
    for a, b in zip(actual, expected):
        if isinstance(a, torch.Tensor):
            _assert_bits(a, b)


@pytest.mark.parametrize("heads,grid", [
    (1, (1, 32, 64)), (3, (2, 32, 33)),
    (8, (1, 128, 256)), (56, (1, 144, 259)),
])
@pytest.mark.parametrize("seed", [11, 43])
@pytest.mark.parametrize("pattern", ["random", "zero", "repeated_rows"])
def test_early_score_release_preserves_permutation_and_inverse(
    tree_before_score_release, heads, grid, seed, pattern,
):
    from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation

    torch.manual_seed(seed)
    tokens = math.prod(grid)
    features = torch.randn(heads, tokens, 128, device="cuda", dtype=torch.bfloat16)
    if pattern == "zero":
        features.zero_()
    elif pattern == "repeated_rows":
        features[:, 1:].copy_(features[:, :1].clone())
    before = features.clone()
    options = dict(batch=heads, tokens=tokens, dim=128, grid_shape=grid,
                   device=features.device, initial_order="flat", return_inverse=True,
                   compact_direct_route=True)
    old = tree_before_score_release.PreparedLandmarkTreeV2Permutation(**options)
    new = PreparedLandmarkTreeV2Permutation(**options)
    expected = old.run(features, allow_cuda_graph=False)
    actual = new.run(features, allow_cuda_graph=False)
    for a, b in zip(actual, expected):
        _assert_bits(a, b)
    _assert_bits(features, before)


@pytest.mark.parametrize("tokens", [2048, 2080])
def test_early_score_release_preserves_cuda_graph_replay(tree_before_score_release, tokens):
    from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation

    torch.manual_seed(tokens)
    options = dict(batch=3, tokens=tokens, dim=128, grid_shape=(1, 1, tokens),
                   device=torch.device("cuda"), initial_order="flat", return_inverse=True,
                   compact_direct_route=True)
    old = tree_before_score_release.PreparedLandmarkTreeV2Permutation(**options)
    new = PreparedLandmarkTreeV2Permutation(**options)
    # Change inputs across bootstrap/capture/replay to catch stale-buffer errors.
    for _ in range(6):
        features = torch.randn(3, tokens, 128, device="cuda", dtype=torch.bfloat16)
        expected, actual = old.run(features), new.run(features)
        for a, b in zip(actual, expected):
            _assert_bits(a, b)
    assert old.graph_active and new.graph_active


@pytest.mark.parametrize("heads,grid", [(3, (1, 32, 64)), (8, (1, 32, 65))])
@pytest.mark.parametrize("seed", [11, 43])
def test_reblock_lifetime_changes_preserve_final_spark_output(
    monkeypatch, tree_before_score_release, heads, grid, seed,
):
    import comfyui_backend as backend
    import comfyui_reblock_plan as planner
    import h3_sparse_attention.landmark_tree_v2 as tree
    from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices

    torch.manual_seed(seed)
    video = math.prod(grid)
    tokens = video + 37
    ids = torch.arange(tokens, device="cuda")
    layout = backend.ComfyPackedLayout(ids, ids, grid, video, tokens,
                                      torch.zeros(video, 3, device="cuda"), 0)
    q, k, v = [torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    key_before = k.clone()
    indices = hilbert_midpoint_sample_indices(grid, video // 64, device="cuda")
    original_build = planner.build_comfy_reblock_permutations

    def consumed_build(controller, query, key, layout, **kwargs):
        compact = key[:, :video].transpose(1, 2).contiguous().transpose(1, 2)
        result = original_build(controller, query.index_select(1, indices), compact, layout,
                                query_sample_indices=indices, consume_key=True, **kwargs)
        assert controller.counts["reblock_plan_inplace_key"] == 1
        return result

    def run():
        controller = backend.ComfySparkController(backend.ComfySparkConfig())
        return backend.comfy_kitchen_spark_attention(controller, q, k, v, layout, 0)

    with monkeypatch.context() as patch:
        patch.setattr(tree, "_recursive_landmark_tree_v2",
                      tree_before_score_release._recursive_landmark_tree_v2)
        expected = run()
    _assert_bits(run(), expected)
    with monkeypatch.context() as patch:
        patch.setattr(planner, "build_comfy_reblock_permutations", consumed_build)
        _assert_bits(run(), expected)
    _assert_bits(k, key_before)


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
