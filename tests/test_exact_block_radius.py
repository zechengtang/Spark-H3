import pytest
import torch
from types import SimpleNamespace


def test_local_block_policy_normalization_preserves_legacy_flags():
    from h3_sparse_attention.local_blocks import normalize_local_block_radius

    assert normalize_local_block_radius(False) == -1
    assert normalize_local_block_radius(True) == 1
    assert normalize_local_block_radius(-1) == -1
    assert normalize_local_block_radius(0) == 0
    assert normalize_local_block_radius(3) == 3
    with pytest.raises(ValueError):
        normalize_local_block_radius(-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("radius", [0, 1, 2])
def test_kernel_exact_route_uses_requested_symmetric_radius(radius):
    from h3_sparse_attention.sol_vaware_compensation import exact_attention

    torch.manual_seed(901 + radius)
    blocks, heads = 5, 1
    shape = (1, blocks * 64, heads, 128)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    kc = k.view(1, blocks, 64, heads, 128).mean(2)
    vs = v.view(1, blocks, 64, heads, 128).sum(2)
    threshold = torch.full(
        (1, blocks, heads), torch.inf, device="cuda", dtype=torch.float32
    )

    _, _, route = exact_attention(
        q,
        k,
        v,
        kc,
        vs,
        threshold=threshold,
        force_local_blocks=radius,
    )

    ids = torch.arange(blocks, device="cuda")
    expected = (ids[:, None] - ids[None, :]).abs() <= radius
    torch.testing.assert_close(route[0, :, 0].bool(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_local_exact_blocks_are_union_after_route_selection():
    from h3_sparse_attention.sol_vaware_compensation import exact_attention

    torch.manual_seed(905)
    blocks = 5
    shape = (1, blocks * 64, 1, 128)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    kc = k.view(1, blocks, 64, 1, 128).mean(2)
    vs = v.view(1, blocks, 64, 1, 128).sum(2)
    threshold = torch.zeros((1, blocks, 1), device="cuda", dtype=torch.float32)

    _, _, baseline = exact_attention(
        q, k, v, kc, vs, threshold=threshold, force_local_blocks=-1
    )
    _, _, self_exact = exact_attention(
        q, k, v, kc, vs, threshold=threshold, force_local_blocks=0
    )

    expected = baseline.bool()
    diagonal = torch.eye(blocks, device="cuda", dtype=torch.bool)
    expected |= diagonal[None, :, None, :]
    torch.testing.assert_close(self_exact.bool(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_verbose_counter_reports_only_local_blocks_missing_from_topk(monkeypatch):
    from h3_sparse_attention.sol_topk_cutoff import _gemm_score_map_prefix
    from h3_sparse_attention.spark_integration import _record_exact_block_verbose

    monkeypatch.setenv("H3_VERBOSE_EXACT_BLOCKS", "1")
    torch.manual_seed(907)
    blocks, heads = 4, 2
    q = torch.randn(
        1, blocks * 64, heads, 128, device="cuda", dtype=torch.bfloat16
    )
    kc = torch.randn(
        1, blocks, heads, 128, device="cuda", dtype=torch.bfloat16
    )
    scores = _gemm_score_map_prefix(
        q, kc, query_blocks=blocks, candidate_blocks=blocks
    )
    ordered = scores.sort(dim=-1, descending=True).values
    threshold = (ordered[..., 1] + ordered[..., 2]) * 0.5
    controller = SimpleNamespace(exact_block_verbose_accumulator=None)

    _record_exact_block_verbose(
        controller,
        q,
        kc,
        threshold,
        video_tokens=blocks * 64,
        query_tokens=blocks * 64,
        radius=1,
    )

    selected = scores > threshold[..., None]
    ids = torch.arange(blocks, device="cuda")
    local = (ids[:, None] - ids[None, :]).abs() <= 1
    expanded_local = local[None, :, None, :].expand_as(selected)
    accumulator = controller.exact_block_verbose_accumulator
    assert int(accumulator["route_rows"].item()) == blocks * heads
    assert int(accumulator["local_candidates"].item()) == int(expanded_local.sum())
    assert int(accumulator["already_selected"].item()) == int(
        (expanded_local & selected).sum()
    )
    assert int(accumulator["added_exact_blocks"].item()) == int(
        (expanded_local & ~selected).sum()
    )
