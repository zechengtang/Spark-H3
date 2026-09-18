"""Threshold construction should not read CUDA scalars outside reporting."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("tokens", [1024, 1025])
@pytest.mark.parametrize("tied", [False, True])
def test_optional_tie_reporting_preserves_threshold_without_scalar_reads(
    monkeypatch, tokens, tied
):
    from h3_sparse_attention.sol_topk_cutoff import gemm_radix_topk_cutoff

    torch.manual_seed(42)
    q = torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(1, (tokens + 63) // 64, 2, 128,
                     device="cuda", dtype=torch.bfloat16)
    if tied:
        q.zero_()
    kwargs = dict(video_tokens=960, sink_tokens=tokens - 960, topk_ratio=.1,
                  _return_first_excluded=True)
    expected, reported = gemm_radix_topk_cutoff(q, kc, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("threshold construction read a host scalar")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        actual, unreported = gemm_radix_topk_cutoff(
            q, kc, collect_tie_stats=False, **kwargs
        )
    assert torch.equal(actual, expected)
    assert torch.equal(unreported.pop("_first_excluded"), reported.pop("_first_excluded"))
    tie_rows = reported.pop("cutoff_tie_rows")
    if tied:
        assert tie_rows > 0
    reported.pop("cutoff_tie_video_query_rows")
    assert unreported == reported


@pytest.mark.parametrize("tokens", [1024, 1025])
def test_score_map_matches_original_partial_block_mean(tokens):
    from h3_sparse_attention.sol_topk_cutoff import _gemm_score_map, _LOG2_E

    torch.manual_seed(43)
    blocks = (tokens + 63) // 64
    q = torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(1, blocks, 2, 128, device="cuda", dtype=torch.bfloat16)
    counts = torch.full((blocks,), 64., device="cuda")
    counts[-1] = tokens - (blocks - 1) * 64
    padded = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, blocks * 64 - tokens))
    means = (padded.view(1, blocks, 64, 2, 128).sum(2, dtype=torch.float32)
             / counts.view(1, blocks, 1, 1)).to(torch.bfloat16)
    expected = torch.einsum("bqhd,bkhd->bqhk", means, kc[:, :15]).float()
    expected.mul_(128 ** -.5 * _LOG2_E)
    actual = _gemm_score_map(q, kc, blocks=blocks, candidate_blocks=15)
    assert torch.equal(actual, expected)
