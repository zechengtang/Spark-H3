"""Fused reweight uses weighted summaries, not ordinary V-block sums."""
import dataclasses
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (12, 0)),
    reason='requires a fused virtual-query CuTe backend (SM90/SM100/SM120)',
)

@pytest.mark.parametrize('tokens', [128, 129, 8193])
def test_key_only_reduction_matches_official_centroids(tokens):
    from sol_attn.preprocess import _reduce_kv
    from h3_sparse_attention.sol_numerator_virtual_q import reduce_virtual_key_centroids
    torch.manual_seed(42)
    k = torch.randn(1, tokens, 2, 128, device='cuda', dtype=torch.bfloat16)
    expected, _ = _reduce_kv(k, torch.randn_like(k))
    actual = reduce_virtual_key_centroids(k)
    assert torch.equal(actual, expected)

@pytest.mark.parametrize('tokens', [1025, 8193])
def test_fused_attention_preserves_output_and_skips_value_reduction(monkeypatch, tokens):
    import sol_attn.preprocess as preprocess
    import h3_sparse_attention.processor as proc
    import h3_sparse_attention.spark_integration as integration
    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', '1')
    # Compare the fused result with ordinary K/V reduction and K-only reduction.
    original = preprocess._reduce_kv
    reductions = []
    def tracked(k, v):
        reductions.append(True)
        return original(k, v)
    monkeypatch.setattr(preprocess, '_reduce_kv', tracked)
    torch.manual_seed(42)
    q, k, v = [torch.randn(1, tokens, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    video = tokens - 1
    ranges = torch.tensor([[0, video], [video, tokens]], device='cuda', dtype=torch.int64)
    mapping = torch.tensor([0] * (video // 64) + [1], device='cuda', dtype=torch.int64)
    layout = SimpleNamespace(video_tokens=video, sequence_length=tokens)
    cfg = proc.H3SparseAttentionConfig.sol(20, sol_route_topk_ratio=.1,
        sol_force_local_blocks=False, sol_log_density=False,
        sol_landmark_preprocess=True, sol_landmark_preprocess_version='v2',
        sol_virtual_query_levels_up=1)
    with torch.no_grad():
        import h3_sparse_attention.sol_numerator_virtual_q as rw
        reduce_k = rw.reduce_virtual_key_centroids
        monkeypatch.setattr(rw, "reduce_virtual_key_centroids", lambda k: preprocess._reduce_kv(k, v)[0])
        expected = integration._spark_topk_attention(proc._Controller(cfg), q, k, v, layout,
                                           virtual_query_data=(ranges, mapping, None))
        assert len(reductions) == 1
        monkeypatch.setattr(rw, "reduce_virtual_key_centroids", reduce_k)
        actual = integration._spark_topk_attention(proc._Controller(cfg), q, k, v, layout,
                                         virtual_query_data=(ranges, mapping, None))
    assert len(reductions) == 1, 'ordinary V sums were recomputed in the fused path'
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
