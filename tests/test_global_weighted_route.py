import pytest
import torch

from h3_sparse_attention import H3SparseAttentionConfig


def test_requires_topk():
    with pytest.raises(ValueError, match='Top-K'):
        H3SparseAttentionConfig(method='sol', total_evaluations=19,
                               sol_route_global_weighted_mean=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('side', ['both', 'query', 'key'])
def test_centroids_and_route_match_reference(side):
    from h3_sparse_attention.global_weighted_route import weighted_centroids, global_weighted_route
    torch.manual_seed(617)
    q, k = [torch.randn(2, 529, 2, 128, device='cuda', dtype=torch.bfloat16)*3 for _ in range(2)]
    def reference(x, a):
        pieces = []
        for block in x.split(64, dim=1):
            xf = block.float()
            w = ((xf*a[:, None]).sum(-1)/128**.5).softmax(1)
            pieces.append((xf*w[..., None]).sum(1))
        return torch.stack(pieces, 1).to(x.dtype)
    qm, km = reference(q, k[:, :512].float().mean(1)), reference(k, q[:, :512].float().mean(1))
    actual = weighted_centroids(q, k[:, :512].float().mean(1))
    torch.testing.assert_close(actual, qm, atol=.016, rtol=.008)
    zero = weighted_centroids(q, torch.zeros(2, 2, 128, device='cuda'))
    uniform = torch.stack([b.float().mean(1) for b in q.split(64, 1)], 1).to(q.dtype)
    torch.testing.assert_close(zero, uniform, atol=0, rtol=0)
    if side == 'key':
        qm = uniform
    if side == 'query':
        from sol_attn.preprocess import _reduce_kv
        km, _ = _reduce_kv(k, k)
    route, stats = global_weighted_route(q, k, video_tokens=512, sink_tokens=17,
        topk_ratio=.25, weighted_side=side, key_centroids=km if side=='query' else None)
    scores = torch.einsum('bqhd,bkhd->bqhk', qm, km[:, :8]).float()
    expected = torch.zeros_like(route)
    expected[..., :8].scatter_(-1, scores.topk(2, -1).indices, True)
    expected[..., 8] = True
    assert torch.equal(route, expected)
    assert stats['target_topk_blocks_per_query'] == 2
    assert (route[..., :8].sum(-1) == 2).all()
