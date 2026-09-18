"""Exact output and graph coverage for the arbitrary CUDA routing kernels."""
import pytest
import torch
from h3_sparse_attention.landmark_v2_terminal import route_scores

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

@pytest.mark.parametrize('children', [1,2,3,5,7,8,15,16,31,32])
@pytest.mark.parametrize('kind', ['random','ties','zero'])
def test_small_exact_keys_and_unequal_capacities(children, kind):
    torch.manual_seed(100+children)
    caps=tuple(9+i%4 for i in range(children)); n=sum(caps)
    ids=torch.stack([torch.randperm(n)*3+72000 for _ in range(3)])
    scores=torch.randn(3,n,children-1)
    if kind=='ties': scores=scores.round()
    if kind=='zero': scores.zero_()
    expected=route_scores(scores,ids,caps)
    result=route_scores(scores.cuda(),ids.cuda(),caps,max_original_index=75000)
    assert torch.equal(result.cpu(),expected)
    for row in result:
        assert torch.equal(torch.bincount(row,minlength=children).cpu(),torch.tensor(caps))

@pytest.mark.parametrize('caps', [(4096,4096,4096),(2500,2496,2500,2496,2496),
                                  (128,)*15,(128,)*16,(96,)*31])
def test_large_batched_and_early_stop_routing(caps):
    torch.manual_seed(len(caps));n=sum(caps)
    ids=torch.stack([torch.randperm(n)+90000 for _ in range(2)])
    scores=torch.randn(2,n,len(caps)-1).round()
    expected=route_scores(scores,ids,caps)
    actual=route_scores(scores.cuda(),ids.cuda(),caps,max_original_index=110000)
    assert torch.equal(actual.cpu(),expected)


def test_large_id_fallback_and_strided_inputs():
    caps=(65,33,17,49,19);n=sum(caps)
    ids=(torch.randperm(n,dtype=torch.long)+(1<<30))[None].expand(2,-1)
    scores=torch.randn(2,len(caps)-1,n).transpose(1,2)
    expected=route_scores(scores,ids,caps)
    actual=route_scores(scores.cuda(),ids.cuda(),caps,max_original_index=(1<<31)-1)
    assert torch.equal(actual.cpu(),expected)

@pytest.mark.parametrize('caps', [(64,)*15,(2048,)*5])
def test_graph_replay_uses_changed_scores(caps):
    n=sum(caps);torch.manual_seed(n)
    scores=torch.randn(2,n,len(caps)-1,device='cuda')
    ids=torch.stack([torch.randperm(n,device='cuda') for _ in range(2)])
    for _ in range(3): route_scores(scores,ids,caps,max_original_index=n-1)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual=route_scores(scores,ids,caps,max_original_index=n-1)
    for _ in range(2):
        scores.copy_(torch.randn_like(scores).round())
        graph.replay()
        expected=route_scores(scores.cpu(),ids.cpu(),caps)
        assert torch.equal(actual.cpu(),expected)

@pytest.mark.parametrize('children', [1,2,3,5,7,8,14,15,16,31,32])
def test_fused_stable_partition(children):
    from h3_sparse_attention.landmark_v2_terminal import partition_scores
    torch.manual_seed(143+children)
    caps=tuple(17+i%3 for i in range(children));n=sum(caps)
    scores=torch.randn(3,n,children-1).round()
    ids=torch.stack([torch.randperm(n)*7+65000 for _ in range(3)])
    labels=route_scores(scores,ids,caps)
    expected=ids.gather(1,labels.argsort(dim=1,stable=True))
    actual=partition_scores(scores.cuda(),ids.cuda(),caps,max_original_index=75000)
    assert torch.equal(actual.cpu(),expected)


def test_score_bit_order_and_full_width_packed_key():
    from h3_sparse_attention.landmark_v2_terminal import partition_scores
    caps=(17,)*32;n=sum(caps)
    scores=torch.tensor([-float('inf'),-1.,-0.,0.,1.,float('inf'),float('nan')]).repeat((3*n*31+6)//7)[:3*n*31].view(3,n,31)
    ids=torch.stack([torch.randperm(n)+(1<<26) for _ in range(3)])
    labels=route_scores(scores,ids,caps)
    actual=route_scores(scores.cuda(),ids.cuda(),caps,max_original_index=(1<<27)-1)
    assert torch.equal(actual.cpu(),labels)
    expected=ids.gather(1,labels.argsort(dim=1,stable=True))
    partitioned=partition_scores(scores.cuda(),ids.cuda(),caps,max_original_index=(1<<27)-1)
    assert torch.equal(partitioned.cpu(),expected)


def test_invalid_capacity_sum_is_rejected():
    from h3_sparse_attention.landmark_v2_terminal import partition_scores
    scores=torch.zeros(1,32,2,device='cuda');ids=torch.arange(32,device='cuda')[None]
    for fn in (route_scores,partition_scores):
        with pytest.raises(ValueError,match='sum to node size'):
            fn(scores,ids,(16,16,16))

@pytest.mark.parametrize('children', [14,15])
@pytest.mark.parametrize('scale', [0.,1.,1.e6])
def test_proxy_register_cap_preserves_directions(monkeypatch, children, scale):
    import h3_sparse_attention.landmark_v2_cosine_fast as cosine
    if torch.cuda.get_device_capability() != (12,0):
        pytest.skip('SM120 launch tuning')
    monkeypatch.setenv('H3_LMV2_SMALL_PROXY_FAST', '0')
    torch.manual_seed(children)
    centers=(torch.randn(1024,32,128,device='cuda')*scale).to(torch.bfloat16)
    weights=torch.full((1024,32),children*2,device='cuda',dtype=torch.long)
    caps=(64,)*children
    monkeypatch.setattr(cosine,'_ARBITRARY_PROXY_MAX_REGISTERS',None)
    expected=cosine.build_cosine_directions(centers,weights,caps)
    monkeypatch.setattr(cosine,'_ARBITRARY_PROXY_MAX_REGISTERS',64)
    actual=cosine.build_cosine_directions(centers,weights,caps)
    assert torch.equal(actual,expected)
