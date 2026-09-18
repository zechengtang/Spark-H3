"""Exact global anchors and prefix-only attention for dense-context callers."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('lengths',[(8192,137),(4096,4096,137)])
def test_parallel_anchor_preserves_sequential_reduction(dtype,lengths):
    import h3_sparse_attention.sol_numerator_virtual_q as rw
    torch.manual_seed(921)
    b,h,t=2,3,sum(lengths)
    q=torch.randn(b,t,h,128,device='cuda',dtype=dtype)
    ends=[0]
    for size in lengths:ends.append(ends[-1]+size)
    ranges=torch.tensor(list(zip(ends[:-1],ends[1:])),device='cuda')
    expected=torch.empty(b,len(lengths),h,128,device='cuda',dtype=dtype)
    rw._anchors[(len(lengths),b*h)](q,ranges,expected,t,h,len(lengths),num_warps=4)
    actual=rw.build_virtual_anchors(q,ranges)
    assert torch.equal(expected,actual)


@pytest.mark.parametrize('precomputed',[False,True])
@pytest.mark.parametrize('head_chunk',[False,True])
def test_query_prefix_preserves_video_and_dense_context(monkeypatch,precomputed,head_chunk):
    if torch.cuda.get_device_capability() != (12,0):pytest.skip('SM120 required')
    import h3_sparse_attention.sol_numerator_virtual_q as rw
    torch.manual_seed(922)
    b,t,h,video=2,8329,3,8192;n=(t+63)//64
    q,k,v=[torch.randn(b,t,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    # More than one video parent exercises truncated streaming as well as
    # the single-video-parent production case in the full-size benchmark.
    ranges=torch.tensor([[0,4096],[4096,8192],[8192,8256],[8256,8320],[8320,t]],device='cuda')
    mapping=torch.tensor([0]*64+[1]*64+[2,3,4],device='cuda')
    anchors=rw.build_virtual_anchors(q,ranges);kc=rw.reduce_virtual_key_centroids(k)
    threshold=torch.full((b,n,h),.02,device='cuda')
    summaries=rw.virtual_summaries(anchors,k,v) if precomputed else None
    if head_chunk:
        monkeypatch.setattr(rw,'_SUMMARY_WORKSPACE_BYTES',b*2*n*516*2)
    def run(prefix):
        out=rw._fused_virtual(q,k,v,anchors,ranges,mapping,kc,threshold,None,video,t-video,summaries,
                              force_local_blocks=False,_query_tokens=video if prefix else None)
        out[:,video:]=torch.nn.functional.scaled_dot_product_attention(q[:,video:].transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        return out
    assert torch.equal(run(False),run(True))
    run(True);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=run(True)
    q.mul_(.5)
    graph.replay()
    assert torch.equal(captured,run(False))
    with pytest.raises(ValueError,match='query-block boundary'):
        rw._fused_virtual(q,k,v,anchors,ranges,mapping,kc,threshold,None,video,t-video,_query_tokens=video-1)


def test_topk_caller_preserves_full_output_default():
    if torch.cuda.get_device_capability() != (12,0):pytest.skip('SM120 required')
    from types import SimpleNamespace
    import h3_sparse_attention.processor as proc
    import h3_sparse_attention.spark_integration as integration
    torch.manual_seed(924)
    video,t,h=8192,8193,2
    q,k,v=[torch.randn(1,t,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    ranges=torch.tensor([[0,video],[video,t]],device='cuda')
    mapping=torch.tensor([0]*128+[1],device='cuda')
    layout=SimpleNamespace(video_tokens=video,sequence_length=t)
    cfg=proc.H3SparseAttentionConfig.sol(20,sol_route_topk_ratio=.1,
        sol_force_local_blocks=False,sol_log_density=False,
        sol_landmark_preprocess=True,sol_landmark_preprocess_version='v2',
        sol_virtual_query_levels_up=1)
    def run(prefix):
        out=integration._spark_topk_attention(proc._Controller(cfg),q,k,v,layout,
            virtual_query_data=(ranges,mapping,None),_query_tokens=video if prefix else None)
        if not prefix:assert torch.isfinite(out).all()
        out[:,video:]=torch.nn.functional.scaled_dot_product_attention(q[:,video:].transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        return out
    assert torch.equal(run(False),run(True))
