"""Exact global anchors and prefix-only attention for dense-context callers."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_spark_query_tokens_exclude_the_mixed_video_context_block():
    from h3_sparse_attention.spark_integration import _spark_query_tokens

    assert _spark_query_tokens(8329, 8192) == 8192
    assert _spark_query_tokens(8329, 8193) == 8192
    assert _spark_query_tokens(8200, 8193) == 8192
    assert _spark_query_tokens(8329, 8193, 'pad') == 8256
    with pytest.raises(ValueError,match='suffix is too short'):
        _spark_query_tokens(8200,8193,'pad')


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
@pytest.mark.parametrize('video',[8192,8193])
def test_query_prefix_preserves_video_and_dense_context(monkeypatch,precomputed,head_chunk,video):
    if torch.cuda.get_device_capability() not in ((9,0),(10,0),(12,0)):pytest.skip('fused kernel requires SM90/SM100/SM120')
    import h3_sparse_attention.sol_numerator_virtual_q as rw
    torch.manual_seed(922)
    b,t,h=2,8329,3;n=(t+63)//64
    query_tokens=(video//64)*64
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
                              force_local_blocks=False,_query_tokens=query_tokens if prefix else None)
        out[:,query_tokens:]=torch.nn.functional.scaled_dot_product_attention(q[:,query_tokens:].transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        return out
    assert torch.equal(run(False),run(True))
    run(True);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=run(True)
    q.mul_(.5)
    graph.replay()
    assert torch.equal(captured,run(False))
    with pytest.raises(ValueError,match='query-block boundary'):
        rw._fused_virtual(q,k,v,anchors,ranges,mapping,kc,threshold,None,video,t-video,_query_tokens=query_tokens-1)


def test_streamed_fallback_query_prefix_preserves_dense_context(monkeypatch):
    if torch.cuda.get_device_capability() not in ((9,0),(10,0),(12,0)):
        pytest.skip('fallback exact kernel requires SM90/SM100/SM120')
    import h3_sparse_attention.sol_numerator_virtual_q as rw
    from sol_attn.preprocess import _reduce_kv

    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED','0')
    torch.manual_seed(923)
    b,t,h,video=1,8329,1,8193
    query_tokens=(video//64)*64
    n=(t+63)//64
    q,k,v=[torch.randn(b,t,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    ranges=torch.tensor([[0,4096],[4096,8192],[8192,8256],[8256,8320],[8320,t]],device='cuda')
    mapping=torch.tensor([0]*64+[1]*64+[2,3,4],device='cuda')
    anchors=rw.build_virtual_anchors(q,ranges)
    kc,vs=_reduce_kv(k,v)
    threshold=torch.full((b,n,h),.02,device='cuda')

    def run(prefix):
        out=rw.virtual_q_attention(
            q,k,v,virtual_ranges=ranges,leaf_to_virtual=mapping,
            virtual_anchors=anchors,key_centroids=kc,value_sums=vs,
            threshold=threshold,sink_start=video,sink_tokens=t-video,
            force_local_blocks=False,
            _query_tokens=query_tokens if prefix else None)
        out[:,query_tokens:]=torch.nn.functional.scaled_dot_product_attention(
            q[:,query_tokens:].transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        ).transpose(1,2)
        return out

    assert torch.equal(run(False),run(True))


def test_topk_caller_preserves_full_output_default():
    if torch.cuda.get_device_capability() not in ((9,0),(10,0),(12,0)):pytest.skip('fused kernel requires SM90/SM100/SM120')
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


@pytest.mark.parametrize('tail_mode',["dense","pad"])
@pytest.mark.parametrize('reblock',[False,True])
def test_nonvirtual_topk_uses_exclusive_video_query_prefix(monkeypatch,reblock,tail_mode):
    from types import SimpleNamespace
    import h3_sparse_attention.processor as proc
    import h3_sparse_attention.spark_integration as integration

    torch.manual_seed(925)
    b,t,h,video=1,137,2,97
    query_tokens=(video//64)*64 if tail_mode=='dense' else ((video+63)//64)*64
    dense_start=query_tokens if tail_mode=='dense' else video
    q,k,v=[torch.randn(b,t,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    layout=SimpleNamespace(video_tokens=video,sequence_length=t,grid=(1,1,video))
    cfg=proc.H3SparseAttentionConfig.sol(
        20,sol_route_topk_ratio=.1,sol_force_local_blocks=False,
        sol_log_density=False,sol_landmark_preprocess=reblock,
        sol_landmark_preprocess_version='v2',sol_video_tail_mode=tail_mode)
    controller=proc._Controller(cfg)
    seen=[]

    if reblock:
        identity=torch.arange(video,device='cuda').view(1,1,video).expand(b,h,video)
        monkeypatch.setattr(
            integration,'_landmark_tree_v2_qk_block_permutations',
            lambda *args:(identity,identity,identity,identity,{},{}))

    def fake_topk(controller,q,k,v,layout,virtual_query_data=None,_query_tokens=None):
        assert virtual_query_data is None
        seen.append(_query_tokens)
        if tail_mode=='pad':
            tail_start=(video//64)*64
            expected_mean=q[:,tail_start:video].mean(1,keepdim=True,dtype=torch.float32).to(q.dtype)
            assert torch.equal(
                q[:,video:query_tokens],
                expected_mean.expand(-1,query_tokens-video,-1,-1),
            )
        return torch.zeros_like(q)

    monkeypatch.setattr(integration,'_spark_topk_attention',fake_topk)
    actual=integration.spark_attention_bthd(controller,q,k,v,layout,0)
    expected=torch.nn.functional.scaled_dot_product_attention(
        q[:,dense_start:].transpose(1,2),k.transpose(1,2),v.transpose(1,2)
    ).transpose(1,2)
    assert seen==[query_tokens]
    assert torch.count_nonzero(actual[:,:dense_start])==0
    assert torch.equal(actual[:,dense_start:],expected)
