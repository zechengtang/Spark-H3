"""Split schedules: padding, bootstrap, lazy planning and carrier parity."""
import os
import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('H3_SPARK_SPLIT_LIBRARY'),
                            reason='CUDA and built split producer required')

@pytest.mark.parametrize('grid,tokens', [((3,10,11),361),((2,16,16),549)])
@pytest.mark.parametrize('bootstrap',[False,True])
def test_independent_schedules(grid,tokens,bootstrap):
    from comfy_kitchen.backends import cuda as ck
    from comfyui_backend import ComfyPackedLayout,ComfySparkConfig,ComfySparkController
    from comfyui_spark_split import spark_split_attention, prepare_split_attention, execute_split_attention, _core
    torch.manual_seed(31)
    h=3;video=__import__('math').prod(grid);dev='cuda'
    qkv=torch.randn(tokens,3*h*128,device=dev,dtype=torch.bfloat16)
    rope=torch.randn(1,tokens,1,32,2,2,device=dev)
    weights=(torch.ones(128,device=dev,dtype=torch.bfloat16),)*2
    ids=torch.arange(tokens,device=dev)
    layout=ComfyPackedLayout(ids,ids,grid,video,tokens,torch.zeros(video,3,device=dev),0)
    cfg=ComfySparkConfig(topk_mode='topk_ratio',topk_ratio=.2,tail_granularity='query',landmark_tree_v2_layout_reuse='q_from_k')
    km=torch.zeros(h,128,device=dev);vs=torch.full_like(km,.04)
    def provider(kind,*,indices=None,start=None,stop=None):
        r=qkv.index_select(0,indices) if indices is not None else qkv[start:stop]
        q,k,v=r.split(h*128,dim=-1)
        return {'q':q,'k':k,'kv':r[:,h*128:],'qkv':r,'qv':(q,v)}[kind]
    answers=[]
    for scheme in ('drop_k','keep_k'):
        c=ComfySparkController(cfg)
        if scheme=='drop_k':
            stages=[]
            plan=prepare_split_attention(c,provider,tokens,h,rope,weights,1e-6,layout,
                None if bootstrap else (km,vs),scheme=scheme,chunk=128,trace=stages.append)
            assert plan.full_k is None
            assert stages[-1]=='planning_done'
            assert 'workspace_ready' not in stages
            out,stats=execute_split_attention(c,provider,plan,trace=stages.append)
            perm=plan.permutation
            assert stages.index('planning_done') < stages.index('workspace_ready') < stages.index('producer_done')
        else:
            out,stats,perm=spark_split_attention(c,provider,tokens,h,rope,weights,1e-6,layout,
                None if bootstrap else (km,vs),scheme=scheme,chunk=128,return_plan=True)
        assert torch.equal(perm.sort(-1).values,torch.arange(video,device=dev)[None,None].expand(1,h,-1))
        assert torch.isfinite(out).all()
        answers.append((out,stats,perm))
        if not bootstrap:
            # Independent full-QKV producer reference; no retired Python API.
            plan=ck._C.sol_attn_plan(1,tokens,h,token_aug=0)
            ws=torch.empty(plan['total'],device=dev,dtype=torch.uint8)
            fab=ck._packed_rope_fab(rope,tokens,64)
            wrap=ck._wrap_for_dlpack;stream=torch.cuda.current_stream().cuda_stream
            ck._C.sol_producer_begin(wrap(ws),1,tokens,h,stream,token_aug=0)
            for start in range(0,tokens,128):
                chunk=qkv[start:start+128]
                ck._C.sol_producer_chunk(*(wrap(x) for x in (ws,chunk,fab,*weights,km,vs)),
                    1e-6,64,start,chunk.shape[0],1,tokens,h,stream,block_len=None,token_aug=0)
            ref_out,ref_stats=_core(ws,plan,perm,km,vs,c,layout,tokens,h)
            ref=(ref_out,*ref_stats)
            for actual,expected in zip((out,*stats),ref):
                assert torch.equal(actual,expected), (scheme, float((actual-expected).abs().max()))
    assert torch.equal(answers[0][2],answers[1][2])
    for a,b in zip((answers[0][0],*answers[0][1]),(answers[1][0],*answers[1][1])):
        assert torch.equal(a,b)

@pytest.mark.parametrize('quantized',[False,True])
def test_split_projection_returns_separate_qv(quantized):
    from comfyui_spark_split import project_parts
    from comfy_kitchen.tensor.base import QuantizedTensor
    torch.manual_seed(77)
    x=torch.randn(64,256,device='cuda',dtype=torch.bfloat16)
    weight=torch.randn(384,256,device='cuda',dtype=torch.bfloat16)
    if quantized:
        weight=QuantizedTensor.from_float(weight,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True)
    class Linear(torch.nn.Module):
        def _forward(self,x,w,b):return torch.nn.functional.linear(x,w,b)
        def forward(self,x):return self._forward(x,weight,None)
    linear=Linear()
    expected=linear(x)
    q,v=project_parts(linear,x,'qv',128)
    assert q.shape==v.shape==(64,128)
    torch.testing.assert_close(q,expected[:,:128],atol=.0625,rtol=.005)
    torch.testing.assert_close(v,expected[:,256:],atol=.0625,rtol=.005)
    assert '_forward' not in linear.__dict__
    assert torch.equal(linear(x),expected)


@pytest.mark.parametrize('quantized',[False,True])
@pytest.mark.parametrize('kind',['q','k','v','kv','qv','qkv'])
def test_split_projection_selects_head_group(quantized,kind):
    from comfyui_spark_split import project_parts, _projection_spans
    from comfy_kitchen.tensor.base import QuantizedTensor
    torch.manual_seed(78)
    heads,dim=4,128
    width=heads*dim
    x=torch.randn(32,256,device='cuda',dtype=torch.bfloat16)
    weight=torch.randn(3*width,256,device='cuda',dtype=torch.bfloat16)
    if quantized:
        weight=QuantizedTensor.from_float(weight,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True)
    class Linear(torch.nn.Module):
        def _forward(self,x,w,b):return torch.nn.functional.linear(x,w,b)
        def forward(self,x):return self._forward(x,weight,None)
    linear=Linear()
    full=linear(x)
    spans=_projection_spans(kind,width,1,3)
    pieces=[full[:,a:b] for a,b in spans]
    expected=tuple(pieces) if kind=='qv' else pieces[0] if len(pieces)==1 else torch.cat(pieces,-1)
    actual=project_parts(linear,x,kind,width,head_start=1,head_end=3)
    actual_parts=actual if isinstance(actual,tuple) else (actual,)
    expected_parts=expected if isinstance(expected,tuple) else (expected,)
    assert len(actual_parts)==len(expected_parts)
    for got,want in zip(actual_parts,expected_parts):
        torch.testing.assert_close(got,want,atol=.0625,rtol=.005)
    assert '_forward' not in linear.__dict__


def test_native_node_schemes_with_comfy_projection(monkeypatch):
    """Exercise real Comfy casting, packed layout, bootstrap and stats reuse."""
    from types import SimpleNamespace
    from comfy.ops import mixed_precision_ops
    from comfy.ldm.minimax.model import rope_rotation_table
    from comfyui_backend import ComfyPackedLayout
    from comfyui_nodes import _RunState, _native_spark_attention

    monkeypatch.delenv('H3_SPARK_SPLIT_SCHEME', raising=False)
    torch.manual_seed(41)
    t,h,video=361,2,330
    width=h*128
    projection=mixed_precision_ops().Linear(width,3*width,bias=False,device='cuda')
    projection.load_state_dict({'weight':torch.randn(3*width,width,device='cuda',dtype=torch.bfloat16)})
    projection.requires_grad_(False)
    attn=SimpleNamespace(heads=h,head_dim=128,qkv_proj=projection,
        q_norm=torch.nn.RMSNorm(128,eps=1e-5,device='cuda',dtype=torch.bfloat16),
        k_norm=torch.nn.RMSNorm(128,eps=1e-5,device='cuda',dtype=torch.bfloat16),
        out_proj=torch.nn.Identity())
    x=torch.randn(t,width,device='cuda',dtype=torch.bfloat16)
    rope=rope_rotation_table(torch.randn(t,128,device='cuda'),torch.bfloat16)
    perm=torch.cat((torch.arange(31,t,device='cuda'),torch.arange(31,device='cuda')))
    layout=ComfyPackedLayout(perm,perm.argsort(),(3,10,11),video,t,
        torch.zeros(video,3,device='cuda'),31)
    states=[_RunState.create(8,.2,.2,chunk_path='keep_k',producer_chunk=128),
            _RunState.create(8,.2,.2,producer_chunk=128,chunk_path='drop_k')]
    for state in states:
        state.spark_layout=lambda *_args:layout
    activation=SimpleNamespace(hit=lambda *_args:None)
    with torch.no_grad():
        for _ in range(2):
            outputs=[_native_spark_attention(attn,x,rope,{'minimax_h3_layout':None},0,s,activation)
                     for s in states]
            assert outputs[0].shape==(t,width)
            assert torch.isfinite(outputs[0]).all()
            assert torch.equal(*outputs)
            for a,b in zip(states[0].chunked_stats[(0,t,h)],states[1].chunked_stats[(0,t,h)]):
                assert torch.equal(a,b)
    for state in states:
        counts=state.controller.counts
        assert counts['split:bootstrap']==counts['split:steady']==1
        assert counts['split_projection:kv']==counts['split_projection:k']==3
        assert counts['split_projection:q']==2
        assert counts['sparse:spark_comfy_native_chunked']==2
    assert states[0].controller.counts['split_projection:qv']==6
    assert states[0].controller.counts['split_projection:qkv']==0
    assert states[1].controller.counts['split_projection:qkv']==6


def test_native_split_schemes_compose_with_uneven_kj_head_groups(monkeypatch):
    """KJ head groups retain split-scheme parity and the full output layout."""
    from types import SimpleNamespace
    from comfy.ops import mixed_precision_ops
    from comfy.ldm.minimax.model import rope_rotation_table
    from comfyui_backend import ComfyPackedLayout
    from comfyui_nodes import _RunState, _native_spark_attention

    torch.manual_seed(43)
    t,h,video=361,4,330
    width=h*128
    projection=mixed_precision_ops().Linear(width,3*width,bias=False,device='cuda')
    projection.load_state_dict({'weight':torch.randn(3*width,width,device='cuda',dtype=torch.bfloat16)})
    projection.requires_grad_(False)
    attn=SimpleNamespace(heads=h,head_dim=128,qkv_proj=projection,
        q_norm=torch.nn.RMSNorm(128,eps=1e-5,device='cuda',dtype=torch.bfloat16),
        k_norm=torch.nn.RMSNorm(128,eps=1e-5,device='cuda',dtype=torch.bfloat16),
        out_proj=torch.nn.Identity())
    x=torch.randn(t,width,device='cuda',dtype=torch.bfloat16)
    rope=rope_rotation_table(torch.randn(t,128,device='cuda'),torch.bfloat16)
    perm=torch.cat((torch.arange(31,t,device='cuda'),torch.arange(31,device='cuda')))
    layout=ComfyPackedLayout(perm,perm.argsort(),(3,10,11),video,t,
        torch.zeros(video,3,device='cuda'),31)
    states=[_RunState.create(3,0.,.2,producer_chunk=128,
                chunk_path=scheme) for scheme in ('drop_k','keep_k')]
    baseline=_RunState.create(3,0.,.2,producer_chunk=128,
                              chunk_path='drop_k')
    for state in (*states,baseline):state.spark_layout=lambda *_args:layout
    activation=SimpleNamespace(hit=lambda *_args:None)
    options={'minimax_h3_layout':None,'minimax_head_chunks':3}
    with torch.no_grad():
        for _ in range(2):
            outputs=[_native_spark_attention(attn,x,rope,options,0,state,activation)
                     for state in states]
            reference=_native_spark_attention(
                attn,x,rope,{'minimax_h3_layout':None},0,baseline,activation)
            assert outputs[0].shape==(t,width)
            assert torch.equal(outputs[0],outputs[1])
            assert torch.equal(outputs[0],reference)
    expected_keys={(0,t,h,0,2),(0,t,h,2,3),(0,t,h,3,4)}
    for state in states:
        assert set(state.chunked_stats)==expected_keys
        assert state.controller.counts['split:kj_head_groups']==6
        assert state.controller.counts['sparse:spark_comfy_native_chunked']==6
        assert state.controller.counts['split:bootstrap']==3
        assert state.controller.counts['split:steady']==3


@pytest.mark.parametrize("reblock,reweight", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("scheme", ["drop_k", "keep_k"])
def test_split_rejects_disabled_reblock_or_reweight_before_production(reblock, reweight, scheme):
    from comfyui_backend import ComfySparkConfig, ComfySparkController
    from comfyui_spark_split import prepare_split_attention

    controller = ComfySparkController(ComfySparkConfig())
    controller.reblock_enabled = reblock
    controller.reweight_enabled = reweight
    def provider(*args, **kwargs):
        raise AssertionError("unsupported configuration must fail before producing QKV")
    with pytest.raises(ValueError, match="default full"):
        prepare_split_attention(controller, provider, 64, 1, None, None, 1e-6,
                                None, None, scheme=scheme, chunk=64)
