"""Independent Spark schedules: drop_k and keep_k.

Neither entry calls spark_attn_chunked. The shared Sol producer and Spark
route/exact kernels are treated as primitives; planning precedes their workspace.
"""
from __future__ import annotations
import ctypes
import dataclasses
import os
from pathlib import Path
import torch

_LIB=None
_CHECKED_PROJECTION=set()

def _library():
    global _LIB
    if _LIB is None:
        path=os.environ.get('H3_SPARK_SPLIT_LIBRARY')
        if not path:
            raise RuntimeError(
                'Spark chunked needs the split producer library. Build it with '
                'scripts/build_spark_split.py and set H3_SPARK_SPLIT_LIBRARY '
                'to the resulting library (see docs/spark_chunked.md).'
            )
        _LIB=ctypes.CDLL(path)
        _LIB.spark_split_normalize.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_float,ctypes.c_void_p]
        _LIB.spark_split_normalize.restype=ctypes.c_int
        _LIB.spark_split_produce.argtypes=[ctypes.c_void_p]*19+[ctypes.c_float]+[ctypes.c_int]*12+[ctypes.c_void_p]
        _LIB.spark_split_produce.restype=ctypes.c_int
    return _LIB


def normalize(raw,fab,weight,h,eps,out=None):
    if raw.dtype!=torch.bfloat16 or raw.stride(-1)!=1: raise ValueError('normalize requires BF16 rows')
    out=out if out is not None else torch.empty((raw.shape[0],h,128),device=raw.device,dtype=raw.dtype)
    status=_library().spark_split_normalize(raw.data_ptr(),fab.data_ptr(),weight.data_ptr(),out.data_ptr(),
        raw.shape[0],h,raw.stride(0),fab.shape[-2],eps,torch.cuda.current_stream().cuda_stream)
    if status: raise RuntimeError(f'split normalize CUDA error {status}')
    return out


def _produce(ws,p,q,k,v,fab,qw,km,vs,eps,rot,start,count,t,h,phase):
    ptr=lambda x:0 if x is None else x.data_ptr()
    offsets=['qiP','qs','kiP','ksb','vTi',None,'vcT','scratch','cen8','cens','qmean','statsV']
    status=_library().spark_split_produce(
        *(ptr(x) for x in (q,k,v,fab,qw,km,vs)),
        *(0 if name is None else ws.data_ptr()+p[name] for name in offsets),
        eps,rot,start,count,t,p['Tp'],h,p['NPAD'],p['NQ'],
        h*128 if q is None else q.stride(0),h*128 if k is None else k.stride(0),
        h*128 if v is None else v.stride(0),phase,torch.cuda.current_stream().cuda_stream)
    if status: raise RuntimeError(f'split producer CUDA error {status}')


def _tree(features,layout,cfg):
    try: from .h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    except ImportError: from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    plan=PreparedLandmarkTreeV2Permutation(
        distance=cfg.landmark_tree_v2_distance,max_children=cfg.landmark_tree_v2_children,
        fanout_mode=cfg.landmark_tree_v2_fanout_mode,final_fanout=cfg.landmark_tree_v2_final_fanout,
        root_fanout=cfg.landmark_tree_v2_root_fanout,landmark_mode='midpoint',landmark_count=32,
        midpoint_direction_mode='legacy',aggregation='linear',input_unit_means=False,metric_unit_means=False,
        order_mode=cfg.landmark_tree_v2_order_mode,group_size=1,batch=features.shape[0],
        tokens=layout.video_tokens,dim=128,grid_shape=tuple(layout.grid),initial_order='flat',
        device=features.device,return_inverse=False,compact_direct_route=True)
    perm,_=plan.run(features,allow_cuda_graph=False)
    return perm.unsqueeze(0),plan.hierarchy


def _core(ws,p,perm,km,vs,controller,layout,t,h):
    from comfy_kitchen.backends import cuda as ck
    from comfy_kitchen.allocation import allocation_context
    cfg=controller.config;dev=ws.device;ntb=p['NTB'];rtp=((ntb+63)//64)*64
    try: from .comfyui_backend import _topk_ratio
    except ImportError: from comfyui_backend import _topk_ratio
    topk=_topk_ratio(controller,layout);sink=layout.video_tokens//64
    count=ck._topk_count(ntb-ck._sink_count(ntb,sink,ntb),topk) if topk else -1
    if topk==1:count=-1
    dtype=torch.bfloat16 if cfg.global_anchor_dtype=='bfloat16' else torch.float32
    out=torch.empty((1,t,h,128),device=dev,dtype=torch.bfloat16)
    bits=torch.empty((1,h,ntb,(ntb+31)//32),device=dev,dtype=torch.int32)
    anchor=torch.empty((1,h,128),device=dev,dtype=dtype)
    ak=torch.empty((1,h,ntb,128),device=dev,dtype=torch.bfloat16);av=torch.empty_like(ak)
    lm=torch.empty((1,h,ntb),device=dev,dtype=torch.float32)
    rki=torch.empty((1,h,rtp,128),device=dev,dtype=torch.int8);rv=torch.empty_like(rki)
    rks=torch.empty((1,h,rtp,2),device=dev,dtype=torch.float32)
    # These statistics survive the block and are reused at the next timestep.
    # Keep them outside Comfy's temporary block allocation graph.
    with allocation_context():
        next_k=torch.empty_like(km)
    vmax=torch.empty_like(vs)
    scratch=out.view(torch.int8).reshape(-1)[:h*p['Tp']*128]
    scales=torch.empty(3*h*p['Tp'],device=dev,dtype=torch.float32)
    direct=controller.direct_output and layout.video_start is not None
    wrap=ck._wrap_for_dlpack
    ck._C.spark_attn_chunked_core(
        *(wrap(x) for x in (ws,out,km,vs,next_k,vmax,bits,anchor,ak,av,lm,rki,rks,rv)),
        t,h,layout.video_tokens,dtype==torch.bfloat16,count,
        -1e9 if topk==1 else cfg.tau,128**-0.5,sink,ntb,sink,ntb,
        cfg.tail_granularity=='block',True,
        False if cfg.force_local_blocks is None else cfg.force_local_blocks,
        torch.cuda.current_stream().cuda_stream,temp_carrier=wrap(scratch),temp_scales=wrap(scales),
        query_permutation=wrap(perm),key_permutation=wrap(perm),
        video_start=int(layout.video_start) if direct else 0)
    controller.output_in_original_layout=direct
    with allocation_context():
        next_vs=(vmax/127*ck._SOL_VSCALE_MARGIN).clamp_min(1e-8)
    return out,(next_k,next_vs)


@dataclasses.dataclass
class SplitAttentionPlan:
    tokens: int
    heads: int
    fab: torch.Tensor
    qw: torch.Tensor
    kw: torch.Tensor
    kmean: torch.Tensor
    vscale: torch.Tensor
    permutation: torch.Tensor
    full_k: torch.Tensor | None
    eps: float
    rot: int
    layout: object
    scheme: str
    chunk: int


@torch.no_grad()
def prepare_split_attention(controller,provider,t,h,rope,weights,eps,layout,stale_stats,
                            *,scheme,chunk=4096,trace=None):
    """Finish global reblock before allocating the attention workspace."""
    from comfy_kitchen.backends import cuda as ck
    try:
        from .h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices,batched_noncentered_metric_factor
        from .h3_sparse_attention.landmark_projection import project_bthd
        from .h3_sparse_attention.landmark_lazy_k import lazy_k_permutation
    except ImportError:
        from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices,batched_noncentered_metric_factor
        from h3_sparse_attention.landmark_projection import project_bthd
        from h3_sparse_attention.landmark_lazy_k import lazy_k_permutation
    if scheme not in ('drop_k','keep_k'):raise ValueError(scheme)
    trace=trace or (lambda stage: None)
    trace('begin')
    cfg=controller.config
    supported={
        'landmark_tree_v2_layout_reuse':'q_from_k',
        'landmark_tree_v2_moment_mode':'raw', 'landmark_tree_v2_mean_mode':'raw',
        'landmark_tree_v2_initial_order':'flat', 'landmark_tree_v2_group_size':1,
        'landmark_tree_v2_landmark_mode':'midpoint', 'landmark_tree_v2_landmark_count':32,
        'landmark_tree_v2_midpoint_direction_mode':'legacy',
        'landmark_tree_v2_distance':'cosine', 'landmark_tree_v2_aggregation':'linear',
        'landmark_tree_v2_order_mode':'parent_order',
    }
    if (not controller.reblock_enabled or not controller.reweight_enabled
            or controller.reblock_reuse_layers != 1
            or any(getattr(cfg,name)!=value for name,value in supported.items())):
        raise ValueError('split schedules require the default full q_reuse_k midpoint configuration')
    if t!=layout.sequence_length or not 0<layout.video_tokens<=t or h<=0 or chunk<=0 or chunk%64:
        raise ValueError('invalid split attention layout or producer chunk')
    dev=rope.device;rot=rope.shape[-3]*2;video=layout.video_tokens
    fab=ck._packed_rope_fab(rope,t,rot)
    qw,kw=(x.to(device=dev,dtype=torch.bfloat16).contiguous() for x in weights)
    ids=hilbert_midpoint_sample_indices(layout.grid,video//64,device=dev)
    raw=provider('q',indices=ids)
    samples=normalize(raw,fab.index_select(0,ids).contiguous(),qw,h,eps)
    factor=batched_noncentered_metric_factor(samples.permute(1,0,2).contiguous(),
        torch.arange(ids.numel(),device=dev),ridge_epsilon=cfg.rope_sol_key_ridge_epsilon).to(torch.bfloat16).contiguous()
    del raw,samples
    features=torch.empty((h,video,128),device=dev,dtype=torch.bfloat16) if scheme=='drop_k' else None
    full_k=torch.empty((t,h,128),device=dev,dtype=torch.bfloat16) if scheme=='keep_k' else None
    bootstrap=stale_stats is None
    sums=torch.zeros((h,128),device=dev,dtype=torch.float32) if bootstrap else None
    vmax=torch.zeros_like(sums) if bootstrap else None
    for start in range(0,t,chunk):
        stop=min(start+chunk,t);n=stop-start
        raw=provider('kv' if bootstrap else 'k',start=start,stop=stop)
        kraw=raw[:,:h*128] if bootstrap else raw
        kt=normalize(kraw,fab[start:stop],kw,h,eps,
                     None if full_k is None else full_k[start:stop])
        if bootstrap:
            sums.add_(kt.sum(0,dtype=torch.float32))
            vmax=torch.maximum(vmax,raw[:,h*128:].view(n,h,128).abs().amax(0).float())
        kept=max(0,min(stop,video)-start)
        if features is not None and kept:
            ft=torch.empty((h,kept,128),device=dev,dtype=torch.bfloat16)
            project_bthd(kt[:kept].unsqueeze(0),factor,out=ft,block_m=64)
            features[:,start:start+kept].copy_(ft)
            del ft
        del raw,kraw,kt
    if bootstrap:
        km=sums/t;vs=(vmax/127*ck._SOL_VSCALE_MARGIN).clamp_min(1e-8)
        del sums,vmax
    else:km,vs=stale_stats
    km=km.float().contiguous();vs=vs.float().clamp_min(1e-8).contiguous()
    trace('planning_input_ready')
    if features is not None:
        perm,hierarchy=_tree(features,layout,cfg)
        del features
    else:
        perm,hierarchy=lazy_k_permutation(full_k,factor,video,layout.grid,cfg)
    del factor
    trace('planning_done')
    controller.landmark_reblock_hierarchy=hierarchy
    controller.counts['reblock_plan_calls']+=1
    controller.counts[f'split:{scheme}']+=1
    controller.counts['split:bootstrap' if bootstrap else 'split:steady']+=1
    return SplitAttentionPlan(t,h,fab,qw,kw,km,vs,perm,full_k,eps,rot,layout,scheme,chunk)


@torch.no_grad()
def execute_split_attention(controller,provider,plan,*,trace=None):
    """Replay producer chunks, then run Spark BSA/reweight with a fixed plan."""
    from comfy_kitchen.backends import cuda as ck
    trace=trace or (lambda stage: None)
    t,h=plan.tokens,plan.heads
    fab,qw,kw=plan.fab,plan.qw,plan.kw
    km,vs,perm=plan.kmean,plan.vscale,plan.permutation
    full_k,plan.full_k=plan.full_k,None
    eps,rot,layout,scheme,chunk=plan.eps,plan.rot,plan.layout,plan.scheme,plan.chunk
    dev=fab.device
    p=ck._C.sol_attn_plan(1,t,h,token_aug=0)
    ws=torch.empty(p['total'],device=dev,dtype=torch.uint8)
    trace('workspace_ready')
    stream=torch.cuda.current_stream().cuda_stream;wrap=ck._wrap_for_dlpack
    ck._C.sol_producer_begin(wrap(ws),1,t,h,stream,token_aug=0)
    if full_k is not None:
        _produce(ws,p,None,full_k.view(t,h*128),None,fab,qw,km,vs,eps,rot,0,t,t,h,2)
        del full_k
        trace('keep_k_released')
    for start in range(0,t,chunk):
        stop=min(start+chunk,t);n=stop-start
        if scheme=='drop_k':
            raw=provider('qkv',start=start,stop=stop)
            ck._C.sol_producer_chunk(*(wrap(x) for x in (ws,raw,fab,qw,kw,km,vs)),
                eps,rot,start,n,1,t,h,stream,block_len=None,token_aug=0)
        else:
            raw=provider('qv',start=start,stop=stop)
            qr,vr=raw if isinstance(raw,tuple) else (raw[:,:h*128],raw[:,h*128:])
            _produce(ws,p,qr,None,vr,fab,qw,km,vs,eps,rot,start,n,t,h,5)
            del qr,vr
        del raw
    trace('producer_done')
    out,stats=_core(ws,p,perm,km,vs,controller,layout,t,h)
    trace('attention_done')
    controller.backend='comfy-kitchen-spark-split'
    controller.counts['comfy_kitchen_calls']+=1
    return out,stats


@torch.no_grad()
def spark_split_attention(controller,provider,t,h,rope,weights,eps,layout,stale_stats,
                          *,scheme,chunk=4096,return_plan=False,trace=None):
    plan=prepare_split_attention(controller,provider,t,h,rope,weights,eps,layout,stale_stats,
                                 scheme=scheme,chunk=chunk,trace=trace)
    out,stats=execute_split_attention(controller,provider,plan,trace=trace)
    return (out,stats,plan.permutation) if return_plan else (out,stats)


@torch.no_grad()
def selflift_drop_k_group(attn,state,layer,activation_log,project,tokens,rope,options,
                                  head_start,head_end,chunk):
    """SelfLift owns norm/projection and output projection; Spark owns planning/core.

    project(kind, native_row_indices) returns raw projections for this head group.
    Rows are mapped before projection; RoPE remains paired with original positions.
    """
    layout=state.spark_layout(options['minimax_h3_layout'],tokens,rope.device)
    ordered_rope=rope.index_select(1,layout.permutation).contiguous()
    heads=head_end-head_start
    key=(layer,tokens,int(attn.heads),head_start,head_end,'selflift',
         tuple(options.get('uuids',())))

    def provider(kind,*,start=None,stop=None,indices=None):
        selected=(layout.permutation.index_select(0,indices) if indices is not None
                  else layout.permutation[start:stop])
        result=project(kind,selected)
        state.controller.counts[f'split_projection:{kind}']+=1
        return result

    plan=prepare_split_attention(state.controller,provider,tokens,heads,ordered_rope,
        (attn.q_norm.weight,attn.k_norm.weight),attn.q_norm.eps,layout,state.chunked_stats.get(key),
        scheme='drop_k',chunk=chunk)
    # No K/K2 feature table survives this boundary. The remaining allocations
    # are the fixed plan/statistics and the subsequent Sol producer workspace.
    out,stats=execute_split_attention(state.controller,provider,plan)
    state.chunked_stats[key]=stats
    if not state.controller.output_in_original_layout:
        out=out.index_select(1,layout.inverse_permutation)
    state.controller.counts['sparse:spark_comfy_native_chunked']+=1
    state.controller.counts['selflift:drop_k_head_groups']+=1
    activation_log.hit(tokens,layout.video_tokens,tuple(torch.cuda.get_device_capability(rope.device)))
    return out.reshape(tokens,heads*int(attn.head_dim))


def _select_weight(weight,begin,end):
    from comfy_kitchen.tensor.base import QuantizedTensor
    if not isinstance(weight,QuantizedTensor):return weight[begin:end]
    if weight._layout_cls not in ('TensorWiseINT8Layout','TensorCoreConvRotW4A4Layout'):
        raise ValueError(f'Unsupported split projection weight: {weight._layout_cls}')
    params=weight._params
    if getattr(params,'transposed',False):raise ValueError('split weights must be untransposed')
    scale=params.scale
    if isinstance(scale,torch.Tensor) and scale.numel()!=1 and scale.shape[0]==weight.shape[0]:
        scale=scale[begin:end]
    params=dataclasses.replace(params,orig_shape=(end-begin,weight.shape[1]),scale=scale)
    return QuantizedTensor(weight._qdata[begin:end],weight._layout_cls,params)


def _projection_spans(kind,width,head_start=0,head_end=None,head_dim=128):
    head_end=width//head_dim if head_end is None else head_end
    if not 0<=head_start<head_end<=width//head_dim:
        raise ValueError('invalid split projection head range')
    a=head_start*head_dim;b=head_end*head_dim
    parts={'q':(0,),'k':(1,),'v':(2,),'kv':(1,2),'qv':(0,2),'qkv':(0,1,2)}[kind]
    return [(part*width+a,part*width+b) for part in parts]


def project_parts(linear,x,kind,width,*,head_start=0,head_end=None,head_dim=128):
    """Project selected Q/K/V rows for a token chunk and optional head group.

    Comfy casting, LoRA, input quantization and offload remain owned by the
    original Linear.  ``qv`` stays a tuple so keep_k never packs Q/V;
    multi-part producer inputs (``kv`` and ``qkv``) are concatenated in their
    native order.
    """
    total_heads=width//head_dim
    head_end=total_heads if head_end is None else head_end
    if kind=='qkv' and head_start==0 and head_end==total_heads:
        return linear(x).contiguous()
    spans=_projection_spans(kind,width,head_start,head_end,head_dim)
    if not hasattr(linear,'_forward'):
        raise TypeError(f'Split projection needs Comfy MixedPrecision Linear, got {type(linear)}')
    had_instance='_forward' in linear.__dict__;previous=linear.__dict__.get('_forward')
    def selected_forward(input,weight,bias):
        outputs=[torch.nn.functional.linear(input,_select_weight(weight,a,b),None if bias is None else bias[a:b])
                 for a,b in spans]
        if len(outputs)==1:return outputs[0]
        if kind=='qv':return tuple(outputs)
        return torch.cat(outputs,dim=-1)
    linear._forward=selected_forward
    try:
        result=linear(x)
        return tuple(part.contiguous() for part in result) if isinstance(result,tuple) else result.contiguous()
    finally:
        if had_instance:linear._forward=previous
        else:del linear._forward


def native_split_attention(attn,x,rope,layout,layer,state,scheme,*,head_chunks=1):
    import comfy.model_management
    import comfy.model_prefetch
    from comfy_kitchen.backends import cuda as ck
    if not hasattr(ck._C, 'spark_attn_chunked_core'):
        raise RuntimeError('Spark chunked requires a rebuilt comfy-kitchen CUDA extension')
    if int(attn.head_dim) != 128 or x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError('Spark chunked requires BF16 [tokens, hidden] input and head_dim=128')
    t=x.shape[0];h=int(attn.heads)
    head_chunks=max(1,min(int(head_chunks),h))
    ordered=rope.index_select(1,layout.permutation).contiguous()
    weights=(comfy.model_management.cast_to(attn.q_norm.weight,device=x.device),
             comfy.model_management.cast_to(attn.k_norm.weight,device=x.device))
    sizes=[h//head_chunks+(1 if index<h%head_chunks else 0) for index in range(head_chunks)]
    combined=torch.empty((1,t,h,128),device=x.device,dtype=x.dtype)
    head_start=0
    for group_index,group_heads in enumerate(sizes):
        head_end=head_start+group_heads
        key=((layer,t,h) if head_chunks==1 else
             (layer,t,h,head_start,head_end))
        def provider(kind,*,start=None,stop=None,indices=None,
                     _head_start=head_start,_head_end=head_end):
            selected=(layout.permutation.index_select(0,indices) if indices is not None
                      else layout.permutation[start:stop])
            with comfy.model_prefetch.pause_malloc_graph():
                source=x.index_select(0,selected)
                value=project_parts(attn.qkv_proj,source,kind,h*128,
                    head_start=_head_start,head_end=_head_end)
                check=(kind,_head_start,_head_end)
                if (os.environ.get('H3_SPARK_SPLIT_VALIDATE_PROJECTION') == '1'
                        and layer == 0 and check not in _CHECKED_PROJECTION):
                    full=attn.qkv_proj(source)
                    spans=_projection_spans(kind,h*128,_head_start,_head_end)
                    expected=[full[:,a:b] for a,b in spans]
                    reference=(tuple(expected) if kind=='qv' else
                               expected[0] if len(expected)==1 else torch.cat(expected,dim=-1))
                    observed=(torch.cat(value,dim=-1) if isinstance(value,tuple) else value)
                    compared=(torch.cat(reference,dim=-1) if isinstance(reference,tuple) else reference)
                    delta=(observed.float()-compared.float()).abs()
                    import json
                    print('[Spark split projection] '+json.dumps(dict(kind=kind,rows=source.shape[0],
                        heads=[_head_start,_head_end],equal=torch.equal(observed,compared),
                        max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                        weight_type=str(type(attn.qkv_proj.weight)),
                        layout=getattr(attn.qkv_proj.weight,'_layout_cls',None))),flush=True)
                    _CHECKED_PROJECTION.add(check)
                    del full,expected,reference,observed,compared,delta
            state.controller.counts[f'split_projection:{kind}']+=1
            state.controller.counts[f'split_projection_rows:{kind}']+=source.shape[0]
            return value
        out,stats=spark_split_attention(state.controller,provider,t,group_heads,ordered,weights,
            attn.q_norm.eps,layout,state.chunked_stats.get(key),scheme=scheme,
            chunk=state.producer_chunk)
        state.chunked_stats[key]=stats
        combined[:,:,head_start:head_end].copy_(out)
        state.controller.counts['sparse:spark_comfy_native_chunked']+=1
        if head_chunks>1:
            state.controller.counts['split:kj_head_groups']+=1
        head_start=head_end
    return combined
