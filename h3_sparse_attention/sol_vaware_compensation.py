"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math
import operator


import torch


import triton


_COMPILED={}
_CUTE_CAPS=((9,0),(10,0),(12,0))


def exact_attention(q,k,v,kc,vs,threshold=None,route=None,scale=None,sink_start=None,sink_tokens=0,exact_only=True,force_local_blocks=True,_query_tokens=None):
    """Stock selector and exact code. Optional export changes no route decisions."""
    b,t,h,d=q.shape;n=triton.cdiv(t,64);scale=d**-.5 if scale is None else scale
    query_tokens=t if _query_tokens is None else operator.index(_query_tokens)
    if not 0 < query_tokens <= t or (query_tokens != t and query_tokens % 64):
        raise ValueError('_query_tokens must end at a physical query-block boundary')
    query_blocks=triton.cdiv(query_tokens,64)
    sink_start=t-sink_tokens if sink_start is None else sink_start
    external=threshold is None and route is not None;hybrid=threshold is not None and route is not None
    if threshold is None:threshold=torch.zeros((b,n,h),device=q.device,dtype=torch.float32)
    actual=torch.empty((b,n,h,n),device=q.device,dtype=torch.uint8) if route is None else route.to(torch.uint8).clone()
    lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
    capability=tuple(torch.cuda.get_device_capability(q.device))
    if capability in _CUTE_CAPS and q.dtype==torch.bfloat16:
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from sol_attn.common import to_cute_tensor
        out=torch.empty_like(q)
        q_kernel=q if query_tokens==t else q[:,:query_tokens].contiguous()
        out_kernel=out if query_tokens==t else torch.empty_like(q_kernel)
        lse_kernel=lse if query_tokens==t else torch.empty(
            (b,query_tokens,h),device=q.device,dtype=torch.float32)
        stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        key=(q.device.index,capability,b,t,query_tokens,h,external,hybrid,exact_only,force_local_blocks)
        compiled=_COMPILED.get(key)
        sink_end_block=triton.cdiv(sink_start+sink_tokens,64) if sink_tokens else sink_start//64
        if capability==(9,0):
            from sol_attn.sm90 import make_kernel
            # SM90 keeps the 8-tensor ABI with the route mask as an optional
            # trailing argument and packs the sink block range into one Int32.
            args=[to_cute_tensor(x) for x in (q_kernel,k,v,out_kernel,kc,vs,threshold,lse_kernel)]
            route_arg=to_cute_tensor(actual)
            sink_range=0 if not sink_tokens else (sink_start//64)|(sink_end_block<<16)
            if compiled is None:
                try:kernel=make_kernel(t,1,external_route=external,hybrid_route=hybrid,exact_only=exact_only,export_route=True,force_local_blocks=force_local_blocks)
                except TypeError as error:raise RuntimeError('installed sol_attn lacks the SM90 route-mask extension') from error
                compiled=cute.compile(kernel,*args,scale,sink_range,stream=stream,mRouteMask=route_arg,options='--enable-tvm-ffi')
                _COMPILED[key]=compiled
            compiled(*args,scale,sink_range,stream=stream,mRouteMask=route_arg)
            if query_tokens != t:
                out[:,:query_tokens].copy_(out_kernel)
                lse[:,:query_tokens].copy_(lse_kernel)
            return out,lse,actual
        if capability==(10,0):
            from sol_attn.sm100 import make_kernel
        else:
            from sol_attn.sm120 import make_kernel
        args=[to_cute_tensor(x) for x in (q_kernel,k,v,out_kernel,kc,vs,threshold,actual,lse_kernel)]
        if compiled is None:
            try:kernel=make_kernel(external_route=external,hybrid_route=hybrid,exact_only=exact_only,export_route=True,force_local_blocks=force_local_blocks)
            except TypeError as error:raise RuntimeError('installed sol_attn lacks the route-mask extension') from error
            compiled=cute.compile(kernel,*args,scale,sink_start//64,sink_end_block,stream=stream,options='--enable-tvm-ffi')
            _COMPILED[key]=compiled
        compiled(*args,scale,sink_start//64,sink_end_block,stream=stream)
        if query_tokens != t:
            out[:,:query_tokens].copy_(out_kernel)
            lse[:,:query_tokens].copy_(lse_kernel)
        return out,lse,actual
    if not force_local_blocks and not external:
        raise NotImplementedError("No-local-band virtual summaries require an SM90/SM100/SM120 BF16 CuTe backend or an explicit route")
    # Existing selected-block Triton implementation, also supports fp16.
    from .sol_logsumexp_correction_triton import _average_route_kernel,_exact_gap_kernel
    out=torch.empty_like(q,dtype=torch.float32);scratch=torch.empty_like(out);al=torch.empty_like(lse);gap=torch.empty_like(lse)
    grid=(4,query_blocks,b*h)
    if not external:
        if hybrid:raise ValueError('hybrid threshold route requires an SM90/SM100/SM120 BF16 CuTe backend')
        _average_route_kernel[grid](q,kc,vs,threshold,actual,scratch,al,scale*math.log2(math.e),t,sink_start//64,triton.cdiv(sink_start+sink_tokens,64),sink_tokens>0,h,n,32,64,32,d,num_warps=4,num_stages=1)
    elif sink_tokens:
        actual[...,sink_start//64:triton.cdiv(sink_start+sink_tokens,64)]=1
    _exact_gap_kernel[grid](q,k,v,kc,actual,out,lse,gap,scale*math.log2(math.e),t,h,n,32,64,32,d,num_warps=4,num_stages=1)
    if not exact_only:raise ValueError('stock export is only available on SM90/SM100/SM120 BF16 CuTe backends')
    return out,lse,actual
