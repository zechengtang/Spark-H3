"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import math


import torch


import triton


_COMPILED={}


def exact_attention(q,k,v,kc,vs,threshold=None,route=None,scale=None,sink_start=None,sink_tokens=0,exact_only=True,force_local_blocks=True):
    """Stock selector and exact code. Optional export changes no route decisions."""
    b,t,h,d=q.shape;n=triton.cdiv(t,64);scale=d**-.5 if scale is None else scale
    sink_start=t-sink_tokens if sink_start is None else sink_start
    external=threshold is None and route is not None;hybrid=threshold is not None and route is not None
    if threshold is None:threshold=torch.zeros((b,n,h),device=q.device,dtype=torch.float32)
    actual=torch.empty((b,n,h,n),device=q.device,dtype=torch.uint8) if route is None else route.to(torch.uint8).clone()
    lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
    if torch.cuda.get_device_capability(q.device)==(12,0) and q.dtype==torch.bfloat16:
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from sol_attn.common import to_cute_tensor
        from sol_attn.sm120 import make_kernel
        out=torch.empty_like(q)
        args=[to_cute_tensor(x) for x in (q,k,v,out,kc,vs,threshold,actual,lse)]
        stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        key=(q.device.index,b,t,h,external,hybrid,exact_only,force_local_blocks)
        compiled=_COMPILED.get(key)
        if compiled is None:
            try:kernel=make_kernel(external_route=external,hybrid_route=hybrid,exact_only=exact_only,export_route=True,force_local_blocks=force_local_blocks)
            except TypeError as error:raise RuntimeError('Apply third_party/sol_attn_vaware_exact_export.patch to installed sol_attn') from error
            compiled=cute.compile(kernel,*args,scale,sink_start//64,triton.cdiv(sink_start+sink_tokens,64) if sink_tokens else sink_start//64,stream=stream,options='--enable-tvm-ffi')
            _COMPILED[key]=compiled
        compiled(*args,scale,sink_start//64,triton.cdiv(sink_start+sink_tokens,64) if sink_tokens else sink_start//64,stream=stream)
        return out,lse,actual
    if not force_local_blocks and not external:
        raise NotImplementedError("No-local-band virtual summaries require SM120 BF16 or an explicit route")
    # Existing selected-block Triton implementation, also supports fp16.
    from .sol_logsumexp_correction_triton import _average_route_kernel,_exact_gap_kernel
    out=torch.empty_like(q,dtype=torch.float32);scratch=torch.empty_like(out);al=torch.empty_like(lse);gap=torch.empty_like(lse)
    grid=(4,n,b*h)
    if not external:
        if hybrid:raise ValueError('hybrid threshold route requires SM120 BF16')
        _average_route_kernel[grid](q,kc,vs,threshold,actual,scratch,al,scale*math.log2(math.e),t,sink_start//64,triton.cdiv(sink_start+sink_tokens,64),sink_tokens>0,h,n,32,64,32,d,num_warps=4,num_stages=1)
    elif sink_tokens:
        actual[...,sink_start//64:triton.cdiv(sink_start+sink_tokens,64)]=1
    _exact_gap_kernel[grid](q,k,v,kc,actual,out,lse,gap,scale*math.log2(math.e),t,h,n,32,64,32,d,num_warps=4,num_stages=1)
    if not exact_only:raise ValueError('stock export is only available on SM120 BF16')
    return out,lse,actual

