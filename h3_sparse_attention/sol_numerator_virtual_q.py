"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import operator


import weakref


import torch


import triton


import triton.language as tl


from .sol_vaware_compensation import exact_attention


_SUMMARY_WORKSPACE_BYTES = 1 << 30


_HOST_RANGE_CACHE = {}


def validate_virtual_layout(virtual_ranges, leaf_to_virtual, total_tokens):
    """Validate host topology, returning immutable integer tuples.

    Ranges partition all tokens and never split a physical leaf. Empty parents
    and invalid mappings are errors, including parents not containing a child.
    """
    t = operator.index(total_tokens)
    if t <= 0:
        raise ValueError('total_tokens must be positive')
    try:
        ranges = tuple(tuple(operator.index(x) for x in pair) for pair in virtual_ranges)
        mapping = tuple(operator.index(x) for x in leaf_to_virtual)
    except (TypeError, ValueError) as exc:
        raise ValueError('topology must contain integer ranges and indices') from exc
    cursor = 0
    for pair in ranges:
        if len(pair) != 2:
            raise ValueError('ranges require start,end pairs')
        start, end = pair
        if start != cursor or end <= start or end > t or start % 64 or (end != t and end % 64):
            raise ValueError('ranges must partition T with nonempty leaf-aligned intervals')
        cursor = end
    if cursor != t or len(mapping) != triton.cdiv(t, 64):
        raise ValueError('incomplete ranges or wrong leaf map length')
    for leaf, parent in enumerate(mapping):
        if parent < 0 or parent >= len(ranges):
            raise ValueError('parent index out of range')
        start, end = ranges[parent]
        if start > leaf * 64 or end < min(t, (leaf + 1) * 64):
            raise ValueError('mapped parent does not contain entire leaf')
    return ranges, mapping


@triton.jit
def _anchors(Q, RANGES, A, T:tl.constexpr, H:tl.constexpr, P:tl.constexpr):
    parent, bh = tl.program_id(0), tl.program_id(1)
    b, h = (bh // H).to(tl.int64), bh % H
    start = tl.load(RANGES + parent * 2); end = tl.load(RANGES + parent * 2 + 1)
    ii = tl.arange(0,64); d = tl.arange(0,128)
    acc = tl.zeros((128,), tl.float32)
    for offset in range(start, end, 64):
        rows = offset + ii
        q = tl.load(Q + ((b*T+rows[:,None])*H+h)*128+d[None,:], (rows<end)[:,None], 0)
        acc += tl.sum(q.to(tl.float32), 0)
    tl.store(A + ((b*P+parent)*H+h)*128+d, acc / (end-start))


@triton.jit(do_not_specialize=["P_START", "P_VALID", "HEAD_START"])
def _summaries(A,K,V,AK,AV,LM,T:tl.constexpr,H:tl.constexpr,N:tl.constexpr,P:tl.constexpr,
               A_BATCH:tl.constexpr,A_PARENT:tl.constexpr,P_START,P_VALID,
               BLOCK_P:tl.constexpr,H_SOURCE:tl.constexpr,HEAD_START):
    qa,kb,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=(bh//H).to(tl.int64),bh%H
    rr=qa*BLOCK_P+tl.arange(0,BLOCK_P);ss=kb*64+tl.arange(0,64);d=tl.arange(0,128)
    abase=(b*P+rr)*H+h
    a=tl.load(A+b*A_BATCH+(P_START+rr[:,None])*A_PARENT+(h+HEAD_START)*128+d[None,:],((rr<P_VALID)&(h+HEAD_START<H_SOURCE))[:,None],0)
    k=tl.load(K+((b*T+ss[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],((ss<T)&(h+HEAD_START<H_SOURCE))[:,None],0)
    v=tl.load(V+((b*T+ss[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],((ss<T)&(h+HEAD_START<H_SOURCE))[:,None],0)
    logits=tl.dot(a,k.T)*(.08838834764831845*1.4426950408889634)
    logits=tl.where((ss<T)[None,:],logits,-float('inf'))
    maximum=tl.max(logits,1);p=tl.exp2(logits-maximum[:,None]);den=tl.sum(p,1)
    tk=(tl.dot(p.to(k.dtype),k)/den[:,None]).to(k.dtype)
    tv=tl.dot(p.to(v.dtype),v)/den[:,None]
    base=abase.to(tl.int64)*N+kb
    tl.store(AK+base[:,None]*128+d[None,:],tk,(rr<P)[:,None])
    tl.store(AV+base[:,None]*128+d[None,:],tv,(rr<P)[:,None])
    shift=tl.sum(a.to(tl.float32)*tk.to(tl.float32),1)*.08838834764831845
    tl.store(LM+base,(maximum+tl.log2(den))*.6931471805599453-shift,rr<P)


def build_virtual_anchors(q, virtual_ranges):
    b,t,h,d=q.shape;p=virtual_ranges.shape[0]
    a=torch.empty((b,p,h,d),device=q.device,dtype=q.dtype)
    _anchors[(p,b*h)](q,virtual_ranges,a,t,h,p,num_warps=4)
    return a


def virtual_summaries(a,k,v):
    if (k.ndim != 4 or v.shape != k.shape or k.shape[-1] != 128
            or a.ndim != 4 or a.shape[0] != k.shape[0] or a.shape[2:] != k.shape[2:]
            or min(k.shape[:3]) <= 0 or a.shape[1] <= 0):
        raise ValueError('requires nonempty A[BP H128] and matching K/V[BT H128]')
    if (k.dtype not in (torch.float16,torch.bfloat16)
            or not all(x.is_cuda and x.device == k.device and x.dtype == k.dtype for x in (a,k,v))
            or not k.is_contiguous() or not v.is_contiguous()
            or a.stride(-1) != 1 or a.stride(-2) != 128):
        raise ValueError('requires matching CUDA BF16/FP16 with contiguous K/V and anchor head/features')
    b,t,h,d=k.shape;n=triton.cdiv(t,64);p=a.shape[1]
    ak=torch.empty((b,p,h,n,d),device=k.device,dtype=k.dtype);av=torch.empty_like(ak)
    lm=torch.empty((b,p,h,n),device=k.device,dtype=torch.float32)
    block_p=16 if p <= 16 else 32
    _summaries[(triton.cdiv(p,block_p),n,b*h)](
        a,k,v,ak,av,lm,t,h,n,p,a.stride(0),a.stride(1),0,p,block_p,h,0,num_warps=4)
    return ak,av,lm


def _summary_parent_chunk(b, p, h, n, d, dtype, workspace_bytes=None):
    """Largest parent count whose AK/AV/LM buffers fit the workspace."""
    if workspace_bytes is None:
        workspace_bytes=_SUMMARY_WORKSPACE_BYTES
    bytes_per_parent = b*h*n*(2*d*torch.empty((), dtype=dtype).element_size()+4)
    return max(1, min(p, workspace_bytes//bytes_per_parent))


def _host_ranges(virtual_ranges):
    """Cache the small static topology used to choose contiguous query grids."""
    key=(virtual_ranges.device.index,virtual_ranges.data_ptr(),virtual_ranges.shape[0])
    cached=_HOST_RANGE_CACHE.get(key)
    if cached is not None and cached[0]() is virtual_ranges:
        return cached[1]
    ranges=tuple(tuple(int(x) for x in pair) for pair in virtual_ranges.detach().cpu().tolist())
    _HOST_RANGE_CACHE[key]=(weakref.ref(virtual_ranges),ranges)
    return ranges


@triton.jit(do_not_specialize=["P_START", "QB_START"])
def _skip_merge_chunk(Q,MAP,AK,AV,LM,R,EO,EL,T:tl.constexpr,H:tl.constexpr,
                      N:tl.constexpr,P:tl.constexpr,P_START,
                      QB_START):
    local_qb,bh=tl.program_id(0),tl.program_id(1)
    qb=QB_START+local_qb;b,h=(bh//H).to(tl.int64),bh%H
    parent=tl.load(MAP+qb).to(tl.int64)-P_START
    rows=qb*64+tl.arange(0,64);d=tl.arange(0,128);jj=tl.arange(0,32)
    abase=(b*P+parent)*H+h
    q=tl.load(Q+((b*T+rows[:,None])*H+h)*128+d[None,:],(rows<T)[:,None],0)
    acc=tl.zeros((64,128),tl.float32);den=tl.zeros((64,),tl.float32);mx=tl.full((64,),-float('inf'),tl.float32)
    for start in range(0,N,32):
        kb=start+jj;base=abase.to(tl.int64)*N+kb
        exact=tl.load(R+((b*N+qb)*H+h)*N+kb,kb<N,1)!=0
        tk=tl.load(AK+base[:,None]*128+d[None,:],(kb<N)[:,None],0)
        tv=tl.load(AV+base[:,None]*128+d[None,:],(kb<N)[:,None],0)
        score=tl.dot(q,tk.T)*(.08838834764831845*1.4426950408889634)+tl.load(LM+base,kb<N,0)[None,:]*1.4426950408889634
        score=tl.where((kb<N)[None,:]&~exact[None,:],score,-float('inf'))
        new=tl.maximum(mx,tl.max(score,1));safe=tl.where(new==-float('inf'),0.,new)
        factor=tl.exp2(tl.where(mx==-float('inf'),-float('inf'),mx-safe));prob=tl.exp2(score-safe[:,None])
        acc=acc*factor[:,None]+tl.dot(prob.to(tv.dtype),tv);den=den*factor+tl.sum(prob,1);mx=new
    skipped=den>0
    approximate=tl.where(skipped[:,None],acc/tl.maximum(den[:,None],1e-30),0.)
    al=tl.where(skipped,(mx+tl.log2(tl.maximum(den,1e-30)))*.6931471805599453,-float('inf'))
    row=(b*T+rows)*H+h
    el=tl.load(EL+row,rows<T,-float('inf'));maximum=tl.maximum(el,al)
    we=tl.exp(el-maximum);wa=tl.exp(al-maximum)
    exact_value=tl.load(EO+row[:,None]*128+d[None,:],(rows<T)[:,None],0).to(tl.float32)
    merged=(tl.where(we[:,None]>0,we[:,None]*exact_value,0.)+
            tl.where(wa[:,None]>0,wa[:,None]*approximate,0.))/(we+wa)[:,None]
    tl.store(EO+row[:,None]*128+d[None,:],merged,(rows<T)[:,None])


def _streamed_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,route,exact_output,exact_lse,
                      precomputed_summaries=None):
    """Build parent summaries in bounded chunks and merge into exact_output."""
    b,t,h,d=q.shape;n=triton.cdiv(t,64);p=a.shape[1]
    parent_chunk=_summary_parent_chunk(b,p,h,n,d,q.dtype)
    ranges=_host_ranges(virtual_ranges)
    if precomputed_summaries is None:
        # Reuse one workspace on this stream; retain the full batch stride for
        # the final short chunk and read anchors directly from their input view.
        ak=torch.empty((b,parent_chunk,h,n,d),device=k.device,dtype=k.dtype)
        av=torch.empty_like(ak)
        lm=torch.empty((b,parent_chunk,h,n),device=k.device,dtype=torch.float32)
    for p_start in range(0,p,parent_chunk):
        p_end=min(p,p_start+parent_chunk)
        if precomputed_summaries is None:
            _summaries[(triton.cdiv(p_end-p_start,32),n,b*h)](
                a,k,v,ak,av,lm,t,h,n,parent_chunk,a.stride(0),a.stride(1),
                p_start,p_end-p_start,32,h,0,num_warps=4)
            summary_parents=parent_chunk
        else:
            ak0,av0,lm0=precomputed_summaries
            ak=ak0[:,p_start:p_end].contiguous()
            av=av0[:,p_start:p_end].contiguous()
            lm=lm0[:,p_start:p_end].contiguous()
            summary_parents=p_end-p_start
        qb_start=ranges[p_start][0]//64
        qb_end=(ranges[p_end-1][1]+63)//64
        _skip_merge_chunk[(qb_end-qb_start,b*h)](
            q,leaf_to_virtual,ak,av,lm,route,exact_output,exact_lse,
            t,h,n,summary_parents,p_start,qb_start,num_warps=4,num_stages=1)
    return exact_output


@torch.no_grad()
def virtual_q_attention(q,k,v,*,virtual_ranges,leaf_to_virtual,virtual_anchors=None,precomputed_summaries=None,key_centroids=None,value_sums=None,threshold=None,route=None,sink_start=None,sink_tokens=0,scale=None,force_local_blocks=True,**kwargs):
    if q.ndim != 4 or q.shape[-1] != 128 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('requires matching BTH128')
    b,t,h,d=q.shape;n=triton.cdiv(t,64)
    if min(b,t,h)==0 or not all(x.is_cuda and x.is_contiguous() and x.device==q.device and x.dtype==q.dtype for x in (q,k,v)) or q.dtype not in (torch.bfloat16,torch.float16):
        raise ValueError('requires nonempty contiguous matching CUDA BF16/FP16 Q/K/V')
    if scale is not None and abs(scale-d**-.5)>1e-12:
        raise ValueError('only standard 1/sqrt(128) scale supported')
    if virtual_ranges.ndim!=2 or virtual_ranges.shape[1]!=2 or virtual_ranges.shape[0]<1 or leaf_to_virtual.shape!=(n,):
        raise ValueError('expected ranges[P,2] and mapping[ceil(T/64)]')
    if not all(x.device==q.device and x.dtype==torch.int64 and x.is_contiguous() for x in (virtual_ranges,leaf_to_virtual)):
        raise ValueError('topology must be validated contiguous CUDA int64 tensors')
    if virtual_anchors is not None and (virtual_anchors.shape != (b,virtual_ranges.shape[0],h,d) or virtual_anchors.device != q.device or virtual_anchors.dtype != q.dtype or not virtual_anchors.is_contiguous()):
        raise ValueError('virtual_anchors must match contiguous BPHD Q dtype/device')
    a=build_virtual_anchors(q,virtual_ranges) if virtual_anchors is None else virtual_anchors
    if precomputed_summaries is not None:
        ak,av,lm=precomputed_summaries
        expected=(b,virtual_ranges.shape[0],h,n,d)
        if ak.shape!=expected or av.shape!=expected or lm.shape!=expected[:-1] or not all(x.device==q.device and x.is_contiguous() for x in (ak,av,lm)) or ak.dtype!=q.dtype or av.dtype!=q.dtype or lm.dtype!=torch.float32:
            raise ValueError('invalid precomputed virtual summaries')
    if virtual_q_backend(q) == 'sm120_fused_virtual_query':
        return _fused_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,key_centroids,
                              threshold,route,sink_start,sink_tokens,precomputed_summaries,
                              force_local_blocks=force_local_blocks)
    eo,el,actual=exact_attention(q,k,v,key_centroids,value_sums,threshold,route,d**-.5,sink_start,sink_tokens,force_local_blocks=force_local_blocks)
    return _streamed_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,actual,eo,el,
                             precomputed_summaries)


_FUSED_COMPILED = {}


def virtual_q_backend(q):
    """Auto-fuse production SM120 lengths; 0/1 force fallback/fusion for A/B.

    Small grids cannot amortize the CuTe invocation/extra summary-key pipeline.
    Their original streamed path remains faster in the measured 4097-row case.
    """
    import os
    selection=os.environ.get('H3_SPARK_REWEIGHT_FUSED', 'auto')
    if (q.is_cuda and q.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(q.device) == (12, 0)
            and selection != '0'
            and (selection == '1' or q.shape[1] > 8192)):
        return 'sm120_fused_virtual_query'
    return 'stock_exact+triton_virtual_query_skipped'


def _fused_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,kc,threshold,route,
                   sink_start,sink_tokens,precomputed_summaries=None,*,export_route=False,force_local_blocks=True):
    """Bounded summaries plus the production mixed exact/approximate mainloop.

    Each query CTA reads one parent's summaries. Native centroid routing and
    exact block compaction stay inside Sol-Attn; no dense route export, exact
    output, skipped output or merge pass is required.
    """
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from sol_attn.common import to_cute_tensor
    from .spark_reweight_sm120 import SparkReweightForwardSm120
    b,t,h,d=q.shape;n=triton.cdiv(t,64);p=a.shape[1]
    if kc is None:
        from sol_attn.preprocess import _reduce_kv
        kc,_=_reduce_kv(k,v)
    external=threshold is None and route is not None
    hybrid=threshold is not None and route is not None
    if threshold is None:
        threshold=torch.zeros((b,n,h),device=q.device,dtype=torch.float32)
    if route is None:
        # The scalar threshold specialization never reads or exports this.
        route=torch.empty((b,n,h,n) if export_route else (1,1,1,1),device=q.device,dtype=torch.uint8)
    else:
        route=route.to(torch.uint8).contiguous()
        if export_route:
            route=route.clone()
    out=torch.empty_like(q)
    lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
    ranges=_host_ranges(virtual_ranges)
    # Keep all query tiles for a head together. Parent-major streaming reloads
    # that head's exact K/V working set for every parent chunk, defeating L2
    # reuse in the production mainloop. Head-major streaming retains locality.
    if precomputed_summaries is None:
        bytes_per_head=b*p*n*(2*d*q.element_size()+4)
        head_chunk=max(1,min(h,_SUMMARY_WORKSPACE_BYTES//bytes_per_head))
        chunk=_summary_parent_chunk(b,p,head_chunk,n,d,q.dtype)
        ak=torch.empty((b,chunk,head_chunk,n,d),device=q.device,dtype=q.dtype)
        av=torch.empty_like(ak)
        lm=torch.empty((b,chunk,head_chunk,n),device=q.device,dtype=torch.float32)
    else:
        chunk=p
        head_chunk=h
        ak,av,lm=precomputed_summaries
    akt=ak.view(b*chunk,head_chunk,n,d).permute(0,2,1,3)
    avt=av.view(b*chunk,head_chunk,n,d).permute(0,2,1,3)
    tensors=(q,k,v,out,kc,avt,threshold,route,lse,akt,lm,leaf_to_virtual)
    args=[to_cute_tensor(x) for x in tensors]
    stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key=(q.device.index,external,hybrid,export_route,force_local_blocks,
         tuple((tuple(x.shape),tuple(x.stride()),x.dtype) for x in tensors))
    compiled=_FUSED_COMPILED.get(key)
    sink_start=t-sink_tokens if sink_start is None else sink_start
    sink_first=sink_start//64
    sink_last=triton.cdiv(sink_start+sink_tokens,64) if sink_tokens else sink_first
    for head_start in range(0,h,head_chunk):
        head_count=min(head_chunk,h-head_start)
        for p_start in range(0,p,chunk):
            p_end=min(p,p_start+chunk)
            if precomputed_summaries is None:
                _summaries[(triton.cdiv(p_end-p_start,32),n,b*head_chunk)](
                    a,k,v,ak,av,lm,t,head_chunk,n,chunk,a.stride(0),a.stride(1),
                    p_start,p_end-p_start,32,h,head_start,num_warps=4)
            qb_start=ranges[p_start][0]//64
            qb_end=triton.cdiv(ranges[p_end-1][1],64)
            scalars=(qb_start,qb_end-qb_start,p_start,head_start,head_count,
                     d**-.5,sink_first,sink_last)
            if compiled is None:
                kernel=SparkReweightForwardSm120(external_route=external,hybrid_route=hybrid,export_route=export_route,force_local_blocks=force_local_blocks)
                compiled=cute.compile(kernel,*args,*scalars,stream=stream,options='--enable-tvm-ffi')
                _FUSED_COMPILED[key]=compiled
            compiled(*args,*scalars,stream=stream)
    return (out,route,lse) if export_route else out

