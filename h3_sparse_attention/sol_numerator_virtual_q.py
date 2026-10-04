"""Shared nonuniform query-parent anchors; routing remains physical 64-token leaves.

Validate CPU topology once with ``validate_virtual_layout`` before caching its
CUDA int64 tensors. The hot path intentionally does not copy topology to CPU.
Tilted K/V use BF16 probabilities/storage as in frozen iter02; the tangent
lower-bound statement therefore applies to exact arithmetic, not rounded output.
"""
import operator
import os
import time
import weakref
import torch
import triton
import triton.language as tl
from .sol_vaware_compensation import exact_attention


# Keep the query-conditioned K/V summaries bounded at production sequence
# lengths.  The old implementation materialized [B,P,H,N,128] for every P;
# prev2 at 768p/10s therefore held about 16.2 GiB in AK/AV alone.
_SUMMARY_WORKSPACE_BYTES = 1 << 30
_HOST_RANGE_CACHE = {}


def reduce_virtual_key_centroids(k):
    """Use Sol's exact K reduction without constructing unused ordinary V sums.

    The fused virtual-query kernel consumes weighted V summaries instead.
    Reuse the official K kernel so routing and its rounding stay unchanged.
    """
    from sol_attn.preprocess import _reduce_kc_kernel
    from triton.tools.tensor_descriptor import TensorDescriptor

    batch, tokens, heads, dim = k.shape
    blocks = triton.cdiv(tokens, 64)
    tile = min(128, triton.next_power_of_2(dim))
    centroids = torch.empty((batch, blocks, heads, dim),
                            device=k.device, dtype=torch.bfloat16)
    descriptor = TensorDescriptor.from_tensor(k, [1, 64, 1, tile])
    _reduce_kc_kernel[(triton.cdiv(dim, tile), blocks, batch * heads)](
        descriptor, centroids, tokens, heads, blocks, dim, 64, tile)
    return centroids


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
               BLOCK_P:tl.constexpr,H_SOURCE:tl.constexpr,HEAD_START,
               FP32_ANCHOR:tl.constexpr,PRE_ROUND_LOGMASS:tl.constexpr,
               WEIGHTED_KV:tl.constexpr,MASS_BIAS:tl.constexpr,
               OUTPUT_NH:tl.constexpr):
    qa,kb,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=(bh//H).to(tl.int64),bh%H
    rr=qa*BLOCK_P+tl.arange(0,BLOCK_P);ss=kb*64+tl.arange(0,64);d=tl.arange(0,128)
    abase=(b*P+rr)*H+h
    a=tl.load(A+b*A_BATCH+(P_START+rr[:,None])*A_PARENT+(h+HEAD_START)*128+d[None,:],((rr<P_VALID)&(h+HEAD_START<H_SOURCE))[:,None],0)
    k=tl.load(K+((b*T+ss[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],((ss<T)&(h+HEAD_START<H_SOURCE))[:,None],0)
    v=tl.load(V+((b*T+ss[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],((ss<T)&(h+HEAD_START<H_SOURCE))[:,None],0)
    block_len=tl.minimum(64,T-kb*64).to(tl.float32)
    if WEIGHTED_KV or MASS_BIAS:
        if FP32_ANCHOR:
            logits=tl.dot(a.to(tl.float32),k.to(tl.float32).T,input_precision="tf32x3")*(.08838834764831845*1.4426950408889634)
        else:
            logits=tl.dot(a,k.T)*(.08838834764831845*1.4426950408889634)
        logits=tl.where((ss<T)[None,:],logits,-float('inf'))
        maximum=tl.max(logits,1);p=tl.exp2(logits-maximum[:,None]);den=tl.sum(p,1)
    if WEIGHTED_KV:
        tk_float=tl.dot(p.to(k.dtype),k)/den[:,None]
        tv=tl.dot(p.to(v.dtype),v)/den[:,None]
    else:
        tk_float=tl.sum(k.to(tl.float32),0)[None,:]/block_len+tl.zeros((BLOCK_P,1),tl.float32)
        tv=(tl.sum(v.to(tl.float32),0)[None,:]/block_len+tl.zeros((BLOCK_P,1),tl.float32)).to(v.dtype)
    tk=tk_float.to(k.dtype)
    mass_base=abase.to(tl.int64)*N+kb
    if OUTPUT_NH:
        base=((b*P+rr).to(tl.int64)*N+kb)*H+h
    else:
        base=mass_base
    tl.store(AK+base[:,None]*128+d[None,:],tk,(rr<P)[:,None])
    tl.store(AV+base[:,None]*128+d[None,:],tv,(rr<P)[:,None])
    if MASS_BIAS:
        shift_key=tk_float if PRE_ROUND_LOGMASS else tk.to(tl.float32)
        shift=tl.sum(a.to(tl.float32)*shift_key,1)*.08838834764831845
        logmass=(maximum+tl.log2(den))*.6931471805599453-shift
    else:
        logmass=tl.log(block_len)+tl.zeros((BLOCK_P,),tl.float32)
    tl.store(LM+mass_base,logmass,rr<P)


@triton.jit(do_not_specialize=["P_START", "P_VALID", "HEAD_START"])
def _summaries_comfy_fp32(A,K,V,AK,AV,LM,T:tl.constexpr,H:tl.constexpr,N:tl.constexpr,P:tl.constexpr,
                          A_BATCH:tl.constexpr,A_PARENT:tl.constexpr,P_START,P_VALID,
                          H_SOURCE:tl.constexpr,HEAD_START,
                          PRE_ROUND_LOGMASS:tl.constexpr,
                          WEIGHTED_KV:tl.constexpr,MASS_BIAS:tl.constexpr,
                          OUTPUT_NH:tl.constexpr):
    """Comfy-style FP32 scalar products/weighted reductions, BF16 summaries.

    This reproduces the precision stages, not the exact CUDA reduction order.
    The Sol INT8 consumer is deliberately outside this reweight ablation.
    """
    parent,kb,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=(bh//H).to(tl.int64),bh%H
    valid_parent=(parent<P_VALID)&(h+HEAD_START<H_SOURCE)
    token=kb*64+tl.arange(0,64)
    block_len=tl.minimum(64,T-kb*64).to(tl.float32)
    if WEIGHTED_KV or MASS_BIAS:
        lane=tl.arange(0,32)
        dot_lanes=tl.zeros((64,32),tl.float32)
        for chunk in tl.static_range(4):
            d=chunk*32+lane
            a32=tl.load(A+b*A_BATCH+(P_START+parent)*A_PARENT+(h+HEAD_START)*128+d,
                        valid_parent,0).to(tl.float32)
            k32=tl.load(K+((b*T+token[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],
                        ((token<T)&valid_parent)[:,None],0).to(tl.float32)
            dot_lanes=tl.fma(a32[None,:],k32,dot_lanes)
        logits=tl.sum(dot_lanes,1)*(.08838834764831845*1.4426950408889634)
        logits=tl.where((token<T)&valid_parent,logits,-float('inf'))
        maximum=tl.max(logits,0)
        safe_max=tl.where(valid_parent,maximum,0.)
        probability=tl.where((token<T)&valid_parent,tl.exp2(logits-safe_max),0.)
        denominator=tl.sum(probability,0)
        safe_den=tl.maximum(denominator,1e-30)
    d=tl.arange(0,128)
    a=tl.load(A+b*A_BATCH+(P_START+parent)*A_PARENT+(h+HEAD_START)*128+d,
              valid_parent,0).to(tl.float32)
    k=tl.load(K+((b*T+token[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],
              ((token<T)&valid_parent)[:,None],0).to(tl.float32)
    v=tl.load(V+((b*T+token[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],
              ((token<T)&valid_parent)[:,None],0).to(tl.float32)
    if WEIGHTED_KV:
        key_float=tl.sum(probability[:,None]*k,0)/safe_den
        value_float=tl.sum(probability[:,None]*v,0)/safe_den
    else:
        key_float=tl.sum(k,0)/block_len
        value_float=tl.sum(v,0)/block_len
    key_stored=key_float.to(AK.dtype.element_ty)
    value_stored=value_float.to(AV.dtype.element_ty)
    mass_offset=((b*P+parent)*H+h).to(tl.int64)*N+kb
    if OUTPUT_NH:
        offset=((b*P+parent).to(tl.int64)*N+kb)*H+h
    else:
        offset=mass_offset
    tl.store(AK+offset*128+d,key_stored,valid_parent)
    tl.store(AV+offset*128+d,value_stored,valid_parent)
    if MASS_BIAS:
        shift_key=key_float if PRE_ROUND_LOGMASS else key_stored.to(tl.float32)
        shift=tl.sum(a*shift_key,0)*.08838834764831845
        logmass=(safe_max+tl.log2(safe_den))*.6931471805599453-shift
    else:
        logmass=tl.log(block_len)
    tl.store(LM+mass_offset,logmass,valid_parent)


@triton.jit(do_not_specialize=["P_START", "P_VALID", "HEAD_START"])
def _summaries_block8(A,K,V,AK,AV,LM,T:tl.constexpr,H:tl.constexpr,N8:tl.constexpr,
                      P:tl.constexpr,A_BATCH:tl.constexpr,A_PARENT:tl.constexpr,
                      P_START,P_VALID,H_SOURCE:tl.constexpr,HEAD_START,
                      PRE_ROUND_LOGMASS:tl.constexpr,WEIGHTED_KV:tl.constexpr,
                      MASS_BIAS:tl.constexpr):
    """Anchor-conditioned summaries of eight K/V rows, only for skipped tail."""
    parent,k8,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=(bh//H).to(tl.int64),bh%H
    valid=(parent<P_VALID)&(h+HEAD_START<H_SOURCE)
    rows=k8*8+tl.arange(0,8);d=tl.arange(0,128)
    a=tl.load(A+b*A_BATCH+(P_START+parent)*A_PARENT+(h+HEAD_START)*128+d,valid,0).to(tl.float32)
    k=tl.load(K+((b*T+rows[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],
              ((rows<T)&valid)[:,None],0).to(tl.float32)
    v=tl.load(V+((b*T+rows[:,None])*H_SOURCE+h+HEAD_START)*128+d[None,:],
              ((rows<T)&valid)[:,None],0).to(tl.float32)
    count=tl.minimum(8,T-k8*8).to(tl.float32)
    if WEIGHTED_KV or MASS_BIAS:
        logits=tl.sum(k*a[None,:],1)*(.08838834764831845*1.4426950408889634)
        logits=tl.where((rows<T)&valid,logits,-float('inf'))
        maximum=tl.max(logits,0);safe_max=tl.where(valid,maximum,0.)
        prob=tl.where((rows<T)&valid,tl.exp2(logits-safe_max),0.)
        denominator=tl.maximum(tl.sum(prob,0),1e-30)
    if WEIGHTED_KV:
        key=tl.sum(prob[:,None]*k,0)/denominator
        value=tl.sum(prob[:,None]*v,0)/denominator
    else:
        key=tl.sum(k,0)/count
        value=tl.sum(v,0)/count
    key_stored=key.to(AK.dtype.element_ty)
    offset=((b*P+parent)*H+h).to(tl.int64)*N8+k8
    tl.store(AK+offset*128+d,key_stored,valid)
    tl.store(AV+offset*128+d,value.to(AV.dtype.element_ty),valid)
    if MASS_BIAS:
        shift_key=key if PRE_ROUND_LOGMASS else key_stored.to(tl.float32)
        shift=tl.sum(a*shift_key,0)*.08838834764831845
        logmass=(safe_max+tl.log2(denominator))*.6931471805599453-shift
    else:
        logmass=tl.log(count)
    tl.store(LM+offset,logmass,valid)


@triton.jit
def _skip(Q,MAP,AK,AV,LM,R,O,L,T:tl.constexpr,H:tl.constexpr,N:tl.constexpr,P:tl.constexpr):
    qb,bh=tl.program_id(0),tl.program_id(1);b,h=(bh//H).to(tl.int64),bh%H
    parent=tl.load(MAP+qb).to(tl.int64)
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
        factor=tl.exp2(tl.where(mx==-float('inf'),-float('inf'),mx-safe));p=tl.exp2(score-safe[:,None])
        acc=acc*factor[:,None]+tl.dot(p.to(tv.dtype),tv);den=den*factor+tl.sum(p,1);mx=new
    valid=den>0
    tl.store(O+((b*T+rows[:,None])*H+h)*128+d[None,:],tl.where(valid[:,None],acc/tl.maximum(den[:,None],1e-30),0.),(rows<T)[:,None])
    tl.store(L+(b*T+rows)*H+h,tl.where(valid,(mx+tl.log2(tl.maximum(den,1e-30)))*.6931471805599453,-float('inf')),rows<T)


@triton.jit
def _anchor_parts(Q, PART, T:tl.constexpr, H:tl.constexpr, N:tl.constexpr):
    block,bh=tl.program_id(0),tl.program_id(1)
    b,h=(bh//H).to(tl.int64),bh%H
    rows=block*64+tl.arange(0,64);d=tl.arange(0,128)
    q=tl.load(Q+((b*T+rows[:,None])*H+h)*128+d[None,:],(rows<T)[:,None],0)
    tl.store(PART+((b*N+block)*H+h)*128+d,tl.sum(q.to(tl.float32),0))


@triton.jit
def _anchors_from_parts(PART,RANGES,A,H:tl.constexpr,N:tl.constexpr,P:tl.constexpr):
    parent,bh=tl.program_id(0),tl.program_id(1)
    b,h=(bh//H).to(tl.int64),bh%H
    start=tl.load(RANGES+parent*2);end=tl.load(RANGES+parent*2+1)
    d=tl.arange(0,128);acc=tl.zeros((128,),tl.float32)
    # Preserve the original sequential sum-of-64-row reduction order.
    for offset in range(start,end,64):
        acc+=tl.load(PART+((b*N+offset//64)*H+h)*128+d)
    tl.store(A+((b*P+parent)*H+h)*128+d,acc/(end-start))


def build_virtual_anchors(q, virtual_ranges, *, dtype=None):
    b,t,h,d=q.shape;p=virtual_ranges.shape[0]
    dtype=q.dtype if dtype is None else dtype
    if dtype not in (q.dtype,torch.float32):
        raise ValueError('anchor dtype must match Q or be float32')
    a=torch.empty((b,p,h,d),device=q.device,dtype=dtype)
    if t >= 8192 and p <= 32:
        n=triton.cdiv(t,64)
        parts=torch.empty((b,n,h,d),device=q.device,dtype=torch.float32)
        _anchor_parts[(n,b*h)](q,parts,t,h,n,num_warps=4)
        _anchors_from_parts[(p,b*h)](parts,virtual_ranges,a,h,n,p,num_warps=4)
    else:
        _anchors[(p,b*h)](q,virtual_ranges,a,t,h,p,num_warps=4)
    return a


def _launch_summaries(a,k,v,ak,av,lm,*,parent_start,parent_count,
                      parent_stride,head_stride,head_start,summary_math,logmass_key,
                      reweight_components='full',
                      tensorcore_block_p=None,output_nh=False):
    if summary_math not in ('tensorcore','comfy_fp32'):
        raise ValueError('summary_math must be tensorcore or comfy_fp32')
    if logmass_key not in ('stored','pre_round'):
        raise ValueError('logmass_key must be stored or pre_round')
    if reweight_components not in ('full','weights_only','bias_only','none'):
        raise ValueError('invalid reweight_components')
    weighted_kv=reweight_components in ('full','weights_only')
    mass_bias=reweight_components in ('full','bias_only')
    b,t,h,_=k.shape;n=triton.cdiv(t,64)
    common=(a,k,v,ak,av,lm,t,head_stride,n,parent_stride,
            a.stride(0),a.stride(1),parent_start,parent_count)
    if summary_math=='comfy_fp32':
        if k.dtype != torch.bfloat16:
            raise ValueError('comfy_fp32 summaries require BF16 K/V')
        _summaries_comfy_fp32[(parent_count,n,b*head_stride)](
            *common,h,head_start,logmass_key=='pre_round',
            weighted_kv,mass_bias,output_nh,num_warps=4)
    else:
        block_p=tensorcore_block_p or (16 if parent_count<=16 else 32)
        _summaries[(triton.cdiv(parent_count,block_p),n,b*head_stride)](
            *common,block_p,h,head_start,a.dtype==torch.float32,
            logmass_key=='pre_round',weighted_kv,mass_bias,output_nh,
            num_warps=4)


def virtual_summaries(a,k,v,*,summary_math='tensorcore',logmass_key='stored',
                      reweight_components='full'):
    if (k.ndim != 4 or v.shape != k.shape or k.shape[-1] != 128
            or a.ndim != 4 or a.shape[0] != k.shape[0] or a.shape[2:] != k.shape[2:]
            or min(k.shape[:3]) <= 0 or a.shape[1] <= 0):
        raise ValueError('requires nonempty A[BP H128] and matching K/V[BT H128]')
    if (k.dtype not in (torch.float16,torch.bfloat16)
            or not all(x.is_cuda and x.device == k.device for x in (a,k,v))
            or a.dtype not in (k.dtype,torch.float32) or v.dtype != k.dtype
            or not k.is_contiguous() or not v.is_contiguous()
            or a.stride(-1) != 1 or a.stride(-2) != 128):
        raise ValueError('requires matching CUDA BF16/FP16 with contiguous K/V and anchor head/features')
    b,t,h,d=k.shape;n=triton.cdiv(t,64);p=a.shape[1]
    ak=torch.empty((b,p,h,n,d),device=k.device,dtype=k.dtype);av=torch.empty_like(ak)
    lm=torch.empty((b,p,h,n),device=k.device,dtype=torch.float32)
    _launch_summaries(a,k,v,ak,av,lm,parent_start=0,parent_count=p,
                      parent_stride=p,head_stride=h,head_start=0,
                      summary_math=summary_math,logmass_key=logmass_key,
                      reweight_components=reweight_components)
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


def skipped_virtual(q,leaf_to_virtual,ak,av,lm,route):
    b,t,h,d=q.shape;n=triton.cdiv(t,64);p=ak.shape[1]
    out=torch.empty_like(q,dtype=torch.float32);lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
    _skip[(n,b*h)](q,leaf_to_virtual,ak,av,lm,route,out,lse,t,h,n,p,num_warps=4,num_stages=1)
    return out,lse


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


@triton.jit(do_not_specialize=["P_START", "QB_START"])
def _skip_merge_chunk_block(Q,MAP,AK,AV,LM,R,EO,EL,T:tl.constexpr,H:tl.constexpr,
                            N:tl.constexpr,P:tl.constexpr,P_START,QB_START):
    """One pooled-tail softmax/output per 64-row query block; exact stays row-wise."""
    local_qb,bh=tl.program_id(0),tl.program_id(1)
    qb=QB_START+local_qb;b,h=(bh//H).to(tl.int64),bh%H
    parent=tl.load(MAP+qb).to(tl.int64)-P_START
    rows=qb*64+tl.arange(0,64);d=tl.arange(0,128);jj=tl.arange(0,32)
    abase=(b*P+parent)*H+h
    q=tl.load(Q+((b*T+rows[:,None])*H+h)*128+d[None,:],(rows<T)[:,None],0)
    count=tl.minimum(64,T-qb*64)
    qmean=(tl.sum(q.to(tl.float32),axis=0)/count).to(q.dtype)
    acc=tl.zeros((128,),tl.float32);den=tl.full((),0.,tl.float32)
    mx=tl.full((),-float('inf'),tl.float32)
    for start in range(0,N,32):
        kb=start+jj;base=abase.to(tl.int64)*N+kb
        exact=tl.load(R+((b*N+qb)*H+h)*N+kb,kb<N,1)!=0
        tk=tl.load(AK+base[:,None]*128+d[None,:],(kb<N)[:,None],0)
        tv=tl.load(AV+base[:,None]*128+d[None,:],(kb<N)[:,None],0)
        logmass=tl.load(LM+base,kb<N,0)
        score=tl.sum(tk.to(tl.float32)*qmean.to(tl.float32)[None,:],axis=1)*(.08838834764831845*1.4426950408889634)+logmass*1.4426950408889634
        score=tl.where((kb<N)&~exact,score,-float('inf'))
        new=tl.maximum(mx,tl.max(score,axis=0))
        safe=tl.where(new==-float('inf'),0.,new)
        factor=tl.exp2(tl.where(mx==-float('inf'),-float('inf'),mx-safe))
        prob=tl.exp2(score-safe)
        acc=acc*factor+tl.sum(prob[:,None]*tv.to(tl.float32),axis=0)
        den=den*factor+tl.sum(prob,axis=0)
        mx=new
    has_tail=den>0
    approx=tl.where(has_tail,acc/tl.maximum(den,1e-30),0.)
    al=tl.where(has_tail,(mx+tl.log2(tl.maximum(den,1e-30)))*.6931471805599453,-float('inf'))
    row=(b*T+rows)*H+h
    el=tl.load(EL+row,rows<T,-float('inf'))
    maximum=tl.maximum(el,al)
    we=tl.exp(el-maximum);wa=tl.exp(al-maximum)
    exact_value=tl.load(EO+row[:,None]*128+d[None,:],(rows<T)[:,None],0).to(tl.float32)
    merged=(tl.where(we[:,None]>0,we[:,None]*exact_value,0.)+
            tl.where(wa[:,None]>0,wa[:,None]*approx[None,:],0.))/tl.maximum((we+wa)[:,None],1e-30)
    tl.store(EO+row[:,None]*128+d[None,:],merged,(rows<T)[:,None])


@triton.jit(do_not_specialize=["P_START", "QB_START"])
def _skip_merge_chunk_block8(Q,MAP,AK,AV,LM,R,EO,EL,T:tl.constexpr,H:tl.constexpr,
                             N:tl.constexpr,N8:tl.constexpr,P:tl.constexpr,P_START,QB_START):
    """Pool eight Q rows and attend to eight-row K summaries for skipped blocks.

    The original 64x64 route and exact output are read unchanged. Only the
    skipped branch is refined, and each real Q row retains its own exact state.
    """
    local_qb,micro_q,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    qb=QB_START+local_qb;b,h=(bh//H).to(tl.int64),bh%H
    parent=tl.load(MAP+qb).to(tl.int64)-P_START
    rows=qb*64+micro_q*8+tl.arange(0,8);d=tl.arange(0,128)
    jj=tl.arange(0,32)
    abase=(b*P+parent)*H+h
    q=tl.load(Q+((b*T+rows[:,None])*H+h)*128+d[None,:],(rows<T)[:,None],0)
    count=tl.maximum(1,tl.minimum(8,T-qb*64-micro_q*8))
    qmean=(tl.sum(q.to(tl.float32),0)/count).to(q.dtype)
    acc=tl.zeros((128,),tl.float32);den=tl.full((),0.,tl.float32)
    mx=tl.full((),-float('inf'),tl.float32)
    for start in range(0,N8,32):
        k8=start+jj;base=abase.to(tl.int64)*N8+k8
        exact=tl.load(R+((b*N+qb)*H+h)*N+k8//8,k8<N8,1)!=0
        tk=tl.load(AK+base[:,None]*128+d[None,:],(k8<N8)[:,None],0)
        tv=tl.load(AV+base[:,None]*128+d[None,:],(k8<N8)[:,None],0)
        logmass=tl.load(LM+base,k8<N8,0)
        score=tl.sum(tk.to(tl.float32)*qmean.to(tl.float32)[None,:],1)*(
            .08838834764831845*1.4426950408889634)+logmass*1.4426950408889634
        score=tl.where((k8<N8)&~exact,score,-float('inf'))
        new=tl.maximum(mx,tl.max(score,0));safe=tl.where(new==-float('inf'),0.,new)
        factor=tl.exp2(tl.where(mx==-float('inf'),-float('inf'),mx-safe))
        prob=tl.exp2(score-safe)
        acc=acc*factor+tl.sum(prob[:,None]*tv.to(tl.float32),0)
        den=den*factor+tl.sum(prob,0);mx=new
    has_tail=den>0
    approx=tl.where(has_tail,acc/tl.maximum(den,1e-30),0.)
    al=tl.where(has_tail,(mx+tl.log2(tl.maximum(den,1e-30)))*.6931471805599453,-float('inf'))
    row=(b*T+rows)*H+h
    el=tl.load(EL+row,rows<T,-float('inf'));maximum=tl.maximum(el,al)
    we=tl.exp(el-maximum);wa=tl.exp(al-maximum)
    exact_value=tl.load(EO+row[:,None]*128+d[None,:],(rows<T)[:,None],0).to(tl.float32)
    merged=(tl.where(we[:,None]>0,we[:,None]*exact_value,0.)+
            tl.where(wa[:,None]>0,wa[:,None]*approx[None,:],0.))/tl.maximum((we+wa)[:,None],1e-30)
    tl.store(EO+row[:,None]*128+d[None,:],merged,(rows<T)[:,None])


@triton.jit
def _unpack_packed_route(P,R,N:tl.constexpr,QB:tl.constexpr,H:tl.constexpr,W:tl.constexpr,SIZE:tl.constexpr,
                         BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    col=idx%N;row=idx//N
    head=row%H;qb=(row//H)%N;b=row//(N*H)
    packed_row=(b*QB+qb)*H+head
    word=tl.load(P+packed_row*W+col//32,(idx<SIZE)&(qb<QB),0)
    bit=(word>>(col%32))&1
    tl.store(R+idx,bit,idx<SIZE)


def unpack_packed_route(route, blocks):
    """Expand an SM120 packed external route for the block-tail reference path."""
    if route.ndim!=4 or route.dtype!=torch.int32 or route.shape[-1]!=triton.cdiv(blocks,32):
        raise ValueError('invalid packed route shape or dtype')
    b,query_blocks,h,words=route.shape
    if not 0<query_blocks<=blocks or not route.is_cuda or not route.is_contiguous():
        raise ValueError('packed route must be contiguous CUDA [B,QB,H,ceil(N/32)]')
    dense=torch.empty((b,blocks,h,blocks),device=route.device,dtype=torch.uint8)
    size=dense.numel()
    _unpack_packed_route[(triton.cdiv(size,1024),)](
        route,dense,blocks,query_blocks,h,words,size,1024,num_warps=4)
    return dense


def _streamed_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,route,exact_output,exact_lse,
                      precomputed_summaries=None,_query_tokens=None,tail_granularity='query',
                      summary_math='tensorcore',logmass_key='stored',reweight_components='full'):
    """Build parent summaries in bounded chunks and merge into exact_output."""
    b,t,h,d=q.shape;n=triton.cdiv(t,64);p=a.shape[1]
    ranges=_host_ranges(virtual_ranges)
    query_tokens=t if _query_tokens is None else operator.index(_query_tokens)
    if not 0 < query_tokens <= t or (query_tokens != t and query_tokens % 64):
        raise ValueError('_query_tokens must end at a physical query-block boundary')
    p=next(i+1 for i,(_,end) in enumerate(ranges) if end>=query_tokens)
    micro8=tail_granularity=='block8x8'
    summary_n=triton.cdiv(t,8) if micro8 else n
    # H3 at 10s/768p leaves little headroom on an 80-GiB SM80 once model
    # weights, Q/K/V and the exact output coexist. Bound only this backend's
    # streamed summary workspace; newer fused backends keep the 1-GiB default.
    capability=tuple(torch.cuda.get_device_capability(q.device))
    workspace_bytes=(3 << 27) if capability == (8, 0) else None
    parent_chunk=_summary_parent_chunk(
        b,p,h,summary_n,d,q.dtype,workspace_bytes=workspace_bytes
    )
    if micro8 and precomputed_summaries is not None:
        raise ValueError('block8x8 requires its own eight-token K/V summaries')
    if precomputed_summaries is None:
        # Reuse one workspace on this stream; retain the full batch stride for
        # the final short chunk and read anchors directly from their input view.
        ak=torch.empty((b,parent_chunk,h,summary_n,d),device=k.device,dtype=k.dtype)
        av=torch.empty_like(ak)
        lm=torch.empty((b,parent_chunk,h,summary_n),device=k.device,dtype=torch.float32)
    for p_start in range(0,p,parent_chunk):
        p_end=min(p,p_start+parent_chunk)
        if precomputed_summaries is None:
            if micro8:
                if summary_math!='tensorcore':
                    raise ValueError('block8x8 currently supports tensorcore numeric mode only')
                _summaries_block8[(p_end-p_start,summary_n,b*h)](
                    a,k,v,ak,av,lm,t,h,summary_n,parent_chunk,
                    a.stride(0),a.stride(1),p_start,p_end-p_start,h,0,
                    logmass_key=='pre_round',
                    reweight_components in ('full','weights_only'),
                    reweight_components in ('full','bias_only'),num_warps=4)
            else:
                _launch_summaries(a,k,v,ak,av,lm,parent_start=p_start,
                                  parent_count=p_end-p_start,parent_stride=parent_chunk,
                                  head_stride=h,head_start=0,summary_math=summary_math,
                                  logmass_key=logmass_key,reweight_components=reweight_components,
                                  tensorcore_block_p=32)
            summary_parents=parent_chunk
        else:
            ak0,av0,lm0=precomputed_summaries
            ak=ak0[:,p_start:p_end].contiguous()
            av=av0[:,p_start:p_end].contiguous()
            lm=lm0[:,p_start:p_end].contiguous()
            summary_parents=p_end-p_start
        qb_start=ranges[p_start][0]//64
        qb_end=min((ranges[p_end-1][1]+63)//64,triton.cdiv(query_tokens,64))
        if micro8:
            _skip_merge_chunk_block8[(qb_end-qb_start,8,b*h)](
                q,leaf_to_virtual,ak,av,lm,route,exact_output,exact_lse,
                t,h,n,summary_n,summary_parents,p_start,qb_start,
                num_warps=4,num_stages=1)
        else:
            merge_kernel=_skip_merge_chunk_block if tail_granularity=='block' else _skip_merge_chunk
            merge_kernel[(qb_end-qb_start,b*h)](
                q,leaf_to_virtual,ak,av,lm,route,exact_output,exact_lse,
                t,h,n,summary_parents,p_start,qb_start,num_warps=4,num_stages=1)
    return exact_output


@torch.no_grad()
def virtual_q_attention(q,k,v,*,virtual_ranges,leaf_to_virtual,virtual_anchors=None,precomputed_summaries=None,key_centroids=None,value_sums=None,threshold=None,route=None,sink_start=None,sink_tokens=0,scale=None,force_local_blocks=True,tail_granularity='query',_query_tokens=None,anchor_dtype=None,summary_math='tensorcore',logmass_key='stored',reweight_components='full',**kwargs):
    fused_topk_ratio = float(kwargs.pop("fused_topk_ratio", 0.0))
    skip_external_route_qk = bool(kwargs.pop("skip_external_route_qk", False))
    if skip_external_route_qk and (route is None or route.dtype != torch.int32):
        raise ValueError('route-QK-free path requires a packed external route')
    if tail_granularity not in ('query','block','block8x8'):
        raise ValueError("tail_granularity must be 'query', 'block', or 'block8x8'")
    if summary_math not in ('tensorcore','comfy_fp32') or logmass_key not in ('stored','pre_round'):
        raise ValueError('invalid reweight summary numeric mode')
    if reweight_components not in ('full','weights_only','bias_only','none'):
        raise ValueError('invalid reweight_components')
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
    anchor_dtype=q.dtype if anchor_dtype is None else anchor_dtype
    if anchor_dtype not in (q.dtype,torch.float32):
        raise ValueError('anchor dtype must match Q or be float32')
    if virtual_anchors is not None and (virtual_anchors.shape != (b,virtual_ranges.shape[0],h,d) or virtual_anchors.device != q.device or virtual_anchors.dtype != anchor_dtype or not virtual_anchors.is_contiguous()):
        raise ValueError('virtual_anchors must match contiguous BPHD anchor dtype/device')
    a=build_virtual_anchors(q,virtual_ranges,dtype=anchor_dtype) if virtual_anchors is None else virtual_anchors
    if precomputed_summaries is not None:
        ak,av,lm=precomputed_summaries
        expected=(b,virtual_ranges.shape[0],h,n,d)
        if ak.shape!=expected or av.shape!=expected or lm.shape!=expected[:-1] or not all(x.device==q.device and x.is_contiguous() for x in (ak,av,lm)) or ak.dtype!=q.dtype or av.dtype!=q.dtype or lm.dtype!=torch.float32:
            raise ValueError('invalid precomputed virtual summaries')
    if tail_granularity=='query' and virtual_q_backend(q).endswith('fused_virtual_query'):
        return _fused_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,key_centroids,
                              threshold,route,sink_start,sink_tokens,precomputed_summaries,
                              force_local_blocks=force_local_blocks,_query_tokens=_query_tokens,
                              fused_topk_ratio=fused_topk_ratio,
                              skip_external_route_qk=skip_external_route_qk,
                              summary_math=summary_math,logmass_key=logmass_key,
                              reweight_components=reweight_components)
    if skip_external_route_qk:
        raise NotImplementedError('route-QK-free specialization requires the fused SM120 virtual-query backend')
    if fused_topk_ratio:
        raise ValueError('block tail requires an external Top-K route, not fused routing')
    if route is not None and route.dtype==torch.int32:
        route=unpack_packed_route(route,n)
    eo,el,actual=exact_attention(q,k,v,key_centroids,value_sums,threshold,route,d**-.5,sink_start,sink_tokens,force_local_blocks=force_local_blocks,_query_tokens=_query_tokens)
    return _streamed_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,actual,eo,el,
                             precomputed_summaries,_query_tokens=_query_tokens,
                             tail_granularity=tail_granularity,
                             summary_math=summary_math,logmass_key=logmass_key,
                             reweight_components=reweight_components).to(q.dtype)


_FUSED_COMPILED = {}
_FUSED_COMPILE_CALLS = 0
_FUSED_COMPILE_SECONDS = 0.0


def _dynamic_tensor_cache_signature(tensor):
    """Describe the static CuTe ABI while excluding dynamic tensor extents.

    ``sol_attn.common.to_cute_tensor`` marks every tensor layout dynamic with
    the final dimension as the unit-stride dimension.  Exact extents and
    extent-derived strides must therefore not split the host callable cache.
    Rank, dtype, stride order, broadcast strides, and singleton structure still
    affect the traced ABI/layout and remain part of the signature.
    """

    dim_order = (
        tuple(tensor.dim_order())
        if hasattr(tensor, "dim_order")
        else tuple(
            sorted(
                range(tensor.ndim),
                key=lambda axis: (abs(tensor.stride()[axis]), axis),
                reverse=True,
            )
        )
    )
    return (
        tensor.ndim,
        tensor.dtype,
        dim_order,
        tuple(stride == 0 for stride in tensor.stride()),
        tuple(size == 1 for size in tensor.shape),
    )


def fused_compile_cache_stats():
    """Return process-local fused-kernel compilation diagnostics."""

    return {
        "entries": len(_FUSED_COMPILED),
        "compile_calls": _FUSED_COMPILE_CALLS,
        "compile_seconds": _FUSED_COMPILE_SECONDS,
    }


def virtual_q_backend(q):
    """Auto-fuse production lengths; 0/1 force fallback/fusion for A/B.

    Small grids cannot amortize the CuTe invocation/extra summary-key pipeline.
    Their original streamed path remains faster in the measured 4097-row case.
    """
    import os
    selection=os.environ.get('H3_SPARK_REWEIGHT_FUSED', 'auto')
    capability=tuple(torch.cuda.get_device_capability(q.device)) if q.is_cuda else None
    if (q.is_cuda and q.dtype == torch.bfloat16
            and capability in ((8, 0), (9, 0), (10, 0), (12, 0))
            and selection != '0'
            and (selection == '1' or q.shape[1] > 8192)):
        return f'sm{capability[0]}{capability[1]}_fused_virtual_query'
    return 'stock_exact+triton_virtual_query_skipped'


def _fused_virtual(q,k,v,a,virtual_ranges,leaf_to_virtual,kc,threshold,route,
                   sink_start,sink_tokens,precomputed_summaries=None,*,export_route=False,force_local_blocks=True,_query_tokens=None,fused_topk_ratio=0.0,skip_external_route_qk=False,summary_math='tensorcore',logmass_key='stored',reweight_components='full'):
    """Bounded summaries plus the production mixed exact/approximate mainloop.

    Each query CTA reads one parent's summaries. Native centroid routing and
    exact block compaction stay inside Sol-Attn; no dense route export, exact
    output, skipped output or merge pass is required.

    Internal caller-only _query_tokens omits a suffix that the caller must
    overwrite densely before reading the output. The default computes all rows.
    """
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from sol_attn.common import to_cute_tensor
    capability=tuple(torch.cuda.get_device_capability(q.device))
    if capability==(8,0):
        from .spark_reweight_sm80 import SparkReweightForwardSm80 as FusedKernel
    elif capability==(9,0):
        from .spark_reweight_sm90 import SparkReweightForwardSm90 as FusedKernel
    elif capability==(10,0):
        from .spark_reweight_sm100 import SparkReweightForwardSm100 as FusedKernel
    else:
        from .spark_reweight_sm120 import SparkReweightForwardSm120 as FusedKernel
    # A cache hit skips constructing the kernel object, so keep SM80's ratio
    # validation before looking up a runtime-ratio specialization.
    if capability == (8,0) and not 0.0 <= fused_topk_ratio <= 1.0:
        raise ValueError("fused_topk_ratio must be in [0, 1]")
    b,t,h,d=q.shape;n=triton.cdiv(t,64);p=a.shape[1]
    ranges=_host_ranges(virtual_ranges)
    query_tokens=t if _query_tokens is None else operator.index(_query_tokens)
    if not 0 < query_tokens <= t or (query_tokens != t and query_tokens % 64):
        raise ValueError('_query_tokens must end at a physical query-block boundary')
    active_parents=next(i+1 for i,(_,end) in enumerate(ranges) if end>=query_tokens)
    if precomputed_summaries is None:
        # Only active parents need streamed summaries.  Production uses the
        # global video root (one parent); sizing from the complete 17-node tree
        # would waste more than 500 MiB at the 10s/768p shape.
        p=active_parents
    if kc is None:
        from sol_attn.preprocess import _reduce_kv
        kc,_=_reduce_kv(k,v)
    packed_external=(
        route is not None and route.dtype == torch.int32 and route.ndim == 4
    )
    if capability == (8,0) and fused_topk_ratio and n > 2048:
        raise ValueError("SM80 fused Top-K supports at most 2048 key blocks")
    if skip_external_route_qk and not packed_external:
        raise ValueError('route-QK-free path requires a packed external route')
    external=threshold is None and route is not None
    hybrid=threshold is not None and route is not None
    if packed_external and capability != (12,0):
        raise NotImplementedError("packed external routes currently require SM120")
    if packed_external and export_route:
        raise ValueError("packed external routes cannot be exported as a dense mask")
    if threshold is None:
        threshold=torch.zeros((b,n,h),device=q.device,dtype=torch.float32)
    if route is None:
        # The scalar threshold specialization never reads or exports this.
        route=torch.empty((b,n,h,n) if export_route else (1,1,1,1),device=q.device,dtype=torch.uint8)
    elif not packed_external:
        route=route.to(torch.uint8).contiguous()
        if export_route:
            route=route.clone()
    else:
        route=route.contiguous()
    # Every CTA consumes and then overwrites one disjoint Q tile.  Reuse the
    # private permuted-Q storage at the 345-frame SM80 boundary, where another
    # full BTHD destination would exceed 80 GiB.  The unlaunched dense suffix
    # remains untouched and is still available to its SDPA call.
    reuse_q=(capability == (8,0) and t > 90_000 and not export_route
             and os.environ.get('H3_SPARK_REWEIGHT_INPLACE_Q','1') != '0')
    out=(q if reuse_q
         else torch.empty_like(q))
    lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
    # Keep all query tiles for a head together. Parent-major streaming reloads
    # that head's exact K/V working set for every parent chunk, defeating L2
    # reuse in the production mainloop. Head-major streaming retains locality.
    direct_sm80_summaries = (
        capability == (8,0)
        and precomputed_summaries is None
        and os.environ.get('H3_SM80_DIRECT_SUMMARIES','1') != '0'
    )
    sm80_prefetch_summary = (
        capability != (8,0)
        or os.environ.get('H3_SM80_PREFETCH_SUMMARY','1') != '0'
    )
    sm80_skip_final_tile_barrier = (
        capability != (8,0)
        or os.environ.get('H3_SM80_SKIP_FINAL_TILE_BARRIER','1') != '0'
    )
    if precomputed_summaries is None:
        bytes_per_head=b*p*n*(2*d*q.element_size()+4)
        head_chunk=max(1,min(h,_SUMMARY_WORKSPACE_BYTES//bytes_per_head))
        chunk=_summary_parent_chunk(b,p,head_chunk,n,d,q.dtype)
        if direct_sm80_summaries:
            # Write summaries directly in the N-major layout consumed by the
            # Ampere cp.async mainloop. The H/N-permuted views are passed only
            # as raw output pointers to the stride-explicit Triton producer.
            akt=torch.empty(
                (b*chunk,n,head_chunk,d),device=q.device,dtype=q.dtype
            )
            avt=torch.empty_like(akt)
            ak=akt.view(b,chunk,n,head_chunk,d).permute(0,1,3,2,4)
            av=avt.view(b,chunk,n,head_chunk,d).permute(0,1,3,2,4)
        else:
            ak=torch.empty((b,chunk,head_chunk,n,d),device=q.device,dtype=q.dtype)
            av=torch.empty_like(ak)
        lm=torch.empty((b,chunk,head_chunk,n),device=q.device,dtype=torch.float32)
    else:
        chunk=p
        head_chunk=h
        ak,av,lm=precomputed_summaries
    if not direct_sm80_summaries:
        akt=ak.view(b*chunk,head_chunk,n,d).permute(0,2,1,3)
        avt=av.view(b*chunk,head_chunk,n,d).permute(0,2,1,3)
    if capability == (8,0) and not direct_sm80_summaries:
        # Ampere cp.async requires a statically provable 16-byte-aligned row.
        # The parent/head transpose is small (P=1 for the production global
        # policy) and making it compact avoids carrying an unprovable dynamic
        # stride into the CuTe kernel.
        akt_view, avt_view = akt, avt
        compact_akt=torch.empty(akt.shape,device=akt.device,dtype=akt.dtype)
        compact_avt=torch.empty(avt.shape,device=avt.device,dtype=avt.dtype)
        compact_akt.copy_(akt_view)
        compact_avt.copy_(avt_view)
        akt, avt = compact_akt, compact_avt
        # _reduce_kv may preserve an arbitrary stride for size-one head modes;
        # cp.async needs the canonical row stride even though PyTorch considers
        # either representation contiguous.
        expected_kc_stride=torch.empty(kc.shape,device='meta',dtype=kc.dtype).stride()
        if kc.stride()!=expected_kc_stride:
            compact_kc=torch.empty(kc.shape,device=kc.device,dtype=kc.dtype)
            compact_kc.copy_(kc)
            kc=compact_kc
    tensors=(q,k,v,out,kc,avt,threshold,route,lse,akt,lm,leaf_to_virtual)
    if capability == (8,0):
        from cutlass.cute.runtime import from_dlpack
        def sm80_tensor(x):
            value=from_dlpack(x,assumed_align=16,enable_tvm_ffi=True)
            if x.ndim==4 and x.is_contiguous() and x.shape[-1]==128:
                value=value.mark_layout_dynamic(leading_dim=3)
                # Explicit order also handles size-one batch/head modes whose
                # equal strides make automatic compact-order deduction
                # ambiguous.
                value=value.mark_compact_shape_dynamic(
                    mode=3,stride_order=x.dim_order(),divisibility=8)
            else:
                value=value.mark_layout_dynamic(leading_dim=x.ndim-1)
            return value
        args=[]
        for x in tensors:
            try:
                args.append(sm80_tensor(x))
            except RuntimeError as error:
                raise RuntimeError(
                    f"failed to describe SM80 tensor shape={tuple(x.shape)} "
                    f"stride={tuple(x.stride())} order={x.dim_order()}"
                ) from error
    else:
        args=[to_cute_tensor(x) for x in tensors]
    stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    static_video_tokens = t - sink_tokens if sink_start is None else operator.index(sink_start)
    runtime_sm80_topk_ratio = (
        capability == (8,0)
        and os.environ.get("H3_SM80_RUNTIME_TOPK_RATIO", "1") != "0"
    )
    runtime_sm120_topk_ratio = (
        capability == (12,0)
        and os.environ.get("H3_SM120_RUNTIME_TOPK_RATIO", "1") != "0"
    )
    cache_topk_ratio = (
        bool(fused_topk_ratio)
        if runtime_sm80_topk_ratio or runtime_sm120_topk_ratio
        else fused_topk_ratio
    )
    key=(q.device.index,capability,external,packed_external,skip_external_route_qk,hybrid,export_route,force_local_blocks,runtime_sm80_topk_ratio,runtime_sm120_topk_ratio,cache_topk_ratio,sm80_prefetch_summary,sm80_skip_final_tile_barrier,
         static_video_tokens,query_tokens,active_parents,chunk,head_chunk,
         tuple(_dynamic_tensor_cache_signature(x) for x in tensors))
    compiled=_FUSED_COMPILED.get(key)
    sink_start=t-sink_tokens if sink_start is None else sink_start
    sink_first=sink_start//64
    sink_last=triton.cdiv(sink_start+sink_tokens,64) if sink_tokens else sink_first
    for head_start in range(0,h,head_chunk):
        head_count=min(head_chunk,h-head_start)
        for p_start in range(0,active_parents,chunk):
            p_end=min(active_parents,p_start+chunk)
            if precomputed_summaries is None:
                _launch_summaries(a,k,v,ak,av,lm,parent_start=p_start,
                                  parent_count=p_end-p_start,parent_stride=chunk,
                                  head_stride=head_chunk,head_start=head_start,
                                  summary_math=summary_math,logmass_key=logmass_key,
                                  reweight_components=reweight_components,
                                  output_nh=direct_sm80_summaries)
                if capability == (8,0) and not direct_sm80_summaries:
                    compact_akt.copy_(akt_view)
                    compact_avt.copy_(avt_view)
            qb_start=ranges[p_start][0]//64
            qb_end=triton.cdiv(min(ranges[p_end-1][1],query_tokens),64)
            scalars=(qb_start,qb_end-qb_start,p_start,head_start,head_count,
                     d**-.5,sink_first,sink_last)
            if capability in ((8,0), (12,0)):
                scalars += (round(fused_topk_ratio * 10000),)
            if compiled is None:
                kernel=(FusedKernel(external_route=external,hybrid_route=hybrid,
                                    export_route=export_route,
                                    force_local_blocks=force_local_blocks,
                                    fused_topk_ratio=fused_topk_ratio,
                                    runtime_fused_topk_ratio=runtime_sm80_topk_ratio,
                                    prefetch_summary=sm80_prefetch_summary,
                                    skip_final_tile_barrier=sm80_skip_final_tile_barrier)
                        if capability==(8,0) else
                        FusedKernel(t,external_route=external,hybrid_route=hybrid,export_route=export_route,force_local_blocks=force_local_blocks)
                        if capability==(9,0) else
                        FusedKernel(external_route=external,packed_external_route=packed_external,
                                    skip_external_route_qk=skip_external_route_qk,
                                    fused_topk_ratio=fused_topk_ratio,
                                    runtime_fused_topk_ratio=runtime_sm120_topk_ratio,
                                    hybrid_route=hybrid,export_route=export_route,
                                    force_local_blocks=force_local_blocks)
                        if capability==(12,0) else
                        FusedKernel(external_route=external,hybrid_route=hybrid,
                                    export_route=export_route,force_local_blocks=force_local_blocks))
                global _FUSED_COMPILE_CALLS, _FUSED_COMPILE_SECONDS
                compile_started = time.perf_counter()
                compiled=cute.compile(kernel,*args,*scalars,stream=stream,options='--enable-tvm-ffi')
                _FUSED_COMPILE_CALLS += 1
                _FUSED_COMPILE_SECONDS += time.perf_counter() - compile_started
                _FUSED_COMPILED[key]=compiled
            compiled(*args,*scalars,stream=stream)
    return (out,route,lse) if export_route else out
