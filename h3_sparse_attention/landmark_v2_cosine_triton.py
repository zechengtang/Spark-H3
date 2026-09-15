"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


import triton


import triton.language as tl


from triton.language.extra.cuda import libdevice


@triton.jit
def _unit128(X, Y, ROWS:tl.constexpr, BR:tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    lanes = tl.arange(0, 32)
    base = rows[:, None] * 128 + lanes[None, :] * 4
    a = tl.load(X + base, rows[:, None] < ROWS, 0).to(tl.float32)
    b = tl.load(X + base + 1, rows[:, None] < ROWS, 0).to(tl.float32)
    c = tl.load(X + base + 2, rows[:, None] < ROWS, 0).to(tl.float32)
    d = tl.load(X + base + 3, rows[:, None] < ROWS, 0).to(tl.float32)
    # Match ATen Reduce.cuh: vector4 accumulator combination is sequential,
    # followed by a decreasing-offset warp sum. Disable FMA contraction.
    partial = ((a*a + b*b) + c*c) + d*d
    denom = tl.maximum(libdevice.sqrt_rn(tl.sum(partial, 1)), 1.e-12)
    tl.store(Y + base, tl.div_rn(a, denom[:, None]), rows[:, None] < ROWS)
    tl.store(Y + base + 1, tl.div_rn(b, denom[:, None]), rows[:, None] < ROWS)
    tl.store(Y + base + 2, tl.div_rn(c, denom[:, None]), rows[:, None] < ROWS)
    tl.store(Y + base + 3, tl.div_rn(d, denom[:, None]), rows[:, None] < ROWS)


def fused_unit128(x):
    """FP32 unit rows with the ATen float-norm128 reduction order."""
    if not (x.is_cuda and x.is_contiguous() and x.shape[-1] == 128
            and x.dtype in (torch.bfloat16, torch.float32) and x.numel() // 128 >= 32):
        fp32 = x.float()
        return fp32 / fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.empty_like(x, dtype=torch.float32)
    rows = x.numel() // 128
    _unit128[(triton.cdiv(rows, 4),)](x, out, ROWS=rows, BR=4,
                                    num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _direction_unit(X, CAP, rows, ROWS:tl.constexpr, NODES:tl.constexpr, MEANS:tl.constexpr):
    lanes = tl.arange(0, 32)
    base = rows[:, None] * 128 + lanes[None, :] * 4
    a = tl.load(X + base, rows[:, None] < ROWS, 0).to(tl.float32)
    b = tl.load(X + base + 1, rows[:, None] < ROWS, 0).to(tl.float32)
    c = tl.load(X + base + 2, rows[:, None] < ROWS, 0).to(tl.float32)
    d = tl.load(X + base + 3, rows[:, None] < ROWS, 0).to(tl.float32)
    if MEANS:
        cap = tl.load(CAP + rows % NODES).to(tl.float32)[:, None]
        a = tl.div_rn(a, cap)
        b = tl.div_rn(b, cap)
        c = tl.div_rn(c, cap)
        d = tl.div_rn(d, cap)
    denom = tl.maximum(libdevice.sqrt_rn(tl.sum(((a*a + b*b) + c*c) + d*d, 1)), 1.e-12)
    return (tl.div_rn(a, denom[:, None]), tl.div_rn(b, denom[:, None]),
            tl.div_rn(c, denom[:, None]), tl.div_rn(d, denom[:, None]))


@triton.jit
def _direction128(LEFT, RIGHT, OUT, LC, RC, ROWS:tl.constexpr, BR:tl.constexpr,
                  NODES:tl.constexpr, MEANS:tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    la, lb, lc, ld = _direction_unit(LEFT, LC, rows, ROWS, NODES, MEANS)
    ra, rb, rc, rd = _direction_unit(RIGHT, RC, rows, ROWS, NODES, MEANS)
    base = rows[:, None] * 128 + tl.arange(0, 32)[None, :] * 4
    tl.store(OUT + base, ra-la, rows[:, None] < ROWS)
    tl.store(OUT + base + 1, rb-lb, rows[:, None] < ROWS)
    tl.store(OUT + base + 2, rc-lc, rows[:, None] < ROWS)
    tl.store(OUT + base + 3, rd-ld, rows[:, None] < ROWS)


def fused_direction128(left, right, left_capacity=None, right_capacity=None):
    """Exact unit(right)-unit(left), without intermediate normalized centers."""
    means = left_capacity is not None
    assert means == (right_capacity is not None)
    if not (left.shape == right.shape and left.is_cuda and right.is_cuda
            and left.device == right.device and left.is_contiguous() and right.is_contiguous()
            and left.shape[-1] == 128 and left.numel() // 128 >= 32
            and left.dtype in (torch.bfloat16, torch.float32)
            and right.dtype in (torch.bfloat16, torch.float32)):
        from .landmark_v2_cosine import unit
        if means:
            left = left / left_capacity[None, :, None]
            right = right / right_capacity[None, :, None]
        return unit(right) - unit(left)
    out = torch.empty_like(left, dtype=torch.float32)
    rows = left.numel() // 128
    _direction128[(triton.cdiv(rows, 4),)](left, right, out, left_capacity, right_capacity,
                                         ROWS=rows, BR=4, NODES=left.shape[-2], MEANS=means,
                                         num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _seeds(X, DIST, ACTIVE, LEFT, RIGHT,
           L:tl.constexpr, D:tl.constexpr, S:tl.constexpr, BD:tl.constexpr):
    b=tl.program_id(0); node=tl.program_id(1)
    ids=tl.arange(0,L)
    w=tl.load(ACTIVE+(b*S+node)*L+ids)
    distance=tl.load(DIST+b*L*L+ids[:,None]*L+ids[None,:])
    valid=(w[:,None]>0)&(w[None,:]>0)&(ids[:,None]<ids[None,:])
    values=tl.where(valid,distance,-float('inf'))
    maximum=tl.max(tl.max(values,axis=1),axis=0)
    pair_ids=ids[:,None]*L+ids[None,:]
    pair=tl.min(tl.min(tl.where(valid&(values==maximum),pair_ids,L*L),axis=1),axis=0)
    only=tl.min(tl.where(w>0,ids,L),axis=0)
    only=tl.where(only==L,0,only)
    degenerate=tl.sum((w>0).to(tl.int32),axis=0)<2
    first=tl.where(degenerate,only,pair//L)
    second=tl.where(degenerate,only,pair%L)
    d=tl.arange(0,BD)
    left=tl.load(X+(b*L+first)*D+d,d<D,0)
    right=tl.load(X+(b*L+second)*D+d,d<D,0)
    tl.store(LEFT+(b*S+node)*D+d,left,d<D)
    tl.store(RIGHT+(b*S+node)*D+d,right,d<D)


@triton.jit
def _partition(ACTIVE, ORDER, CAP, LEFT, RIGHT, LF, RF,
               L:tl.constexpr, S:tl.constexpr, FLOAT_OUTPUT:tl.constexpr):
    b=tl.program_id(0);node=tl.program_id(1)
    ids=tl.arange(0,L);base=(b*S+node)*L
    order=tl.load(ORDER+base+ids)
    weight=tl.load(ACTIVE+base+order)
    before=tl.cumsum(weight,axis=0)-weight
    cap=tl.load(CAP+node)
    take=tl.minimum(tl.maximum(cap-before,0),weight)
    tl.store(LEFT+base+order,take)
    tl.store(RIGHT+base+order,weight-take)
    if FLOAT_OUTPUT:
        tl.store(LF+base+order,take.to(tl.float32))
        tl.store(RF+base+order,(weight-take).to(tl.float32))


def proxy_seeds(x,distance,active):
    batch,nodes,landmarks=active.shape
    if landmarks not in (16,32):raise ValueError('cosine proxy expects16/32 landmarks')
    left=torch.empty((batch,nodes,x.shape[-1]),device=x.device,dtype=x.dtype)
    right=torch.empty_like(left)
    _seeds[(batch,nodes)](x,distance,active,left,right,L=landmarks,D=x.shape[-1],S=nodes,
        BD=triton.next_power_of_2(x.shape[-1]),num_warps=4)
    return left,right


def partition_weights(active,order,left_capacity,*,float_output=False):
    batch,nodes,landmarks=active.shape
    left=torch.empty_like(active);right=torch.empty_like(active)
    lf=torch.empty_like(active,dtype=torch.float32) if float_output else None
    rf=torch.empty_like(lf) if float_output else None
    _partition[(batch,nodes)](active,order.contiguous(),left_capacity,left,right,lf,rf,
        L=landmarks,S=nodes,FLOAT_OUTPUT=float_output,num_warps=1)
    return (left,right,lf,rf) if float_output else (left,right)

