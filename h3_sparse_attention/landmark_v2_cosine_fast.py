"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import os


import torch


import triton


import triton.language as tl


from triton.language.extra.cuda import libdevice


@triton.jit
def _fp16x3_dot(a, b):
    """Three FP16 products with scaled residuals; bounded FP32 operands.

    a is a normalized token tile and b is a transposed proxy direction tile.
    Scaling the residual (not the original value) preserves additional bits
    and avoids losing most residuals to FP16's subnormal range. All products
    accumulate in FP32. The second-order residual product is omitted.
    """
    ah = a.to(tl.float16)
    bh = b.to(tl.float16)
    al = ((a - ah.to(tl.float32)) * 4096.0).to(tl.float16)
    bl = ((b - bh.to(tl.float32)) * 4096.0).to(tl.float16)
    correction = tl.dot(al, bh)
    correction = tl.dot(ah, bl, correction)
    main = tl.dot(ah, bh)
    return main + correction * (1.0 / 4096.0)


@triton.jit
def _score(X, DIR, OUT, INDICES, N:tl.constexpr, S:tl.constexpr, BM:tl.constexpr,
           MODE:tl.constexpr, INDIRECT:tl.constexpr, FP8:tl.constexpr):
    b=tl.program_id(1)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    k=tl.arange(0,128)
    cols=tl.arange(0,32 if S > 16 else 16)
    if INDIRECT:
        source_rows=tl.load(INDICES+b*N+rows,rows<N,0).to(tl.int64)
    else:
        source_rows=b*N+rows
    if FP8:
        # FP8 E4M3 rows stored as uint8; decode after the load.
        x=tl.load(X+source_rows[:,None]*128+k[None,:],rows[:,None]<N,0).to(tl.float8e4nv,bitcast=True).to(tl.float32)
    else:
        x=tl.load(X+source_rows[:,None]*128+k[None,:],rows[:,None]<N,0).to(tl.float32)
    d=tl.load(DIR+b*S*128+cols[:,None]*128+k[None,:],cols[:,None]<S,0).to(tl.float32)
    norm=tl.maximum(libdevice.sqrt_rn(tl.sum(x*x,axis=1)),1.e-12)
    if MODE == 'fp16':
        # Normalize in FP32 before narrowing: raw BF16 features can exceed
        # FP16 range. Directions are unit(right)-unit(left), hence bounded.
        dot=tl.dot((x/norm[:,None]).to(tl.float16),tl.trans(d.to(tl.float16)))
    elif MODE == 'fp16x3':
        dot=_fp16x3_dot(x/norm[:,None], tl.trans(d))
    elif MODE == 'bf16':
        dot=tl.dot(x.to(tl.bfloat16),tl.trans(d.to(tl.bfloat16)))
    elif MODE == 'tf32':
        x=tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;',constraints='=f,f',args=[x],dtype=tl.float32,is_pure=True,pack=1)
        d=tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;',constraints='=f,f',args=[d],dtype=tl.float32,is_pure=True,pack=1)
        dot=tl.dot(x,tl.trans(d),input_precision='tf32')
    else:
        dot=tl.dot(x,tl.trans(d),input_precision='tf32x3')
    scores=dot if MODE == 'fp16' or MODE == 'fp16x3' else dot/norm[:,None]
    tl.store(OUT+b*N*S+rows[:,None]*S+cols[None,:],scores,(rows[:,None]<N)&(cols[None,:]<S))


def fused_cosine_scores(samples,directions,mode='tf32x3',block_m=64):
    """dot(x, unit(right)-unit(left))/max(norm(x),eps), seven directions in one pass.

    Directions already contain raw weighted-center normalization. Normalizing
    samples after their dot product changes rounding, not the cosine formula.
    FP16 instead normalizes in FP32 before casting to keep operands in range;
    its dot accumulation and output remain FP32.
    """
    assert mode in ('fp16','fp16x3','bf16','tf32','tf32x3')
    d=directions if isinstance(directions,torch.Tensor) else torch.cat(directions,dim=1).contiguous()
    if not (samples.is_cuda and samples.is_contiguous() and samples.shape[-1]==128
            and d.shape[1]<=32):
        from .landmark_v2_cosine import unit
        return torch.bmm(unit(samples),d.transpose(1,2))
    batch,n,_=samples.shape;s=d.shape[1]
    out=torch.empty((batch,n,s),device=samples.device,dtype=torch.float32)
    _score[(triton.cdiv(n,block_m),batch)](samples,d,out,samples,N=n,S=s,BM=block_m,MODE=mode,INDIRECT=False,
                                       FP8=False,num_warps=4)
    return out


def fused_cosine_scores_indexed(source, indices, directions, mode='tf32x3', *, fp8=False):
    """Same score arithmetic, loading group-one features through row indices.

    With ``fp8=True`` the source is a uint8 view of an FP8 E4M3 feature table.
    """
    if mode not in ('fp16', 'fp16x3', 'bf16', 'tf32', 'tf32x3'):
        raise ValueError("unsupported cosine precision")
    if fp8 and (source.dtype != torch.uint8 or not source.is_contiguous()):
        raise ValueError("fp8 scores require a contiguous uint8 FP8 table")
    batch, n = indices.shape
    s = directions.shape[1]
    out = torch.empty((batch, n, s), device=source.device, dtype=torch.float32)
    _score[(triton.cdiv(n,64),batch)](
        source, directions, out, indices, N=n, S=s, BM=64, MODE=mode,
        INDIRECT=True, FP8=fp8, num_warps=4,
    )
    return out


@triton.jit
def _cosine_proxy_node(
    centers,
    pair_distance,
    active_weight,
    left_capacity,
    right_capacity,
    directions,
    weight_slots,
    landmarks: tl.constexpr,
    dim: tl.constexpr,
    internal_nodes: tl.constexpr,
    node_offset: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ITERATIONS: tl.constexpr,
    WRITE_CHILDREN: tl.constexpr,
):
    """Build one complete proxy-tree depth with one program per node/row."""

    local_node = tl.program_id(0)
    batch = tl.program_id(1)
    node = node_offset + local_node
    weight_node = tl.load(weight_slots + node)
    landmark = tl.arange(0, BLOCK_K)
    dims = tl.arange(0, BLOCK_D)
    landmark_mask = landmark < landmarks
    dim_mask = dims < dim
    weight = tl.load(
        active_weight
        + batch * internal_nodes * landmarks
        + weight_node * landmarks
        + landmark,
        mask=landmark_mask,
        other=0,
    ).to(tl.int32)

    first_slot = landmark[:, None]
    second_slot = landmark[None, :]
    pair_valid = (
        (first_slot < landmarks)
        & (second_slot < landmarks)
        & (first_slot < second_slot)
        & (weight[:, None] > 0)
        & (weight[None, :] > 0)
    )
    distance = tl.load(
        pair_distance
        + batch * landmarks * landmarks
        + first_slot * landmarks
        + second_slot,
        mask=pair_valid,
        other=-float("inf"),
    )
    maximum_distance = tl.max(tl.max(distance, axis=1), axis=0)
    pair_index = first_slot * BLOCK_K + second_slot
    candidate = tl.where(
        distance == maximum_distance, pair_index, BLOCK_K * BLOCK_K
    )
    farthest = tl.min(tl.min(candidate, axis=1), axis=0)
    first = farthest // BLOCK_K
    second = farthest % BLOCK_K
    active_count = tl.sum((weight > 0).to(tl.int32), axis=0)
    only = tl.argmax((weight > 0).to(tl.int32), axis=0, tie_break_left=True)
    first = tl.where(active_count < 2, only, first)
    second = tl.where(active_count < 2, only, second)

    center = tl.load(
        centers
        + batch * landmarks * dim
        + landmark[:, None] * dim
        + dims[None, :],
        mask=landmark_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    left_center = tl.load(
        centers + batch * landmarks * dim + first * dim + dims,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    right_center = tl.load(
        centers + batch * landmarks * dim + second * dim + dims,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    left_cap = tl.load(left_capacity + node).to(tl.int32)
    right_cap = tl.load(right_capacity + node).to(tl.int32)
    sorted_landmark = landmark
    sorted_left_weight = tl.zeros((BLOCK_K,), tl.int32)
    sorted_right_weight = weight
    for _ in tl.static_range(ITERATIONS):
        left_unit=left_center/tl.maximum(libdevice.sqrt_rn(tl.sum(left_center*left_center,0)),1.e-12)
        right_unit=right_center/tl.maximum(libdevice.sqrt_rn(tl.sum(right_center*right_center,0)),1.e-12)
        direction=right_unit-left_unit
        center_norm=tl.maximum(libdevice.sqrt_rn(tl.sum(center*center,1)),1.e-12)
        delta=tl.sum((center/center_norm[:,None])*direction[None,:],axis=1)
        delta = tl.where(landmark_mask, delta, float("inf"))
        # Pack the ordered FP32 bits and landmark id into one unique key.
        # This is the exact stable ``(delta, landmark_index)`` order and
        # avoids the former O(K^2) rank reconstruction matrices.
        delta = tl.where(delta == 0.0, 0.0, delta)
        bits = delta.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        negative = (bits & 0x80000000) != 0
        ordered = tl.where(
            negative, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
        )
        sort_key = ordered * BLOCK_K + landmark.to(tl.int64)
        sorted_key = tl.sort(sort_key, dim=0, descending=False)
        sorted_landmark = (sorted_key % BLOCK_K).to(tl.int32)
        sorted_weight = tl.load(
            active_weight
            + batch * internal_nodes * landmarks
            + weight_node * landmarks
            + sorted_landmark,
            mask=sorted_landmark < landmarks,
            other=0,
        ).to(tl.int32)
        prefix_before = tl.cumsum(sorted_weight, axis=0) - sorted_weight
        take = tl.maximum(left_cap - prefix_before, 0)
        take = tl.minimum(take, sorted_weight)
        sorted_left_weight = take
        sorted_right_weight = sorted_weight - take
        sorted_center = tl.load(
            centers
            + batch * landmarks * dim
            + sorted_landmark[:, None] * dim
            + dims[None, :],
            mask=(sorted_landmark[:, None] < landmarks)
            & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        left_center = tl.sum(
            sorted_center * sorted_left_weight[:, None].to(tl.float32), axis=0
        ) / left_cap.to(tl.float32)
        right_center = tl.sum(
            sorted_center * sorted_right_weight[:, None].to(tl.float32), axis=0
        ) / right_cap.to(tl.float32)

    left_unit=left_center/tl.maximum(libdevice.sqrt_rn(tl.sum(left_center*left_center,0)),1.e-12)
    right_unit=right_center/tl.maximum(libdevice.sqrt_rn(tl.sum(right_center*right_center,0)),1.e-12)
    tl.store(directions+(batch*internal_nodes+node)*dim+dims,right_unit-left_unit,dim_mask)
    if WRITE_CHILDREN:
        tl.store(
            active_weight
            + batch * internal_nodes * landmarks
            + (2 * node + 1) * landmarks
            + sorted_landmark,
            sorted_left_weight,
            mask=sorted_landmark < landmarks,
        )
        tl.store(
            active_weight
            + batch * internal_nodes * landmarks
            + (2 * node + 2) * landmarks
            + sorted_landmark,
            sorted_right_weight,
            mask=sorted_landmark < landmarks,
        )


_CAPACITY_CACHE = {}


_ARBITRARY_PROXY_MAX_REGISTERS = 64


def build_cosine_directions(centers, weights, child_capacities, *, return_partition=False):
    """Bound large-landmark pairwise scratch to 128 parent nodes per batch."""
    if centers.shape[1] <= 128 or centers.shape[0] <= 128:
        return _build_cosine_directions_unbatched(centers, weights, child_capacities,
                                                 return_partition=return_partition)
    chunks = [_build_cosine_directions_unbatched(
        centers[start:start + 128], weights[start:start + 128], child_capacities,
        return_partition=return_partition) for start in range(0, centers.shape[0], 128)]
    if return_partition:
        return tuple(torch.cat([chunk[i] for chunk in chunks], dim=0) for i in range(3))
    return torch.cat(chunks, dim=0)


def _build_cosine_directions_unbatched(centers,weights,child_capacities, *, return_partition=False):
    """One launch per binary-tree depth; two raw weighted-mean updates per node."""
    from .landmark_v2_cosine_triton import fused_unit128
    batch,landmarks,dim=centers.shape
    children=len(child_capacities);internal=children-1
    assert 16 <= landmarks <= 256 and dim==128 and 2 <= children <= 32
    normalized=fused_unit128(centers)
    if landmarks > 128:
        distance=torch.bmm(normalized,normalized.transpose(1,2))
        torch.sub(1, distance, out=distance)
    else:
        distance=1-torch.bmm(normalized,normalized.transpose(1,2))
    storage_nodes = 2*children-1
    active=torch.empty((batch,storage_nodes,landmarks),device=centers.device,dtype=torch.int32)
    active[:,0].copy_(weights)
    key=(centers.device,tuple(child_capacities))
    from .landmark_v2_terminal import split_topology
    topology = split_topology(tuple(child_capacities))
    if key not in _CAPACITY_CACHE:
        _CAPACITY_CACHE[key] = tuple(torch.tensor(values,device=centers.device,dtype=torch.int32)
            for values in ([row[1] for row in topology], [row[2] for row in topology],
                           [row[0] for row in topology]))
    lc,rc,slots=_CAPACITY_CACHE[key]
    directions=torch.empty((batch,storage_nodes,dim),device=centers.device,dtype=torch.float32)
    # Tiny proxy trees spend more time coordinating eight warps than computing.
    # One warp changes FP32 reduction order (and potentially near-tied routes).
    # Keep the legacy arithmetic available for reproducibility-sensitive runs.
    small_proxy_fast = (
        os.environ.get("H3_LMV2_SMALL_PROXY_FAST", "1") == "1"
        and landmarks == 32 and centers.dtype == torch.bfloat16
        and children in (14, 15, 16) and sum(child_capacities) <= 1024
        and torch.cuda.get_device_capability(centers.device) == (12, 0)
    )
    proxy_warps = 1 if small_proxy_fast else (16 if landmarks > 128 else 8)
    proxy_launch_options = {}
    if (not small_proxy_fast and _ARBITRARY_PROXY_MAX_REGISTERS is not None and landmarks == 32
            and batch >= 1024 and children in (14, 15)
            and torch.cuda.get_device_capability(centers.device) == (12, 0)):
        proxy_launch_options['maxnreg'] = _ARBITRARY_PROXY_MAX_REGISTERS
    offset=0
    for level in range(topology[-1][5]+1):
        nodes=sum(row[5]==level for row in topology)
        _cosine_proxy_node[(nodes,batch)](centers,distance,active,lc,rc,directions,slots,
            landmarks=landmarks,dim=dim,internal_nodes=storage_nodes,node_offset=offset,
            BLOCK_K=triton.next_power_of_2(landmarks),BLOCK_D=128,ITERATIONS=2,
            WRITE_CHILDREN=(not small_proxy_fast or return_partition
                            or level < topology[-1][5]),num_warps=proxy_warps,
            **proxy_launch_options)
        offset+=nodes
    if return_partition:
        return directions[:, :internal], normalized, active
    return directions[:, :internal].contiguous()

