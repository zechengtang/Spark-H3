"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import os


import torch


import triton


import triton.language as tl


from triton.language.extra.cuda import libdevice


from .landmark_v2_cosine_fast import _fp16x3_dot


FUSED_NODE_ENABLED = os.environ.get("H3_LMV2_FUSED_NODE", "1") == "1"


FUSED_NODE_MAX_TOKENS = int(os.environ.get("H3_LMV2_FUSED_NODE_MAX_TOKENS", "1024"))


FUSED_NODE_WARPS = os.environ.get("H3_LMV2_FUSED_NODE_WARPS")


_LANDMARKS = 32


@triton.jit
def _fused_midpoint_directions_kernel(
    source, global_indices, directions,
    N: tl.constexpr, LANDMARKS: tl.constexpr,
    INTERNAL: tl.constexpr, SPLIT_TREE: tl.constexpr,
    FP8: tl.constexpr,
):
    """Load midpoint landmarks and build the complete proxy tree in one CTA."""
    parent = tl.program_id(0)
    dims = tl.arange(0, 128)
    landmark = tl.arange(0, LANDMARKS)
    start = landmark * N // LANDMARKS
    length = (landmark + 1) * N // LANDMARKS - start
    rows = tl.load(global_indices + parent.to(tl.int64) * N + start + length // 2)
    if FP8:
        center = tl.load(
            source + rows[:, None] * 128 + dims[None, :]
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        center = tl.load(
            source + rows[:, None] * 128 + dims[None, :]
        ).to(tl.float32)
    unit = center / tl.maximum(
        libdevice.sqrt_rn(tl.sum(center * center, axis=1)), 1.0e-12
    )[:, None]
    distance = 1.0 - tl.dot(unit, tl.trans(unit), input_precision="ieee")
    weight_nodes = (length.to(tl.int32),)
    for node in tl.static_range(INTERNAL):
        direction, left_weight, right_weight = _proxy_split(
            center, unit, distance,
            weight_nodes[SPLIT_TREE[node][0]],
            SPLIT_TREE[node][1], SPLIT_TREE[node][2], LANDMARKS, 2,
        )
        weight_nodes += (left_weight, right_weight)
        tl.store(
            directions + (parent * INTERNAL + node) * 128 + dims,
            direction,
        )


def fused_midpoint_directions(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    child_capacities: tuple[int, ...],
    *,
    fp8: bool = False,
    landmarks: int = 32,
) -> torch.Tensor:
    """Exact midpoint centers plus proxy directions without intermediates."""
    expected_dtypes = (torch.uint8,) if fp8 else (torch.float16, torch.bfloat16)
    if not (
        source.is_cuda and source.ndim == 2 and source.shape[1] == 128
        and source.is_contiguous() and source.dtype in expected_dtypes
    ):
        raise ValueError("source must be a contiguous CUDA 128-wide feature table")
    if global_indices.ndim != 2 or global_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("global_indices must be int32/int64 [parents,tokens]")
    if landmarks != 32:
        raise ValueError("the fused midpoint direction path currently requires 32 landmarks")
    from .landmark_v2_terminal import split_topology
    capacities = tuple(int(value) for value in child_capacities)
    topology = split_topology(capacities)
    parents, n = global_indices.shape
    output = torch.empty(
        (parents, len(topology), 128), device=source.device, dtype=torch.float32
    )
    _fused_midpoint_directions_kernel[(parents,)](
        source, global_indices.contiguous(), output,
        N=n, LANDMARKS=landmarks, INTERNAL=len(topology),
        SPLIT_TREE=topology, FP8=fp8,
        num_warps=2 if torch.cuda.get_device_capability(source.device) == (12, 0) else 4,
    )
    return output


def use_fused_node(node_tokens: int, children: int, dim: int) -> bool:
    return (
        FUSED_NODE_ENABLED
        and dim == 128
        # Eight-child fusion wins for small nodes; larger shapes regress.
        and (children in (2, 4) or (children == 8 and node_tokens <= 512))
        and _LANDMARKS <= node_tokens <= FUSED_NODE_MAX_TOKENS
    )


@triton.jit
def _unit_row(vector):
    """FP32 unit vector with the proxy kernel's clamp."""
    return vector / tl.maximum(libdevice.sqrt_rn(tl.sum(vector * vector, 0)), 1.0e-12)


@triton.jit
def _proxy_split(center, unit, distance, weight, left_cap: tl.constexpr,
                 right_cap: tl.constexpr, LANDMARKS: tl.constexpr,
                 ITERATIONS: tl.constexpr):
    """One capacity-weighted binary proxy split; returns direction and child weights.

    ``center``/``unit`` are [K, D] FP32, ``distance`` is [K, K] cosine
    distance, ``weight`` is the [K] int32 active landmark weight at this node.
    The sorted-prefix capacity rule of ``_cosine_proxy_node`` is evaluated
    through a rank comparison, which is the same total order ``(delta, id)``.
    """
    landmark = tl.arange(0, LANDMARKS)
    first_slot = landmark[:, None]
    second_slot = landmark[None, :]
    pair_valid = (first_slot < second_slot) & (weight[:, None] > 0) & (weight[None, :] > 0)
    masked = tl.where(pair_valid, distance, -float("inf"))
    maximum_distance = tl.max(tl.max(masked, axis=1), axis=0)
    pair_index = first_slot * LANDMARKS + second_slot
    candidate = tl.where(masked == maximum_distance, pair_index, LANDMARKS * LANDMARKS)
    farthest = tl.min(tl.min(candidate, axis=1), axis=0)
    first = farthest // LANDMARKS
    second = farthest % LANDMARKS
    active_count = tl.sum((weight > 0).to(tl.int32), axis=0)
    only = tl.argmax((weight > 0).to(tl.int32), axis=0, tie_break_left=True)
    first = tl.where(active_count < 2, only, first)
    second = tl.where(active_count < 2, only, second)
    left_center = tl.sum(tl.where(landmark[:, None] == first, center, 0.0), axis=0)
    right_center = tl.sum(tl.where(landmark[:, None] == second, center, 0.0), axis=0)
    left_weight = tl.zeros((LANDMARKS,), tl.int32)
    right_weight = weight
    for _ in tl.static_range(ITERATIONS):
        direction = _unit_row(right_center) - _unit_row(left_center)
        delta = tl.sum(unit * direction[None, :], axis=1)
        # Stable (delta, landmark) order: weight of everything sorted before i.
        before = (delta[None, :] < delta[:, None]) | (
            (delta[None, :] == delta[:, None]) & (second_slot < first_slot)
        )
        prefix_before = tl.sum(tl.where(before, weight[None, :], 0), axis=1)
        take = tl.minimum(tl.maximum(left_cap - prefix_before, 0), weight)
        left_weight = take
        right_weight = weight - take
        left_center = tl.sum(center * left_weight[:, None].to(tl.float32), axis=0) / left_cap
        right_center = tl.sum(center * right_weight[:, None].to(tl.float32), axis=0) / right_cap
    direction = _unit_row(right_center) - _unit_row(left_center)
    return direction, left_weight, right_weight


@triton.jit
def _fused_proxy_directions_kernel(
    centers, root_weight, directions,
    LANDMARKS: tl.constexpr, INTERNAL: tl.constexpr,
    SPLIT_TREE: tl.constexpr,
):
    parent = tl.program_id(0)
    landmark = tl.arange(0, LANDMARKS)
    dims = tl.arange(0, 128)
    center = tl.load(
        centers + (parent * LANDMARKS + landmark[:, None]) * 128 + dims[None, :]
    ).to(tl.float32)
    unit = center / tl.maximum(
        libdevice.sqrt_rn(tl.sum(center * center, axis=1)), 1.0e-12
    )[:, None]
    # Keep seed selection aligned with fused_midpoint_directions.  A TF32
    # bmm can change the farthest landmark pair even when its numeric error is
    # small, which then changes an entire root branch.
    distance = 1.0 - tl.dot(unit, tl.trans(unit), input_precision="ieee")
    weight_nodes = (tl.load(root_weight + landmark).to(tl.int32),)
    for node in tl.static_range(INTERNAL):
        direction, left_weight, right_weight = _proxy_split(
            center, unit, distance,
            weight_nodes[SPLIT_TREE[node][0]],
            SPLIT_TREE[node][1], SPLIT_TREE[node][2], LANDMARKS, 2,
        )
        weight_nodes += (left_weight, right_weight)
        tl.store(
            directions + (parent * INTERNAL + node) * 128 + dims,
            direction,
        )


def fused_proxy_directions(
    centers: torch.Tensor,
    weights: torch.Tensor,
    child_capacities: tuple[int, ...],
) -> torch.Tensor:
    """Build all proxy directions in one CTA from legacy-rounded inputs."""
    if not (
        centers.is_cuda and centers.dtype == torch.bfloat16
        and centers.ndim == 3 and centers.shape[1:] == (32, 128)
    ):
        raise ValueError("centers must be CUDA BF16 [parents,32,128]")
    if weights.shape != centers.shape[:2] or weights.dtype != torch.int32:
        raise ValueError("weights must be int32 [parents,32]")
    from .landmark_v2_terminal import split_topology
    topology = split_topology(tuple(int(value) for value in child_capacities))
    output = torch.empty(
        (centers.shape[0], len(topology), 128),
        device=centers.device, dtype=torch.float32,
    )
    _fused_proxy_directions_kernel[(centers.shape[0],)](
        centers, weights[0], output,
        LANDMARKS=32, INTERNAL=len(topology), SPLIT_TREE=topology,
        num_warps=2 if torch.cuda.get_device_capability(centers.device) == (12, 0) else 4,
    )
    return output


@triton.jit
def _fused_node_split_kernel(
    source, global_indices, row_batch, scratch, output, tokens,
    N: tl.constexpr, BLOCK_N: tl.constexpr, MAX_LEN: tl.constexpr,
    LANDMARKS: tl.constexpr, CHILDREN: tl.constexpr, INTERNAL: tl.constexpr,
    CHILD_OFFSETS: tl.constexpr,
    BLOCK_T: tl.constexpr, ROWS_PER_STEP: tl.constexpr, MODE: tl.constexpr,
    FP8: tl.constexpr,
    MIDPOINT: tl.constexpr = False,
    SPLIT_TREE: tl.constexpr = (),
    INDEX_BITS: tl.constexpr = 0,
):
    parent = tl.program_id(0)
    dims = tl.arange(0, 128)
    landmark = tl.arange(0, LANDMARKS)
    base = global_indices + parent.to(tl.int64) * N

    # 1. Interval means in current token order.  ROWS_PER_STEP rows per
    # landmark are gathered in one load for memory-level parallelism, then
    # added one at a time so the accumulation order matches the unfused
    # row-by-row kernel exactly (adding masked zeros is exact in FP32).
    start = landmark * N // LANDMARKS
    length = (landmark + 1) * N // LANDMARKS - start
    sample_start = start + length // 2 if MIDPOINT else start
    sample_length = tl.full((LANDMARKS,), 1, tl.int32) if MIDPOINT else length
    total = tl.zeros((LANDMARKS, 128), tl.float32)
    step_rows = tl.arange(0, ROWS_PER_STEP)
    for offset in tl.static_range(0, 1 if MIDPOINT else MAX_LEN, ROWS_PER_STEP):
        local_rows = offset + step_rows
        active = local_rows[None, :] < sample_length[:, None]
        rows = tl.load(base + sample_start[:, None] + local_rows[None, :], mask=active, other=0)
        if FP8:
            value = tl.load(source + rows[:, :, None] * 128 + dims[None, None, :],
                            mask=active[:, :, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            value = tl.load(source + rows[:, :, None] * 128 + dims[None, None, :],
                            mask=active[:, :, None], other=0.0).to(tl.float32)
        if ROWS_PER_STEP == 1:
            total += tl.sum(value, axis=1)
        else:
            for step in tl.static_range(ROWS_PER_STEP):
                total += tl.sum(tl.where((step_rows == step)[None, :, None], value, 0.0), axis=1)
    if FP8:
        # The unfused FP8 path stores BF16 centers; round the same way.
        center = (total / sample_length[:, None]).to(tl.bfloat16).to(tl.float32)
    else:
        center = (total / sample_length[:, None]).to(source.dtype.element_ty).to(tl.float32)

    # 2. Proxy tree.
    unit = center / tl.maximum(libdevice.sqrt_rn(tl.sum(center * center, axis=1)), 1.0e-12)[:, None]
    distance = 1.0 - tl.dot(unit, tl.trans(unit), input_precision="ieee")
    weight = length.to(tl.int32)
    # Fit the landmarks once. A branch's proxy weights stop at one child;
    # only internal branches get directions and enter subsequent depths.
    weight_nodes = (weight,)
    columns = tl.arange(0, 16)
    directions = tl.full((16,128),0.0,tl.float32)
    for node in tl.static_range(INTERNAL):
        direction, lw, rw = _proxy_split(center,unit,distance,
            weight_nodes[SPLIT_TREE[node][0]],SPLIT_TREE[node][1],SPLIT_TREE[node][2],LANDMARKS,2)
        weight_nodes += (lw,rw)
        directions = tl.where((columns == node)[:,None],direction[None,:],directions)

    # 3. Cosine proxy scores for every token, tile by tile.
    scratch_base = scratch + parent.to(tl.int64) * N * INTERNAL
    for tile in tl.static_range(0, BLOCK_N, BLOCK_T):
        rows = tile + tl.arange(0, BLOCK_T)
        active = rows < N
        source_rows = tl.load(base + rows, mask=active, other=0)
        if FP8:
            x = tl.load(source + source_rows[:, None] * 128 + dims[None, :],
                        mask=active[:, None], other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            x = tl.load(source + source_rows[:, None] * 128 + dims[None, :],
                        mask=active[:, None], other=0.0).to(tl.float32)
        norm = tl.maximum(libdevice.sqrt_rn(tl.sum(x * x, axis=1)), 1.0e-12)
        if MODE == "fp16":
            # Keep raw features in FP32 until normalization bounds them.
            dot = tl.dot((x / norm[:, None]).to(tl.float16),
                         tl.trans(directions.to(tl.float16)))
        elif MODE == "fp16x3":
            dot = _fp16x3_dot(x / norm[:, None], tl.trans(directions))
        elif MODE == "bf16":
            dot = tl.dot(x.to(tl.bfloat16), tl.trans(directions.to(tl.bfloat16)))
        elif MODE == "tf32":
            dot = tl.dot(x, tl.trans(directions), input_precision="tf32")
        else:
            dot = tl.dot(x, tl.trans(directions), input_precision="tf32x3")
        scores = dot if MODE == "fp16" or MODE == "fp16x3" else dot / norm[:, None]
        tl.store(scratch_base + rows[:, None] * INTERNAL + columns[None, :], scores,
                 mask=active[:, None] & (columns < INTERNAL)[None, :])

    # The score tiles and the routing vectors use different thread layouts;
    # make the scratch stores visible before any thread reads them back.
    tl.debug_barrier()

    # 4. Exact-capacity routing.
    positions = tl.arange(0, BLOCK_N)
    valid = positions < N
    global_ids = tl.load(base + positions, mask=valid, other=0)
    original = global_ids - tl.load(row_batch + parent) * tokens
    infinite = tl.full((BLOCK_N,), 9223372036854775807, tl.int64)
    tag = tl.zeros((BLOCK_N,),tl.int32)
    # One sort per binary depth, not one sort per internal node. Pack
    # (node, ordered FP32 score, original id) into an exact int64 key.
    for depth in tl.static_range(SPLIT_TREE[-1][5]+1):
        active = valid & (tag >= 0)
        score = tl.load(scratch_base + positions*INTERNAL + tl.maximum(tag,0),mask=active,other=0.0)
        bits = score.to(tl.int32,bitcast=True).to(tl.int64) & 0xFFFFFFFF
        ordered_bits = tl.where((bits & 0x80000000)!=0,(~bits)&0xFFFFFFFF,bits^0x80000000)
        key = (tag.to(tl.int64) << (32+INDEX_BITS)) | (ordered_bits << INDEX_BITS) | original
        ordered = tl.sort(tl.where(active,key,infinite),dim=0,descending=False)
        offset = 0
        next_tag = tag
        for node in tl.static_range(INTERNAL):
            if SPLIT_TREE[node][5] == depth:
                cutoff = tl.sum(tl.where(positions==offset+SPLIT_TREE[node][1]-1,ordered,0),axis=0)
                next_tag = tl.where(active & (tag==node),
                    tl.where(key<=cutoff,SPLIT_TREE[node][3],SPLIT_TREE[node][4]),next_tag)
                offset += SPLIT_TREE[node][1]+SPLIT_TREE[node][2]
        tag = next_tag
    label = -1-tag

    # 5. Stable partition into children, writing original token ids.
    out_base = output + parent.to(tl.int64) * N
    for child in tl.static_range(CHILDREN):
        child_offset = CHILD_OFFSETS[child]
        member = valid & (label == child)
        rank = tl.cumsum(member.to(tl.int32), axis=0) - 1
        tl.store(out_base + child_offset + rank, original, mask=member)


def _default_warps(n: int, device: torch.device) -> int:
    if FUSED_NODE_WARPS is not None:
        return int(FUSED_NODE_WARPS)
    return 2 if torch.cuda.get_device_capability(device) == (12, 0) else 4


def _default_block_t(n: int) -> int:
    return 64


def _default_rows_per_step(n: int) -> int:
    return 1


def fused_node_split(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    row_batch: torch.Tensor,
    tokens: int,
    child_capacities: tuple[int, ...],
    *,
    mode: str = "tf32x3",
    num_warps: int | None = None,
    block_t: int | None = None,
    rows_per_step: int | None = None,
    fp8: bool = False,
    midpoint: bool = False,
    landmarks: int = 32,
    terminal: bool = False,
) -> torch.Tensor:
    """Split every node in ``global_indices`` and return original ids in child order.

    ``source`` is the contiguous ``[rows, 128]`` feature table, ``global_indices``
    the ``[parents, N]`` int64 rows of each node in current order,
    ``row_batch`` the ``[parents]`` batch row of each node and ``tokens`` the
    per-row token count, so original ids are ``global - row_batch * tokens``.
    Any 2..16 positive child capacities are supported. ``terminal`` is retained
    for caller compatibility; it no longer selects a separate algorithm.
    """
    expected_dtypes = (torch.uint8,) if fp8 else (torch.float16, torch.bfloat16)
    if not (source.is_cuda and source.ndim == 2 and source.shape[1] == 128
            and source.is_contiguous() and source.dtype in expected_dtypes):
        raise ValueError("fused node split requires a contiguous CUDA [rows,128] table "
                         "(FP16/BF16, or uint8-viewed FP8 E4M3 with fp8=True)")
    if global_indices.ndim != 2 or global_indices.dtype not in (torch.int32, torch.long):
        raise ValueError("global_indices must be int32/int64 [parents, tokens]")
    parents, n = global_indices.shape
    children = len(child_capacities)
    if not 2 <= children <= 16 or min(child_capacities) <= 0:
        raise ValueError("expected 2 to 16 positive child capacities")
    if sum(child_capacities) != n:
        raise ValueError("child capacities must sum to N")
    if landmarks not in (16, 32, 64, 128) or not landmarks <= n <= FUSED_NODE_MAX_TOKENS:
        raise ValueError("node size outside the fused range")
    if row_batch.shape != (parents,) or row_batch.dtype != torch.long:
        raise ValueError("row_batch must be int64 [parents]")
    if mode not in ("fp16", "fp16x3", "bf16", "tf32", "tf32x3"):
        raise ValueError("unsupported score precision")
    from .landmark_v2_terminal import split_topology
    split_tree = split_topology(tuple(child_capacities))
    child_offsets = tuple(sum(child_capacities[:i]) for i in range(children))
    index_bits = max(1,(int(tokens)-1).bit_length())
    if index_bits + 32 + (children-2).bit_length() > 63:
        raise ValueError("routing key exceeds int64 capacity")
    global_indices = global_indices.contiguous()
    row_batch = row_batch.contiguous()
    scratch = torch.empty((parents, n, children - 1), device=source.device, dtype=torch.float32)
    output = torch.empty((parents, n), device=source.device, dtype=global_indices.dtype)
    _fused_node_split_kernel[(parents,)](
        source, global_indices, row_batch, scratch, output, int(tokens),
        N=n, BLOCK_N=triton.next_power_of_2(n), MAX_LEN=triton.cdiv(n, landmarks),
        LANDMARKS=landmarks, CHILDREN=children, INTERNAL=children - 1,
        CHILD_OFFSETS=child_offsets,
        BLOCK_T=_default_block_t(n) if block_t is None else block_t,
        ROWS_PER_STEP=_default_rows_per_step(n) if rows_per_step is None else rows_per_step,
        MODE=mode, FP8=fp8, MIDPOINT=midpoint,
        SPLIT_TREE=split_tree, INDEX_BITS=index_bits,
        num_warps=(16 if landmarks >= 128 else 8 if landmarks == 64 else _default_warps(n, source.device)) if num_warps is None else num_warps,
    )
    return output
