"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


from functools import lru_cache


import torch


import triton


import triton.language as tl


@triton.jit
def _ordered_bits(score):
    bits = score.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)


@triton.jit
def _root_keys(S, I, K, N: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
               APPROX_KEY32: tl.constexpr, ID_BITS: tl.constexpr):
    row = tl.program_id(1)
    pos = tl.program_id(0) * B + tl.arange(0, B)
    valid = pos < N
    score = tl.load(S + (row * N + pos) * C, valid, 0)
    original = tl.load(I + row * N + pos, valid, 0)
    if APPROX_KEY32:
        # Scores are cosine margins in [-2, 2].  Use every high bit left by
        # the global token ID, then flip the sign bit so signed int32 order is
        # identical to the packed unsigned order.  The ID keeps keys unique,
        # hence every child still receives its exact requested capacity.
        levels: tl.constexpr = (1 << (32 - ID_BITS)) - 1
        quantized = ((score + 2.0) * (levels / 4.0)).to(tl.int32)
        quantized = tl.maximum(0, tl.minimum(levels, quantized))
        packed = (quantized.to(tl.uint32) << ID_BITS) | original.to(tl.uint32)
        key = (packed ^ 0x80000000).to(tl.int32, bitcast=True)
    else:
        key = (_ordered_bits(score).to(tl.int64) - 0x80000000) * 0x100000000 + original
    tl.store(K + row * N + pos, key, valid)


@triton.jit
def _compact_keys(S, I, T, M, OFFSETS, COUNTS, K, PACKED,
                  N: tl.constexpr, C: tl.constexpr, CHILDREN: tl.constexpr,
                  GROUPS: tl.constexpr, B: tl.constexpr,
                  APPROX_KEY32: tl.constexpr, ID_BITS: tl.constexpr):
    row = tl.program_id(1)
    pos = tl.program_id(0) * B + tl.arange(0, B)
    valid = pos < N
    tag = tl.load(T + row * N + pos, valid, -1)
    active = valid & (tag >= 0)
    score = tl.load(S + (row * N + pos) * C + tl.maximum(tag, 0), active, 0)
    original = tl.load(I + row * N + pos, active, 0)
    if APPROX_KEY32:
        levels: tl.constexpr = (1 << (32 - ID_BITS)) - 1
        quantized = ((score + 2.0) * (levels / 4.0)).to(tl.int32)
        quantized = tl.maximum(0, tl.minimum(levels, quantized))
        packed_key = (quantized.to(tl.uint32) << ID_BITS) | original.to(tl.uint32)
        key = (packed_key ^ 0x80000000).to(tl.int32, bitcast=True)
    else:
        key = (_ordered_bits(score).to(tl.int64) - 0x80000000) * 0x100000000 + original
    label = tl.load(M + tag + CHILDREN)
    tl.store(K + row * N + pos, key, active)
    # Internal compaction need not be stable: kth selection compares unique
    # (score, original-ID) keys. Atomic reservations avoid the three-pass
    # stable histogram/prefix/scatter; the final public partition stays stable.
    for group in tl.static_range(GROUPS):
        member = active & (label == group)
        count = tl.sum(member.to(tl.int32), 0)
        base = tl.atomic_add(COUNTS + row * GROUPS + group, count,
                             mask=count > 0, sem="relaxed")
        offset = tl.load(OFFSETS + group)
        rank = tl.cumsum(member.to(tl.int32), 0) - 1
        tl.store(PACKED + row * N + offset + base + rank, key, member)


@triton.jit
def _update(T, K, CUT, LEFT, RIGHT, OUT, N: tl.constexpr,
            ACTIVE: tl.constexpr, START: tl.constexpr, ROOT: tl.constexpr,
            FINAL: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(1)
    pos = tl.program_id(0) * B + tl.arange(0, B)
    valid = pos < N
    if ROOT:
        tag = tl.full((B,), 0, tl.int32)
    else:
        tag = tl.load(T + row * N + pos, valid, -1)
    active = valid & (tag >= 0)
    cut = tl.load(CUT + row * ACTIVE + tag - START, active, 0)
    key = tl.load(K + row * N + pos, active, 0)
    left = tl.load(LEFT + tag, active, -1)
    right = tl.load(RIGHT + tag, active, -1)
    tag = tl.where(active, tl.where(key <= cut, left, right), tag)
    if FINAL:
        tl.store(OUT + row * N + pos, -1 - tag, valid)
    else:
        tl.store(T + row * N + pos, tag, valid)


@triton.jit
def _small_route(S, I, CUT_POS, LEFT, RIGHT, OUT, N: tl.constexpr,
                 C: tl.constexpr, DEPTH: tl.constexpr, ID_BITS: tl.constexpr,
                 B: tl.constexpr, PARTITION: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, B)
    valid = pos < N
    original = tl.load(I + row * N + pos, valid, 0).to(tl.uint64)
    tags = tl.full((B,), 0, tl.int32)
    # Group, exact FP32 score bits, and the original ID fit in a single key.
    # One on-chip sort per depth replaces per-branch full-parent selection.
    for depth in range(DEPTH):
        active = valid & (tags >= 0)
        score = tl.load(S + (row * N + pos) * C + tl.maximum(tags, 0), active, 0)
        group = tl.where(tags >= 0, tags, C).to(tl.uint64)
        key = (group << (32 + ID_BITS)) | (_ordered_bits(score).to(tl.uint64) << ID_BITS) | original
        key = tl.where(valid, key, 0xffffffffffffffff)
        sorted_keys = tl.sort(key, descending=False)
        cut_pos = tl.load(CUT_POS + tags, active, 0)
        cutoff = tl.gather(sorted_keys, cut_pos, axis=0)
        left = tl.load(LEFT + tags, active, -1)
        right = tl.load(RIGHT + tags, active, -1)
        tags = tl.where(active, tl.where(key <= cutoff, left, right), tags)
    if PARTITION:
        # Sorting by (child, parent position) also performs the stable final
        # partition, without a separate histogram/prefix/scatter pipeline.
        order_key = tl.where(valid, (-1-tags).to(tl.uint32) * B + pos.to(tl.uint32), 0xffffffff)
        order = tl.sort(order_key, descending=False) % B
        ordered = tl.gather(original, order.to(tl.int32), axis=0)
        tl.store(OUT + row * N + pos, ordered, valid)
    else:
        tl.store(OUT + row * N + pos, -1 - tags, valid)


@lru_cache(maxsize=None)
def _tables(capacities, device):
    from .landmark_v2_terminal import split_topology
    topology = split_topology(capacities)
    children = len(capacities)
    left = torch.tensor([t[3] for t in topology], device=device, dtype=torch.int32)
    right = torch.tensor([t[4] for t in topology], device=device, dtype=torch.int32)
    cut_positions = [0] * len(topology)
    levels = []
    frontier = [0]
    for depth in range(topology[-1][5] + 1):
        mapping = [0] * (len(topology) + children)
        offsets = []
        active = []
        offset = 0
        for label, tag in enumerate(frontier):
            mapping[tag + children] = label
            offsets.append(offset)
            size = capacities[-1-tag] if tag < 0 else sum(topology[tag][1:3])
            if tag >= 0:
                active.append((tag, offset, size, topology[tag][1]))
            offset += size
        # The small kernel puts all completed branches after active branches.
        active_offset = 0
        for tag, _, size, target in active:
            cut_positions[tag] = active_offset + target - 1
            active_offset += size
        runs = []
        for tag, offset, size, target in active:
            if (runs and runs[-1][2:4] == (size, target)
                    and runs[-1][0] + runs[-1][4] == tag
                    and runs[-1][1] + runs[-1][4] * size == offset):
                start, off, size, target, count = runs[-1]
                runs[-1] = (start, off, size, target, count+1)
            else:
                runs.append((tag, offset, size, target, 1))
        levels.append((torch.tensor(mapping, device=device, dtype=torch.int32),
                       torch.tensor(offsets, device=device, dtype=torch.long),
                       tuple(runs), active[0][0], len(active)))
        frontier = [child for tag in frontier
                    for child in ((tag,) if tag < 0 else topology[tag][3:5])]
    return left, right, torch.tensor(cut_positions, device=device, dtype=torch.int32), tuple(levels)


def route_scores_cuda(scores, original, capacities, *, max_original_index=None,
                      partition=False, approximate_key32=False):
    """Route scores with exact capacities and optional quantized ordering.

    ``max_original_index`` is an optional upper bound on nonnegative IDs. It
    enables packed on-chip routing for small nodes without a GPU-to-CPU read.
    ``approximate_key32`` retains as many score bits as fit beside the unique
    token ID in an int32 key.  It may reorder near-tied scores but never changes
    child capacities.
    """
    capacities = tuple(capacities)
    batch, tokens = original.shape
    children = len(capacities)
    if children == 1:
        return original.clone() if partition else torch.zeros_like(original)
    if scores.device != original.device:
        raise ValueError('scores and original IDs must share a device')
    if scores.dtype != torch.float32 or original.dtype not in (torch.int32, torch.int64):
        raise ValueError('routing requires FP32 scores and int32/int64 original IDs')
    scores = scores.contiguous()
    original = original.contiguous()
    id_bits = max(1, int(max_original_index).bit_length()) if max_original_index is not None else 32
    if approximate_key32 and id_bits >= 31:
        raise ValueError("approximate int32 route requires at least two score bits")
    small = (tokens <= 1024 and max_original_index is not None
             and max_original_index >= 0
             and max_original_index.bit_length() + 32 + (children-1).bit_length() <= 64)
    if partition and not small:
        from .landmark_tree_clustering import _stable_counting_partition
        labels = route_scores_cuda(
            scores, original, capacities,
            max_original_index=max_original_index,
            approximate_key32=approximate_key32,
        )
        return _stable_counting_partition(labels, capacities, validate=False, source_indices=original)
    left, right, cut_positions, levels = _tables(capacities, scores.device)
    result = torch.empty_like(original)
    if small:
        _small_route[(batch,)](scores, original, cut_positions, left, right, result,
                              tokens, children-1, len(levels), id_bits,
                              triton.next_power_of_2(tokens), partition, num_warps=8 if tokens >= 512 else 4)
        return result
    tags = torch.empty((batch, tokens), dtype=torch.int32, device=scores.device)
    # The ComfyUI approximate path uses int32 score/ID keys; reproducibility-
    # sensitive callers retain the exact int64 key.
    key_dtype = torch.int32 if approximate_key32 else torch.long
    keys = torch.empty(original.shape, dtype=key_dtype, device=original.device)
    for depth, (mapping, offsets, runs, start, active) in enumerate(levels):
        if depth == 0:
            _root_keys[(triton.cdiv(tokens, 256), batch)](
                scores, original, keys, tokens, children-1, 256,
                approximate_key32, id_bits)
            packed = keys
        else:
            packed = torch.empty_like(keys)
            counts = torch.zeros((batch, offsets.numel()), device=scores.device, dtype=torch.int32)
            _compact_keys[(triton.cdiv(tokens, 256), batch)](
                scores, original, tags, mapping, offsets, counts, keys, packed,
                tokens, children-1, children, offsets.numel(), 256,
                approximate_key32, id_bits, num_warps=4)
        cuts = torch.empty((batch, active), device=scores.device, dtype=key_dtype)
        for node, offset, size, target, count in runs:
            segment = packed[:, offset:offset+count*size].view(batch, count, size)
            indices = torch.empty((batch, count), device=scores.device, dtype=torch.long)
            torch.kthvalue(segment, target, dim=2,
                           out=(cuts[:, node-start:node-start+count], indices))
        _update[(triton.cdiv(tokens, 256), batch)](
            tags, keys, cuts, left, right, result, tokens, active, start,
            depth == 0, depth == len(levels)-1, 256)
    return result
