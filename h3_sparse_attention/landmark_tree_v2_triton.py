"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _headwise_permute_bthd_kernel(
        values,
        permutation,
        output,
        total_elements,
        tokens,
        heads,
        video_tokens,
        dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        output_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = output_offset < total_elements
        feature = output_offset % dim
        row = output_offset // dim
        head = row % heads
        token_batch = row // heads
        token = token_batch % tokens
        batch = token_batch // tokens
        is_video = token < video_tokens
        safe_token = tl.where(is_video, token, 0)
        source_video_token = tl.load(
            permutation + (batch * heads + head) * video_tokens + safe_token,
            mask=mask & is_video,
            other=0,
        )
        source_token = tl.where(is_video, source_video_token, token)
        source_offset = (
            ((batch * tokens + source_token) * heads + head) * dim + feature
        )
        value = tl.load(values + source_offset, mask=mask)
        tl.store(output + output_offset, value, mask=mask)

    @triton.jit
    def _headwise_permute_pair_bthd_kernel(
        first,
        second,
        permutation,
        first_output,
        second_output,
        total_elements,
        tokens,
        heads,
        video_tokens,
        dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        output_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = output_offset < total_elements
        feature = output_offset % dim
        row = output_offset // dim
        head = row % heads
        token_batch = row // heads
        token = token_batch % tokens
        batch = token_batch // tokens
        is_video = token < video_tokens
        safe_token = tl.where(is_video, token, 0)
        source_video_token = tl.load(
            permutation + (batch * heads + head) * video_tokens + safe_token,
            mask=mask & is_video,
            other=0,
        )
        source_token = tl.where(is_video, source_video_token, token)
        source_offset = (
            ((batch * tokens + source_token) * heads + head) * dim + feature
        )
        first_value = tl.load(first + source_offset, mask=mask)
        second_value = tl.load(second + source_offset, mask=mask)
        tl.store(first_output + output_offset, first_value, mask=mask)
        tl.store(second_output + output_offset, second_value, mask=mask)

    @triton.jit
    def _indexed_group_mean_kernel(
        source,
        indices,
        output,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        group_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        group = tl.program_id(0) * BLOCK_G + tl.arange(0, BLOCK_G)
        parent = tl.program_id(1)
        dims = tl.arange(0, BLOCK_D)[None, :]
        group = group[:, None]
        valid_group = group < tokens // group_size
        dim_mask = dims < dim
        total = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)
        group_start = group * group_size
        for offset in tl.static_range(group_size):
            source_row = tl.load(indices + parent * tokens + group_start + offset, valid_group, 0)
            value = tl.load(
                source + source_row * dim + dims,
                mask=dim_mask & valid_group,
                other=0.0,
            ).to(tl.float32)
            total += value
        tl.store(
            output + (parent * (tokens // group_size) + group) * dim + dims,
            total / group_size,
            mask=dim_mask & valid_group,
        )


    @triton.jit
    def _indexed_group_means_8_and_4_kernel(
        source,
        indices,
        coarse_output,
        fine_output,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Load an 8-token group once and emit its exact 8/4 means."""

        group = tl.program_id(0)
        parent = tl.program_id(1)
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < dim
        coarse_total = tl.zeros((BLOCK_D,), dtype=tl.float32)
        fine_left = tl.zeros((BLOCK_D,), dtype=tl.float32)
        fine_right = tl.zeros((BLOCK_D,), dtype=tl.float32)
        group_start = group * 8
        for offset in tl.static_range(8):
            source_row = tl.load(indices + parent * tokens + group_start + offset)
            value = tl.load(
                source + source_row * dim + dims,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            coarse_total += value
            if offset < 4:
                fine_left += value
            else:
                fine_right += value
        tl.store(
            coarse_output + (parent * (tokens // 8) + group) * dim + dims,
            coarse_total / 8,
            mask=dim_mask,
        )
        fine_base = (parent * (tokens // 4) + group * 2) * dim
        tl.store(fine_output + fine_base + dims, fine_left / 4, mask=dim_mask)
        tl.store(
            fine_output + fine_base + dim + dims,
            fine_right / 4,
            mask=dim_mask,
        )


    @triton.jit
    def _interval_mean_kernel(
        representatives,
        indices,
        centers,
        weights,
        groups: tl.constexpr,
        dim: tl.constexpr,
        landmarks: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INDIRECT: tl.constexpr,
        FP8: tl.constexpr,
        MIDPOINT: tl.constexpr = False,
    ):
        landmark = tl.program_id(0)
        parent = tl.program_id(1)
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < dim
        start = landmark * groups // landmarks
        end = (landmark + 1) * groups // landmarks
        if parent == 0:
            tl.store(weights + landmark, end - start)
        if MIDPOINT:
            start = start + (end - start) // 2
            end = start + 1
        total = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for group in tl.range(start, end, loop_unroll_factor=4):
            if INDIRECT:
                source_row = tl.load(indices + parent * groups + group).to(tl.int64)
            else:
                source_row = parent * groups + group
            if FP8:
                # FP8 E4M3 rows stored as uint8; decode after the load.
                value = tl.load(
                    representatives + source_row * dim + dims,
                    mask=dim_mask,
                    other=0,
                ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            else:
                value = tl.load(
                    representatives + source_row * dim + dims,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
            total += value
        tl.store(
            centers + (parent * landmarks + landmark) * dim + dims,
            total / (end - start),
            mask=dim_mask,
        )


def _available(tensor: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16)
        and 0 < tensor.shape[-1] <= 256
    )


def headwise_permute_bthd(
    values: torch.Tensor,
    permutation: torch.Tensor,
    *,
    video_tokens: int,
) -> torch.Tensor:
    """Gather per-head video rows directly into one contiguous BTHD output."""

    if triton is None or not values.is_cuda:
        raise ValueError("headwise_permute_bthd requires CUDA and Triton")
    if values.ndim != 4 or permutation.ndim != 3:
        raise ValueError("values/permutation must be BTHD/BHN")
    batch, tokens, heads, dim = values.shape
    if permutation.shape != (batch, heads, video_tokens):
        raise ValueError("permutation shape does not match BTHD values")
    if not values.is_contiguous() or not permutation.is_contiguous():
        raise ValueError("fused headwise permutation requires contiguous inputs")
    output = torch.empty_like(values)
    elements = values.numel()
    block = 1024
    _headwise_permute_bthd_kernel[(triton.cdiv(elements, block),)](
        values,
        permutation,
        output,
        elements,
        tokens,
        heads,
        video_tokens,
        dim=dim,
        BLOCK=block,
        num_warps=8,
    )
    return output


def headwise_permute_pair_bthd(
    first: torch.Tensor,
    second: torch.Tensor,
    permutation: torch.Tensor,
    *,
    video_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one per-head permutation to two contiguous BTHD tensors."""

    if triton is None or not first.is_cuda:
        raise ValueError("headwise_permute_pair_bthd requires CUDA and Triton")
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError("paired values must have matching shape and dtype")
    if first.ndim != 4 or permutation.ndim != 3:
        raise ValueError("values/permutation must be BTHD/BHN")
    batch, tokens, heads, dim = first.shape
    if permutation.shape != (batch, heads, video_tokens):
        raise ValueError("permutation shape does not match BTHD values")
    if not first.is_contiguous() or not second.is_contiguous() or not permutation.is_contiguous():
        raise ValueError("fused paired permutation requires contiguous inputs")
    first_output = torch.empty_like(first)
    second_output = torch.empty_like(second)
    elements = first.numel()
    block = 1024
    _headwise_permute_pair_bthd_kernel[(triton.cdiv(elements, block),)](
        first,
        second,
        permutation,
        first_output,
        second_output,
        elements,
        tokens,
        heads,
        video_tokens,
        dim=dim,
        BLOCK=block,
        num_warps=8,
    )
    return first_output, second_output


def indexed_group_mean(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Mean consecutive index groups without materializing gathered features."""

    if not _available(source):
        raise ValueError("indexed_group_mean requires CUDA FP16/BF16 D<=256")
    if source.ndim != 2 or global_indices.ndim != 2:
        raise ValueError("source and indices must have shapes [rows,D] and [P,N]")
    parents, tokens = global_indices.shape
    dim = source.shape[1]
    if group_size not in (1, 2, 4, 8) or tokens % group_size:
        raise ValueError("group_size must be 1, 2, 4, or 8 and divide the token count")
    indices = global_indices.contiguous()
    output = torch.empty(
        (parents, tokens // group_size, dim),
        device=source.device,
        dtype=source.dtype,
    )
    block_d = triton.next_power_of_2(dim)
    _indexed_group_mean_kernel[(triton.cdiv(tokens // group_size, 16), parents)](
        source,
        indices,
        output,
        tokens=tokens,
        dim=dim,
        group_size=group_size,
        BLOCK_D=block_d,
        BLOCK_G=16,
        num_warps=4,
    )
    return output


def indexed_group_means_8_and_4(
    source: torch.Tensor,
    global_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact group-8 and nested group-4 means with one source read."""

    if not _available(source):
        raise ValueError("nested group means require CUDA FP16/BF16 D<=256")
    if source.ndim != 2 or global_indices.ndim != 2:
        raise ValueError("source and indices must have shapes [rows,D] and [P,N]")
    parents, tokens = global_indices.shape
    dim = source.shape[1]
    if tokens % 8:
        raise ValueError("the token count must be divisible by 8")
    indices = global_indices.contiguous()
    coarse = torch.empty(
        (parents, tokens // 8, dim), device=source.device, dtype=source.dtype
    )
    fine = torch.empty(
        (parents, tokens // 4, dim), device=source.device, dtype=source.dtype
    )
    block_d = triton.next_power_of_2(dim)
    _indexed_group_means_8_and_4_kernel[(tokens // 8, parents)](
        source,
        indices,
        coarse,
        fine,
        tokens=tokens,
        dim=dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return coarse, fine


def contiguous_interval_means(
    representatives: torch.Tensor,
    landmarks: int,
    *, midpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact contiguous-interval means and integer interval weights."""

    if not _available(representatives):
        raise ValueError("interval means require CUDA FP16/BF16 D<=256")
    if representatives.ndim != 3 or not 16 <= landmarks <= 256:
        raise ValueError("representatives must be [P,G,D] with 16–256 landmarks")
    parents, groups, dim = representatives.shape
    if groups < landmarks:
        raise ValueError("the number of representatives must cover all landmarks")
    centers = torch.empty(
        (parents, landmarks, dim),
        device=representatives.device,
        dtype=representatives.dtype,
    )
    weights = torch.empty((landmarks,), device=representatives.device, dtype=torch.int32)
    block_d = triton.next_power_of_2(dim)
    _interval_mean_kernel[(landmarks, parents)](
        representatives.contiguous(),
        representatives,
        centers,
        weights,
        groups=groups,
        dim=dim,
        landmarks=landmarks,
        BLOCK_D=block_d,
        INDIRECT=False,
        FP8=False,
        MIDPOINT=midpoint,
        num_warps=4,
    )
    return centers, weights.expand(parents, -1)


def fp8_feature_table(source: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 copy of a BF16/FP16 feature table, viewed as uint8 for Triton."""
    if source.ndim != 2 or source.dtype not in (torch.float16, torch.bfloat16) or not source.is_cuda:
        raise ValueError("fp8 feature table requires CUDA FP16/BF16 [rows, D]")
    return source.to(torch.float8_e4m3fn).view(torch.uint8)


def _fp8_available(table: torch.Tensor) -> bool:
    return bool(
        triton is not None
        and table.is_cuda
        and table.dtype == torch.uint8
        and table.ndim == 2
        and table.is_contiguous()
        and 0 < table.shape[-1] <= 256
    )


def indexed_interval_means(
    source: torch.Tensor,
    global_indices: torch.Tensor,
    landmarks: int,
    *,
    fp8: bool = False,
    center_dtype: torch.dtype | None = None,
    midpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-one landmarks without materializing ordered token features.

    Accumulate rows in the same order as contiguous_interval_means: a different
    reduction tree can change BF16 centers and exact-capacity routing.  With
    ``fp8=True`` the source is a uint8 view of an FP8 E4M3 table (see
    ``fp8_feature_table``) and centers are written in ``center_dtype``.
    """
    if fp8:
        if not _fp8_available(source):
            raise ValueError("fp8 interval means require a contiguous CUDA uint8 [rows,D] table")
        if center_dtype is None:
            center_dtype = torch.bfloat16
    elif not _available(source) or source.ndim != 2 or not source.is_contiguous():
        raise ValueError("indexed interval means require contiguous CUDA FP16/BF16 [rows,D]")
    if global_indices.ndim != 2 or not 16 <= landmarks <= 256:
        raise ValueError("indices must be [parents,tokens] with 16–256 landmarks")
    parents, tokens = global_indices.shape
    if tokens < landmarks:
        raise ValueError("tokens must cover all landmarks")
    dim = source.shape[-1]
    centers = torch.empty((parents, landmarks, dim), device=source.device,
                          dtype=source.dtype if center_dtype is None else center_dtype)
    weights = torch.empty((landmarks,), device=source.device, dtype=torch.int32)
    _interval_mean_kernel[(landmarks, parents)](
        source, global_indices.contiguous(), centers, weights,
        groups=tokens, dim=dim, landmarks=landmarks,
        BLOCK_D=triton.next_power_of_2(dim), INDIRECT=True, FP8=fp8, MIDPOINT=midpoint, num_warps=4,
    )
    return centers, weights.expand(parents, -1)
