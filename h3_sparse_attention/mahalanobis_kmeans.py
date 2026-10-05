"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import os


import torch


def hilbert_distances_3d(coordinates: torch.Tensor, bits: int) -> torch.Tensor:
    """Return Skilling Hilbert distances for integer ``[N, 3]`` coordinates."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Hilbert coordinates must have shape [N, 3]")
    if bits <= 0 or torch.any(coordinates < 0) or torch.any(coordinates >= 2**bits):
        raise ValueError("Hilbert coordinates must fit the requested cube")
    values = coordinates.long().clone()
    q = 1 << (bits - 1)
    while q > 1:
        mask = q - 1
        for axis in range(3):
            high = (values[:, axis] & q) != 0
            values[high, 0] ^= mask
            low = ~high
            exchange = (values[low, 0] ^ values[low, axis]) & mask
            values[low, 0] ^= exchange
            values[low, axis] ^= exchange
        q >>= 1
    values[:, 1] ^= values[:, 0]
    values[:, 2] ^= values[:, 1]
    exchange = torch.zeros(values.shape[0], device=values.device, dtype=torch.long)
    q = 1 << (bits - 1)
    while q > 1:
        high = (values[:, 2] & q) != 0
        exchange[high] ^= q - 1
        q >>= 1
    values ^= exchange[:, None]

    distance = torch.zeros(values.shape[0], device=values.device, dtype=torch.long)
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            distance = (distance << 1) | ((values[:, axis] >> bit) & 1)
    return distance


def hilbert_order_3d(
    grid_shape: tuple[int, int, int],
    *,
    device: torch.device | str,
    frame_offset: int = 0,
) -> torch.Tensor:
    """Return raster token ids in 3-D Hilbert order for a temporal slab."""

    frames, height, width = (int(value) for value in grid_shape)
    if min(frames, height, width) <= 0:
        raise ValueError("grid dimensions must be positive")
    tokens = frames * height * width
    local = torch.arange(tokens, device=device)
    per_frame = height * width
    coordinates = torch.stack(
        (
            local // per_frame,
            (local // width) % height,
            local % width,
        ),
        dim=-1,
    )
    bits = max(1, (max(grid_shape) - 1).bit_length())
    distance = hilbert_distances_3d(coordinates, bits)
    return local.index_select(0, distance.argsort(stable=True)) + frame_offset * per_frame


def _uniform_midpoints(order: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or count > order.numel():
        raise ValueError("center count must lie in [1, number of samples]")
    # With count=N//block_size this selects the middle token of each uniform
    # Hilbert block.  The proportional form also handles short tail windows.
    positions = torch.floor(
        (torch.arange(count, device=order.device, dtype=torch.float64) + 0.5)
        * order.numel()
        / count
    ).long().clamp_max(order.numel() - 1)
    return order.index_select(0, positions)


def hilbert_midpoint_sample_indices(
    grid_shape: tuple[int, int, int],
    count: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample uniformly spaced token ids from the global 3-D Hilbert order."""

    order = hilbert_order_3d(grid_shape, device=device)
    return _uniform_midpoints(order, min(max(1, int(count)), order.numel()))


def metric_factor_method() -> str:
    """Factorization of the metric moments: ``cholesky`` (default) or ``eigh``.

    Read at call time so experiments can switch arms inside one process.
    """
    method = os.environ.get("H3_METRIC_FACTOR", "cholesky")
    if method not in ("cholesky", "eigh"):
        raise ValueError("H3_METRIC_FACTOR must be cholesky or eigh")
    return method


M2_ESTIMATORS = (
    "hilbert_midpoint",
    "flat64_block_mean",
    "flat64_midpoint_1",
    "flat64_midpoint_2",
    "flat64_midpoint_4",
    "flat64_mean_diag",
    "full",
)


def flat64_midpoint_sample_indices(tokens: int, samples_per_block: int, *, device):
    """Return deterministic real-token midpoint samples for every flat64 block.

    The returned weights make each block contribute ``n_b / N`` and each of
    its samples contribute equally within the block, including the short tail.
    """
    if type(tokens) is not int or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if samples_per_block not in (1, 2, 4):
        raise ValueError("samples_per_block must be 1, 2, or 4")
    indices: list[int] = []
    weights: list[float] = []
    for start in range(0, tokens, 64):
        length = min(64, tokens - start)
        count = min(samples_per_block, length)
        for j in range(count):
            indices.append(start + ((2 * j + 1) * length) // (2 * count))
            weights.append(length / (tokens * count))
    return (
        torch.tensor(indices, device=device, dtype=torch.long),
        torch.tensor(weights, device=device, dtype=torch.float32),
    )


def _weighted_outer(samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    flat = samples.float().reshape(-1, samples.shape[-2], samples.shape[-1])
    if weights.ndim == 1:
        weights = weights.expand(flat.shape[0], -1)
    else:
        weights = weights.reshape(-1, weights.shape[-1])
    if weights.shape != flat.shape[:2]:
        raise ValueError("moment weights do not match the sampled tokens")
    weighted = flat * weights.to(device=flat.device, dtype=torch.float32).unsqueeze(-1)
    moment = torch.bmm(flat.transpose(1, 2), weighted)
    return moment.reshape(*samples.shape[:-2], samples.shape[-1], samples.shape[-1])


def estimate_noncentered_second_moment(
    samples: torch.Tensor,
    estimator: str,
    *,
    sample_indices: torch.Tensor | None = None,
    full_chunk_size: int = 4096,
) -> tuple[torch.Tensor, dict]:
    """Estimate ``E[x x.T]`` over all input tokens without centering.

    Inputs are ``[..., N, D]`` and all reductions are FP32.  Flat64 estimators
    always use the original input order and include the final short block.
    """
    if estimator not in M2_ESTIMATORS:
        raise ValueError(f"unknown M2 estimator: {estimator}")
    if samples.ndim < 2 or not samples.is_floating_point():
        raise ValueError("samples must be floating point [..., N, D]")
    tokens, dim = samples.shape[-2:]
    if tokens < 1 or dim < 1:
        raise ValueError("samples must contain tokens and features")

    if estimator == "hilbert_midpoint":
        if sample_indices is None:
            raise ValueError("hilbert_midpoint requires sample_indices")
        indices = sample_indices.to(device=samples.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("sample_indices must be a nonempty in-range vector")
        in_range = (indices >= 0).all() & (indices < tokens).all()
        if indices.is_cuda:
            torch._assert_async(in_range, "sample_indices must be in range")
        elif not bool(in_range):
            raise ValueError("sample_indices must be a nonempty in-range vector")
        selected = samples.index_select(-2, indices).float()
        moment = torch.matmul(selected.transpose(-1, -2), selected) / indices.numel()
        return moment, {
            "estimator": estimator,
            "sample_count": int(indices.numel()),
            "sample_indices": indices,
            "input_cast_dtype": "float32",
            "matmul_dtype": "float32",
            "accumulation_dtype": "float32",
        }

    flat = samples.float().reshape(-1, tokens, dim)
    batch = flat.shape[0]
    if estimator.startswith("flat64_midpoint_"):
        per_block = int(estimator.rsplit("_", 1)[1])
        indices, weights = flat64_midpoint_sample_indices(
            tokens, per_block, device=samples.device
        )
        selected = samples.index_select(-2, indices)
        moment = _weighted_outer(selected, weights)
        return moment, {
            "estimator": estimator,
            "sample_count": int(indices.numel()),
            "sample_indices": indices,
            "samples_per_block": per_block,
            "input_cast_dtype": "float32",
            "matmul_dtype": "float32",
            "accumulation_dtype": "float32",
        }

    if estimator in ("flat64_block_mean", "flat64_mean_diag"):
        moment = torch.zeros((batch, dim, dim), device=samples.device, dtype=torch.float32)
        approximate_diag = torch.zeros((batch, dim), device=samples.device, dtype=torch.float32)
        for start in range(0, tokens, 64):
            block = flat[:, start : min(start + 64, tokens)]
            block_length = block.shape[1]
            mean = block.mean(dim=1)
            weight = block_length / tokens
            moment.add_(torch.bmm(mean.unsqueeze(2), mean.unsqueeze(1)), alpha=weight)
            approximate_diag.add_(mean.square(), alpha=weight)
        diagnostics = {
            "estimator": estimator,
            "input_token_count": tokens,
            "representative_count": (tokens + 63) // 64,
            "flat64_blocks": (tokens + 63) // 64,
            "input_cast_dtype": "float32",
            "reduction_dtype": "float32",
            "matmul_dtype": "float32",
            "accumulation_dtype": "float32",
        }
        if estimator == "flat64_mean_diag":
            exact_diag = flat.square().mean(dim=1)
            correction = exact_diag - approximate_diag
            scale = exact_diag.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
            tolerance = 2e-6 * scale
            valid = (correction >= -tolerance).all()
            if correction.is_cuda:
                torch._assert_async(
                    valid, "flat64 diagonal variance is significantly negative"
                )
                clamped_count = None
            elif not bool(valid):
                worst = float((correction / scale).amin())
                raise RuntimeError(
                    f"flat64 diagonal variance is significantly negative: {worst:.3e}"
                )
            else:
                clamped_count = int((correction < 0).sum())
            correction = correction.clamp_min_(0.0)
            moment.diagonal(dim1=-2, dim2=-1).add_(correction)
            diagnostics["negative_diagonal_clamped"] = clamped_count
            diagnostics["negative_diagonal_policy"] = (
                "clamp rounding negatives within 2e-6 of per-batch exact-diagonal scale; "
                "reject larger negatives"
            )
        return moment.reshape(*samples.shape[:-2], dim, dim), diagnostics

    if type(full_chunk_size) is not int or full_chunk_size < 1:
        raise ValueError("full_chunk_size must be a positive integer")
    moment = torch.zeros((batch, dim, dim), device=samples.device, dtype=torch.float32)
    for start in range(0, tokens, full_chunk_size):
        chunk = flat[:, start : min(start + full_chunk_size, tokens)]
        moment.add_(torch.bmm(chunk.transpose(1, 2), chunk))
    moment.div_(tokens)
    return moment.reshape(*samples.shape[:-2], dim, dim), {
        "estimator": estimator,
        "sample_count": tokens,
        "full_chunk_size": full_chunk_size,
        "input_cast_dtype": "float32",
        "matmul_dtype": "float32",
        "accumulation_dtype": "float32",
    }


def factor_noncentered_second_moment(
    moment: torch.Tensor, *, ridge_epsilon: float
) -> torch.Tensor:
    """Factor a non-centered second moment after trace-scaled diagonal ridge."""
    if moment.ndim < 2 or moment.shape[-1] != moment.shape[-2]:
        raise ValueError("moment must be [..., D, D]")
    if ridge_epsilon < 0:
        raise ValueError("ridge_epsilon must be non-negative")
    dim = moment.shape[-1]
    flat = moment.float().reshape(-1, dim, dim)
    symmetric = 0.5 * (flat + flat.transpose(1, 2))
    trace_scale = symmetric.diagonal(dim1=-2, dim2=-1).sum(-1) / dim
    symmetric.diagonal(dim1=-2, dim2=-1).add_(ridge_epsilon * trace_scale[:, None])
    if metric_factor_method() == "cholesky":
        factor, info = torch.linalg.cholesky_ex(symmetric, check_errors=False)
        valid = (info == 0).all()
        if info.is_cuda:
            torch._assert_async(valid, "ridged M2 Cholesky factorization failed")
        elif not bool(valid):
            raise RuntimeError("ridged M2 Cholesky factorization failed")
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        factor = eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(-2)
    return factor.reshape(*moment.shape)


def batched_cross_metric_factors(
    query_samples: torch.Tensor,
    key_samples: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    ridge_epsilon: float = 1e-3,
    key_ridge_from_uncentered: bool = False,
    key_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Q second moment and K covariance/moment factors in one EIGH.

    ``key_centered=False`` uses E[KK.T]; the default preserves existing callers.

    Query and Key are independent inputs with identical shapes.  Only their
    small sampled moment matrices are concatenated, avoiding a full-token Q/K
    copy while reducing two solver submissions/synchronizations to one.
    """

    if query_samples.shape != key_samples.shape or query_samples.ndim < 2:
        raise ValueError(
            "query_samples and key_samples must have equal [..., N, d] shapes"
        )
    if (
        not query_samples.is_floating_point()
        or not key_samples.is_floating_point()
    ):
        raise ValueError("query_samples and key_samples must be floating-point")
    if ridge_epsilon < 0:
        raise ValueError("ridge_epsilon must be non-negative")
    indices = sample_indices.to(device=query_samples.device, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("sample_indices must be a non-empty vector")
    if key_samples.device != query_samples.device:
        raise ValueError("query_samples and key_samples must share a device")

    query_selected = query_samples.index_select(-2, indices).float()
    key_selected = key_samples.index_select(-2, indices).float()
    batch_shape = query_selected.shape[:-2]
    count, dim = query_selected.shape[-2:]
    query_flat = query_selected.reshape(-1, count, dim)
    key_flat = key_selected.reshape(-1, count, dim)
    batch = query_flat.shape[0]

    key_uncentered_trace = (
        key_flat.square().sum(dim=(1, 2)) / float(count * dim)
        if key_ridge_from_uncentered
        else None
    )
    if key_centered:
        key_flat = key_flat - key_flat.mean(dim=1, keepdim=True)
    query_moment = torch.bmm(query_flat.transpose(1, 2), query_flat) / count
    key_covariance = torch.bmm(key_flat.transpose(1, 2), key_flat) / count
    query_moment = 0.5 * (query_moment + query_moment.transpose(1, 2))
    key_covariance = 0.5 * (key_covariance + key_covariance.transpose(1, 2))
    query_trace = query_moment.diagonal(dim1=-2, dim2=-1).sum(-1) / dim
    key_trace = key_covariance.diagonal(dim1=-2, dim2=-1).sum(-1) / dim
    if key_uncentered_trace is not None:
        key_trace = key_uncentered_trace
    query_moment.diagonal(dim1=-2, dim2=-1).add_(
        ridge_epsilon * query_trace[:, None]
    )
    key_covariance.diagonal(dim1=-2, dim2=-1).add_(
        ridge_epsilon * key_trace[:, None]
    )

    joint = torch.cat((query_moment, key_covariance), dim=0)
    if metric_factor_method() == "cholesky":
        # L @ L.T equals the ridged moment exactly like the eigen factor
        # V diag(sqrt(lambda)); the two differ by a rotation, so every inner
        # product of transformed features is identical.  Cholesky is about
        # 12x cheaper than the batched eigendecomposition and, with
        # check_errors=False, performs no host synchronization.  The trace
        # ridge keeps the matrices positive definite.
        factors, _ = torch.linalg.cholesky_ex(joint, check_errors=False)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(joint)
        factors = eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(-2)
    query_factor, key_factor = factors.split(batch, dim=0)
    shape = (*batch_shape, dim, dim)
    return query_factor.reshape(shape), key_factor.reshape(shape)


def batched_noncentered_metric_factor(
    samples: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    ridge_epsilon: float = 1e-3,
) -> torch.Tensor:
    """Build one sampled non-centered M2 factor without a dummy peer batch.

    This is the single-layout counterpart of ``batched_cross_metric_factors``.
    It deliberately uses the same Gram, symmetrization, ridge, and solver
    sequence so selecting one factor from the old joint call remains bitwise
    reproducible.
    """

    if samples.ndim < 2 or not samples.is_floating_point():
        raise ValueError("samples must be floating point [..., N, D]")
    if ridge_epsilon < 0:
        raise ValueError("ridge_epsilon must be non-negative")
    indices = sample_indices.to(device=samples.device, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("sample_indices must be a non-empty vector")
    selected = samples.index_select(-2, indices).float()
    batch_shape = selected.shape[:-2]
    count, dim = selected.shape[-2:]
    flat = selected.reshape(-1, count, dim)
    moment = torch.bmm(flat.transpose(1, 2), flat) / count
    moment = 0.5 * (moment + moment.transpose(1, 2))
    trace = moment.diagonal(dim1=-2, dim2=-1).sum(-1) / dim
    moment.diagonal(dim1=-2, dim2=-1).add_(ridge_epsilon * trace[:, None])
    if metric_factor_method() == "cholesky":
        factor, _ = torch.linalg.cholesky_ex(moment, check_errors=False)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(moment)
        factor = eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(-2)
    return factor.reshape(*batch_shape, dim, dim)
