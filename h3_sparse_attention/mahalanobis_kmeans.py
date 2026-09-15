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

