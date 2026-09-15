"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


from functools import lru_cache


from typing import Literal


import torch


from .mahalanobis_kmeans import hilbert_distances_3d


InitialOrder = Literal[
    "flat", "tile_t4h4w4", "hilbert_thw", "hilbert_twh", "hilbert_htw",
    "hilbert_hwt", "hilbert_wth", "hilbert_wht",
]


INITIAL_ORDERS = (
    "flat", "tile_t4h4w4", "hilbert_thw", "hilbert_twh", "hilbert_htw",
    "hilbert_hwt", "hilbert_wth", "hilbert_wht",
)


def normalize_initial_order(initial_order: str) -> str:
    if initial_order not in INITIAL_ORDERS:
        raise ValueError(f"initial_order must be one of {', '.join(INITIAL_ORDERS)}")
    return initial_order


def _grid_tuple(grid_shape):
    if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 3:
        raise ValueError("grid_shape must contain three positive integers (T,H,W)")
    if any(type(value) is not int or value <= 0 for value in grid_shape):
        raise ValueError("grid_shape must contain three positive integers (T,H,W)")
    return tuple(grid_shape)


@lru_cache(maxsize=32)
def _cpu_order(grid_shape, initial_order):
    frames, height, width = grid_shape
    ids = torch.arange(frames * height * width, dtype=torch.long, device="cpu")
    if initial_order == "flat":
        return ids
    t, h, w = ids // (height * width), (ids // width) % height, ids % width
    if initial_order == "tile_t4h4w4":
        tile = ((t // 4) * ((height + 3) // 4) + h // 4) * ((width + 3) // 4) + w // 4
        within = (t % 4) * 16 + (h % 4) * 4 + w % 4
        return (tile * 64 + within).argsort(stable=True)
    axes = {"t": t, "h": h, "w": w}
    coordinates = torch.stack([axes[axis] for axis in initial_order[8:]], dim=-1)
    bits = max(1, (max(grid_shape) - 1).bit_length())
    return hilbert_distances_3d(coordinates, bits).argsort(stable=True)


@lru_cache(maxsize=32)
def _device_order(grid_shape, initial_order, device):
    return _cpu_order(grid_shape, initial_order).to(device=device)


def initial_token_indices(grid_shape, initial_order="flat", *, device):
    """Return cached, read-only indices into original samples; never move features.

    Hilbert suffixes reorder the coordinate axes supplied to the Hilbert encoder.
    Tile traversal is THW-major over tiles and within each 4x4x4 tile, clipping
    boundary tiles. Device-less CUDA requests are resolved before cache lookup.
    """
    grid = _grid_tuple(grid_shape)
    name = normalize_initial_order(initial_order)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _device_order(grid, name, device)

