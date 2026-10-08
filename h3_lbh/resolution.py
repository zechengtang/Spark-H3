"""ComfyUI ResolutionSelector-compatible canvas planning for LBH."""

from __future__ import annotations

import math


def megapixel_canvas(megapixels: float, aspect_ratio: float, align: int = 32) -> tuple[int, int]:
    """Return ``(height, width)`` using Comfy's Mi-pixel convention and grid rounding."""
    if megapixels <= 0:
        raise ValueError("megapixels must be positive")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")
    if align < 1:
        raise ValueError("align must be positive")
    pixels = float(megapixels) * 1024 * 1024
    raw_height = math.sqrt(pixels / float(aspect_ratio))
    raw_width = raw_height * float(aspect_ratio)
    height = max(align, round(raw_height / align) * align)
    width = max(align, round(raw_width / align) * align)
    return height, width


def official_canvas_pair(
    *, low_megapixels: float = 0.2, target_megapixels: float = 1.0,
    aspect_ratio: float = 16 / 9, align: int = 32,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the independently aligned low/high canvases from LBH's example."""
    low = megapixel_canvas(low_megapixels, aspect_ratio, align)
    high = megapixel_canvas(target_megapixels, aspect_ratio, align)
    if low[0] > high[0] or low[1] > high[1]:
        raise ValueError("the low-resolution canvas must not exceed the target canvas")
    return low, high


__all__ = ["megapixel_canvas", "official_canvas_pair"]
