"""Narrow hardware policies for device-specific memory workarounds."""
from __future__ import annotations

import torch


_A800_80GB_MIN_BYTES = 75 << 30
_A800_80GB_MAX_BYTES = 85 << 30


def is_a800_80gb(device) -> bool:
    """Return whether *device* is specifically an NVIDIA A800 80GB GPU.

    Compute capability alone is insufficient because A100 and multiple memory
    capacities share SM80.  Require the product/capacity tokens reported by
    the driver and verify the physical memory range as a second guard.
    """

    try:
        capability = tuple(torch.cuda.get_device_capability(device))
        properties = torch.cuda.get_device_properties(device)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    compact_name = "".join(
        character for character in str(properties.name).upper()
        if character.isalnum()
    )
    total_memory = int(properties.total_memory)
    return bool(
        capability == (8, 0)
        and "A800" in compact_name
        and "80GB" in compact_name
        and _A800_80GB_MIN_BYTES <= total_memory <= _A800_80GB_MAX_BYTES
    )


__all__ = ["is_a800_80gb"]
