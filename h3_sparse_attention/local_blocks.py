"""Shared normalization for kernel-local exact-block policies."""


def normalize_local_block_radius(value: bool | int) -> int:
    """Map the legacy bool policy to a symmetric radius.

    ``True`` retains the historical +/-1 behavior, ``False`` disables local
    retention, and a nonnegative integer selects an explicit radius.
    """

    if type(value) is bool:
        return 1 if value else -1
    if type(value) is int and value >= -1:
        return value
    raise ValueError("local block policy must be bool or an integer >= -1")
