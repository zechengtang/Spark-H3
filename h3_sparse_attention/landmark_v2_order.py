"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


def scalar_tree_order(scores, original_indices, child_capacities):
    """Concatenate leaves after recursive score/index lexicographic sorts.

    Nodes use the same breadth-first binary indices as _exact_tree_route.
    Only scalar/index tensors move; raw token groups are gathered by caller.
    """
    batch, groups, nodes = scores.shape
    if nodes != len(child_capacities) - 1 or sum(child_capacities) != groups:
        raise ValueError("scalar tree score/capacity shape mismatch")
    root = torch.arange(groups, device=scores.device).expand(batch, -1)

    def split(indices, node, start, end):
        if end - start == 1:
            return indices
        # Sort secondary key first, then stable-sort primary key. This keeps
        # equal-score membership identical to the existing exact quantiles.
        ties = original_indices.gather(1, indices)
        indices = indices.gather(1, ties.argsort(dim=1, stable=True))
        values = scores[:, :, node].gather(1, indices)
        indices = indices.gather(1, values.argsort(dim=1, stable=True))
        middle = (start + end) // 2
        capacity = sum(child_capacities[start:middle])
        left = split(indices[:, :capacity], 2 * node + 1, start, middle)
        right = split(indices[:, capacity:], 2 * node + 2, middle, end)
        return torch.cat((left, right), dim=1)

    return split(root, 0, 0, len(child_capacities)).contiguous()

