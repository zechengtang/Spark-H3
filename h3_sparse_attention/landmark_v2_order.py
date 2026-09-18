"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


def scalar_tree_order(scores, original_indices, child_capacities):
    """Concatenate leaves after recursive score/index lexicographic sorts.

    Nodes use explicit breadth-first branch tags from split_topology.
    Only scalar/index tensors move; raw token groups are gathered by caller.
    """
    batch, groups, nodes = scores.shape
    if nodes != len(child_capacities) - 1 or sum(child_capacities) != groups:
        raise ValueError("scalar tree score/capacity shape mismatch")
    root = torch.arange(groups, device=scores.device).expand(batch, -1)

    from .landmark_v2_terminal import split_topology
    topology = split_topology(tuple(child_capacities))

    def split(indices, node):
        if node < 0:
            return indices
        # Sort secondary key first, then stable-sort primary key. This keeps
        # equal-score membership identical to the existing exact quantiles.
        ties = original_indices.gather(1, indices)
        indices = indices.gather(1, ties.argsort(dim=1, stable=True))
        values = scores[:, :, node].gather(1, indices)
        indices = indices.gather(1, values.argsort(dim=1, stable=True))
        _, capacity, _, left_tag, right_tag, _ = topology[node]
        left = split(indices[:, :capacity], left_tag)
        right = split(indices[:, capacity:], right_tag)
        return torch.cat((left, right), dim=1)

    return split(root, 0).contiguous()

