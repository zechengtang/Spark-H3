"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


from .mahalanobis_kmeans import batched_cross_metric_factors


def landmark_direction_factors(query, key, indices, moment="raw", ridge=.001):
    if moment == "raw":
        return batched_cross_metric_factors(
            query, key, indices, ridge_epsilon=ridge, key_centered=False)
    if moment != "unit":
        raise ValueError(moment)
    # Normalize only the unchanged sampled tokens; no full-token copy is needed.
    q = query.index_select(-2, indices).float()
    k = key.index_select(-2, indices).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return batched_cross_metric_factors(
        q, k, torch.arange(indices.numel(), device=indices.device),
        ridge_epsilon=ridge, key_centered=False)

