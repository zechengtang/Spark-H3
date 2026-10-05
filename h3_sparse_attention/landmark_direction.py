"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


import torch


from .mahalanobis_kmeans import batched_cross_metric_factors


def landmark_single_direction_factor(
    query,
    key,
    indices,
    feature_side,
    moment="raw",
    ridge=.001,
    *,
    m2_estimator="hilbert_midpoint",
    full_chunk_size=4096,
):
    """Return the sole factor needed by one shared-layout source feature."""

    if feature_side not in ("query", "key"):
        raise ValueError("feature_side must be query or key")
    opposite = key if feature_side == "query" else query
    if moment == "raw":
        if m2_estimator == "hilbert_midpoint":
            from .mahalanobis_kmeans import batched_noncentered_metric_factor

            return batched_noncentered_metric_factor(
                opposite, indices, ridge_epsilon=ridge
            )
        from .mahalanobis_kmeans import (
            estimate_noncentered_second_moment,
            factor_noncentered_second_moment,
        )

        m2, _ = estimate_noncentered_second_moment(
            opposite,
            m2_estimator,
            sample_indices=indices,
            full_chunk_size=full_chunk_size,
        )
        return factor_noncentered_second_moment(m2, ridge_epsilon=ridge)
    if moment != "unit":
        raise ValueError(moment)
    if m2_estimator != "hilbert_midpoint":
        raise ValueError("unit moments currently require hilbert_midpoint")
    from .mahalanobis_kmeans import batched_noncentered_metric_factor

    selected = opposite.index_select(-2, indices).float()
    selected = selected / selected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    local_indices = torch.arange(indices.numel(), device=indices.device)
    return batched_noncentered_metric_factor(
        selected, local_indices, ridge_epsilon=ridge
    )


def landmark_direction_factors(
    query,
    key,
    indices,
    moment="raw",
    ridge=.001,
    *,
    m2_estimator="hilbert_midpoint",
    m2_side="both",
    return_diagnostics=False,
    full_chunk_size=4096,
):
    """Return Q/K M2 factors used on the opposite side's reblock features.

    ``m2_side='query'`` means only query clustering features receive the K M2
    transform; correspondingly the returned Q-moment factor (applied to K) is
    identity.  The default path is left unchanged for production compatibility.
    """
    if m2_side not in ("both", "query", "key", "none"):
        raise ValueError("m2_side must be both, query, key, or none")
    if m2_side == "none":
        dim = query.shape[-1]
        identity = torch.eye(dim, device=query.device, dtype=torch.float32).expand(
            *query.shape[:-2], dim, dim
        )
        factors = (identity, identity)
        if return_diagnostics:
            return (*factors, {
                "estimator": m2_estimator,
                "m2_side": m2_side,
                "query": {"constructed": False},
                "key": {"constructed": False},
            })
        return factors
    if moment == "raw" and m2_estimator == "hilbert_midpoint" and m2_side == "both":
        factors = batched_cross_metric_factors(
            query, key, indices, ridge_epsilon=ridge, key_centered=False)
        if return_diagnostics:
            return (*factors, {
                "estimator": m2_estimator,
                "m2_side": m2_side,
                "sample_count": int(indices.numel()),
                "sample_indices": indices,
            })
        return factors
    dim = query.shape[-1]
    identity = torch.eye(dim, device=query.device, dtype=torch.float32).expand(
        *query.shape[:-2], dim, dim
    )
    if moment == "raw":
        from .mahalanobis_kmeans import (
            estimate_noncentered_second_moment,
            factor_noncentered_second_moment,
        )

        query_m2 = key_m2 = None
        query_diagnostics = {"constructed": False}
        key_diagnostics = {"constructed": False}
        if m2_side in ("both", "key"):
            query_m2, query_diagnostics = estimate_noncentered_second_moment(
                query, m2_estimator, sample_indices=indices,
                full_chunk_size=full_chunk_size,
            )
            query_diagnostics["constructed"] = True
        if m2_side in ("both", "query"):
            key_m2, key_diagnostics = estimate_noncentered_second_moment(
                key, m2_estimator, sample_indices=indices,
                full_chunk_size=full_chunk_size,
            )
            key_diagnostics["constructed"] = True
        # query_factor is applied to K; key_factor is applied to Q.
        query_factor = (
            factor_noncentered_second_moment(query_m2, ridge_epsilon=ridge)
            if m2_side in ("both", "key") else identity
        )
        key_factor = (
            factor_noncentered_second_moment(key_m2, ridge_epsilon=ridge)
            if m2_side in ("both", "query") else identity
        )
        if return_diagnostics:
            return query_factor, key_factor, {
                "estimator": m2_estimator,
                "m2_side": m2_side,
                "query": query_diagnostics,
                "key": key_diagnostics,
            }
        return query_factor, key_factor
    if moment != "unit":
        raise ValueError(moment)
    if m2_estimator != "hilbert_midpoint":
        raise ValueError("unit moments currently require hilbert_midpoint")
    # Normalize only the unchanged sampled tokens; no full-token copy is needed.
    q = query.index_select(-2, indices).float()
    k = key.index_select(-2, indices).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if m2_side == "both":
        factors = batched_cross_metric_factors(
            q, k, torch.arange(indices.numel(), device=indices.device),
            ridge_epsilon=ridge, key_centered=False)
    else:
        from .mahalanobis_kmeans import (
            estimate_noncentered_second_moment,
            factor_noncentered_second_moment,
        )
        selected_indices = torch.arange(indices.numel(), device=indices.device)
        if m2_side == "key":
            q_m2, _ = estimate_noncentered_second_moment(
                q, "hilbert_midpoint", sample_indices=selected_indices)
            factors = (
                factor_noncentered_second_moment(q_m2, ridge_epsilon=ridge), identity
            )
        else:
            k_m2, _ = estimate_noncentered_second_moment(
                k, "hilbert_midpoint", sample_indices=selected_indices)
            factors = (
                identity, factor_noncentered_second_moment(k_m2, ridge_epsilon=ridge)
            )
    if return_diagnostics:
        return (*factors, {
            "estimator": m2_estimator, "m2_side": m2_side,
            "sample_count": int(indices.numel()), "sample_indices": indices,
        })
    return factors
