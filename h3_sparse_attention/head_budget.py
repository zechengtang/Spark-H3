"""Exact per-head Top-K budget routing for experimental allocation studies."""
from __future__ import annotations

import torch


def route_with_head_budgets(scores, budgets, candidate_blocks, video_tokens, sink_tokens):
    """Exact per-head counts, including deterministic handling of score ties."""
    if budgets.shape != (scores.shape[2],):
        raise ValueError("expected one budget per head")
    order = scores[..., :candidate_blocks].argsort(dim=-1, descending=True, stable=True)
    selected = torch.arange(candidate_blocks, device=scores.device)[None, None, None, :] < budgets[None, None, :, None]
    chosen = torch.zeros_like(order, dtype=torch.bool).scatter_(-1, order, selected.expand_as(order))
    route = torch.zeros_like(scores, dtype=torch.bool)
    route[..., :candidate_blocks] = chosen
    # Partial video blocks and all conditioning remain exact.
    if sink_tokens or video_tokens % 64:
        route[..., candidate_blocks:] = True
    return route.contiguous()
