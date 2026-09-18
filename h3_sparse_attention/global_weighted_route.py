"""Global opposite-side anchors, block-local softmax centroids for routing only."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _centroids(X, A, OUT, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               M: tl.constexpr, BD: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = (bh // H).to(tl.int64), bh % H
    rows = block * 64 + tl.arange(0, 64)
    d = tl.arange(0, BD)
    x = tl.load(X + ((b*T+rows[:, None])*H+h)*D+d[None, :],
                (rows<T)[:, None] & (d<D)[None, :], 0).to(tl.float32)
    a = tl.load(A + (b*H+h)*D+d, d<D, 0)
    logits = tl.sum(x*a[None, :], 1) * (D ** -0.5)
    logits = tl.where(rows<T, logits, -float('inf'))
    w = tl.exp(logits-tl.max(logits, 0))
    mean = tl.sum(x*w[:, None], 0)/tl.sum(w, 0)
    tl.store(OUT + ((b*M+block)*H+h)*D+d, mean, d<D)


def weighted_centroids(x, anchor):
    """Return BF16 block means tilted by an FP32 global opposite-side mean."""
    b, t, h, d = x.shape
    if not x.is_cuda or not x.is_contiguous() or anchor.shape != (b, h, d):
        raise ValueError('requires contiguous CUDA BTHD and BHD anchor')
    out = torch.empty((b, triton.cdiv(t, 64), h, d), device=x.device, dtype=x.dtype)
    _centroids[(out.shape[1], b*h)](x, anchor.contiguous(), out, t, h, d,
                                    out.shape[1], triton.next_power_of_2(d))
    return out


def global_weighted_route(q, k, *, video_tokens, sink_tokens, topk_ratio,
                          weighted_side='both', key_centroids=None):
    """Use video-only global Q/K; context blocks stay exact outside Top-K."""
    from .spark_integration import _sol_topk_route_from_scores
    if weighted_side not in ('both', 'query', 'key'):
        raise ValueError('weighted_side must be both, query, or key')
    m, c = triton.cdiv(q.shape[1], 64), video_tokens // 64
    if weighted_side in ('both', 'key'):
        q_global = q[:, :video_tokens].mean(1, dtype=torch.float32)
        km = weighted_centroids(k, q_global)
    elif key_centroids is not None:
        km = key_centroids
    else:
        # Standalone callers may omit the already-computed official K means.
        km = torch.stack([block.mean(1, dtype=torch.float32)
                          for block in k.split(64, dim=1)], dim=1).to(k.dtype)
    if weighted_side in ('both', 'query'):
        k_global = k[:, :video_tokens].mean(1, dtype=torch.float32)
        qm = weighted_centroids(q, k_global)
        scores = torch.einsum('bqhd,bkhd->bqhk', qm, km[:, :c]).float()
        scores.mul_(q.shape[-1]**-0.5 * math.log2(math.e))
    else:
        from .sol_topk_cutoff import _gemm_score_map
        scores = _gemm_score_map(q, km, blocks=m, candidate_blocks=c)
    quota = max(1, round(topk_ratio*c))
    adjacent = torch.zeros((m, m), device=q.device, dtype=torch.bool)
    route = _sol_topk_route_from_scores(scores, adjacent, candidate_blocks=c,
        target_topk=quota, video_tokens=video_tokens, sink_tokens=sink_tokens)
    return route, dict(block_size=64, blocks=m, candidate_video_blocks=c,
        query_video_blocks=math.ceil(video_tokens/64),
        sink_blocks=math.ceil((video_tokens+sink_tokens)/64)-c if sink_tokens else 0,
        target_topk_blocks_per_query=quota, route_topk_ratio=topk_ratio,
        route_threshold_mode='global_qk_weighted_mean_explicit_topk',
        weighted_side=weighted_side,
        global_anchor_scope='video_tokens', weight_normalization='softmax_within_64_token_block')
