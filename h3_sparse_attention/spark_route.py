"""Experimental routing from shared query-conditioned K summaries.

All modes use the same explicit Top-K policy: additional quota outside the
mandatory +/-1 band, followed by context blocks. No token-level QK tiles.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_scores(Q, MAP, AK, LM, OUT, T: tl.constexpr, H: tl.constexpr,
                     M: tl.constexpr, P: tl.constexpr, C: tl.constexpr,
                     MASS: tl.constexpr):
    qa, kb, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = (bh // H).to(tl.int64), bh % H
    rows = qa * 64 + tl.arange(0, 64)
    dd = tl.arange(0, 128)
    q = tl.load(Q + ((b*T+rows[:, None])*H+h)*128+dd[None, :], (rows<T)[:, None], 0)
    qm = (tl.sum(q.to(tl.float32), 0) / tl.minimum(64, T-qa*64)).to(Q.dtype.element_ty)
    parent = tl.load(MAP + qa).to(tl.int64)
    keys = kb * 32 + tl.arange(0, 32)
    base = ((b*P+parent)*H+h)*M+keys
    ak = tl.load(AK+base[:, None]*128+dd[None, :], (keys<C)[:, None], 0)
    # Match mean routing's BF16 dot-result rounding before scale; FP32
    # reduction here is not itself a claim of bit-identical GEMM arithmetic.
    dot = tl.sum(ak.to(tl.float32)*qm.to(tl.float32)[None, :], 1).to(Q.dtype.element_ty).to(tl.float32)
    score = dot * 0.08838834764831845
    if MASS:
        score += tl.load(LM+base, keys<C, 0)
    tl.store(OUT+((b*M+qa)*H+h)*C+keys, score*1.4426950408889634, keys<C)


def routing_scores(q, key_centroids, mapping, ak, lm, *, video_tokens, mode):
    b, t, h, d = q.shape
    m, c = triton.cdiv(t, 64), video_tokens // 64
    if mode not in ('mean', 'weighted_k', 'weighted_mass') or d != 128 or c < 1:
        raise ValueError('requires valid mode, D128 and complete video blocks')
    if mode == 'mean':
        from .sol_topk_cutoff import _gemm_score_map
        return _gemm_score_map(q, key_centroids, blocks=m, candidate_blocks=c)
    scores = torch.empty((b,m,h,c), device=q.device, dtype=torch.float32)
    _weighted_scores[(m,triton.cdiv(c,32),b*h)](q,mapping,ak,lm,scores,t,h,m,ak.shape[1],c,
        mode=='weighted_mass',num_warps=4)
    return scores


def route_from_scores(scores, *, video_tokens, sink_tokens, topk_ratio):
    from .spark_integration import _sol_topk_route_from_scores
    m, c = scores.shape[1], video_tokens // 64
    ids = torch.arange(m, device=scores.device)
    adjacent = (ids[:, None]-ids[None, :]).abs() <= 1
    quota = max(1, round(topk_ratio*c))
    route = _sol_topk_route_from_scores(scores, adjacent, candidate_blocks=c,
        target_topk=quota, video_tokens=video_tokens, sink_tokens=sink_tokens)
    stats = dict(block_size=64, blocks=m, candidate_video_blocks=c,
        query_video_blocks=math.ceil(video_tokens/64),
        sink_blocks=math.ceil((video_tokens+sink_tokens)/64)-video_tokens//64 if sink_tokens else 0,
        target_topk_blocks_per_query=quota, route_topk_ratio=topk_ratio,
        route_threshold_mode='spark_explicit_topk', score_rounding='BF16 dot, FP32 scale/mass')
    return route, stats


def reweighted_route(q, key_centroids, mapping, ak, lm, *, video_tokens, sink_tokens, topk_ratio, mode):
    scores = routing_scores(q,key_centroids,mapping,ak,lm,video_tokens=video_tokens,mode=mode)
    route, stats = route_from_scores(scores, video_tokens=video_tokens,
        sink_tokens=sink_tokens,topk_ratio=topk_ratio)
    stats['spark_route_score'] = mode
    return route, stats
