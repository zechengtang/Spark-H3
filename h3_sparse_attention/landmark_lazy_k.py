"""Landmark-v2 over resident attention K; transform rows only when consumed.

The only persistent feature tensor is original THD K. Transformed rows are staged in bounded tiles and discarded after each use.
No full transformed feature table is retained.
BF16 rounding after the transform is retained, matching a BF16 feature table.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .landmark_v2_cosine_fast import _fp16x3_dot


@triton.jit
def _project(K, F, IDS, HEADS, OUT, DIR, N:tl.constexpr, T:tl.constexpr,
             H:tl.constexpr, S:tl.constexpr, KIND:tl.constexpr, MODE:tl.constexpr,
             BM:tl.constexpr):
    b=tl.program_id(1)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    d=tl.arange(0,128)
    if KIND == 'norm':
        head=b
        token=rows
    else:
        head=tl.load(HEADS+b)
        token=tl.load(IDS+b*N+rows,rows<N,0)
    x=tl.load(K+(token[:,None].to(tl.int64)*H+head)*128+d[None,:],rows[:,None]<N,0)
    f=tl.load(F+head*128*128+d[:,None]*128+d[None,:])
    y=tl.dot(x,f).to(tl.bfloat16)
    if KIND == 'features':
        tl.store(OUT+(b*N+rows[:,None])*128+d[None,:],y,rows[:,None]<N)
    elif KIND == 'norm':
        z=y.to(tl.float32)
        norm=tl.sum(z*z,1)
        tl.store(OUT+b*N+rows,norm,rows<N)
    else:
        c=tl.arange(0,32 if S>16 else 16)
        direction=tl.load(DIR+(b*S+c[:,None])*128+d[None,:],c[:,None]<S,0).to(tl.float32)
        z=y.to(tl.float32)
        norm=tl.maximum(libdevice.sqrt_rn(tl.sum(z*z,1)),1.e-12)
        if MODE == 'fp16':
            score=tl.dot((z/norm[:,None]).to(tl.float16),tl.trans(direction.to(tl.float16)))
        elif MODE == 'fp16x3':
            score=_fp16x3_dot(z/norm[:,None],tl.trans(direction))
        else:
            score=tl.dot(z,tl.trans(direction),input_precision='tf32x3')/norm[:,None]
        tl.store(OUT+(b*N+rows[:,None])*S+c[None,:],score,(rows[:,None]<N)&(c[None,:]<S))


def projected_rows(k, factor, indices, heads):
    n=indices.shape[1]
    out=torch.empty((*indices.shape,128),device=k.device,dtype=torch.bfloat16)
    _project[(triton.cdiv(n,64),indices.shape[0])](
        k,factor,indices,heads,out,out,n,k.shape[0],k.shape[1],0,'features','fp16',64,
        num_warps=4)
    return out


def projected_scores(k,factor,indices,heads,directions,mode):
    n=indices.shape[1]; s=directions.shape[1]
    out=torch.empty((*indices.shape,s),device=k.device,dtype=torch.float32)
    _project[(triton.cdiv(n,64),indices.shape[0])](
        k,factor,indices,heads,out,directions,n,k.shape[0],k.shape[1],s,'scores',mode,64,
        num_warps=4)
    return out


@torch.no_grad()
def lazy_k_permutation(k, factor, video_tokens, grid, cfg, *, max_feature_rows=262144):
    from .landmark_tree_triton import stable_binary_mask_partition, squared_norm
    from .landmark_v2_cosine import FAST_PRECISION
    from .landmark_v2_cosine_fast import build_cosine_directions, fused_cosine_scores_indexed
    from .landmark_v2_fused_node import use_fused_node, fused_node_split
    from .landmark_v2_route import route_scores_cuda
    from .landmark_v2_terminal import partition_scores
    from .reblock_hierarchy import build_reblock_hierarchy
    if k.ndim!=3 or k.shape[-1]!=128 or not k.is_contiguous():
        raise ValueError('lazy K expects contiguous THD K')
    t=video_tokens; h=k.shape[1]; dev=k.device
    if not (cfg.landmark_tree_v2_layout_reuse=='q_from_k'
            and cfg.landmark_tree_v2_initial_order=='flat'
            and cfg.landmark_tree_v2_group_size==1
            and cfg.landmark_tree_v2_landmark_mode=='midpoint'
            and cfg.landmark_tree_v2_landmark_count==32
            and cfg.landmark_tree_v2_midpoint_direction_mode=='legacy'
            and cfg.landmark_tree_v2_distance=='cosine'
            and cfg.landmark_tree_v2_aggregation=='linear'
            and cfg.landmark_tree_v2_mean_mode=='raw'
            and cfg.landmark_tree_v2_order_mode=='parent_order'):
        raise ValueError('lazy K currently supports the default q_reuse_k tree')
    remainder=t%64
    norms=torch.empty((h,t),device=dev,dtype=torch.float32)
    norm_heads=torch.arange(h,device=dev,dtype=torch.long)
    norm_chunk=max(64,(max_feature_rows//h//64)*64)
    for first in range(0,t,norm_chunk):
        last=min(first+norm_chunk,t)
        indices=torch.arange(first,last,device=dev,dtype=torch.int32)[None].expand(h,-1).contiguous()
        feature=projected_rows(k,factor,indices,norm_heads)
        norms[:,first:last]=squared_norm(feature)
        del feature
    if remainder:
        threshold=norms.topk(remainder,dim=1,largest=True,sorted=False).values.amin(1,keepdim=True)
        greater=norms>threshold; equal=norms==threshold
        needed=remainder-greater.sum(1,keepdim=True)
        excluded_mask=greater|(equal&(equal.to(torch.int32).cumsum(1)<=needed))
    else:
        excluded_mask=torch.zeros((h,t),device=dev,dtype=torch.bool)
    order=torch.arange(t,device=dev,dtype=torch.long)
    partitioned=stable_binary_mask_partition(excluded_mask.contiguous(),order,left_count=t-remainder).to(torch.int32)
    del norms,excluded_mask
    leaves=t//64
    perm=torch.empty((h,t),device=dev,dtype=torch.int32)
    blocks=perm[:,:leaves*64].view(h,leaves,64)
    rows=torch.arange(h,device=dev,dtype=torch.long)
    hierarchy=build_reblock_hierarchy(t,(cfg.landmark_tree_v2_children,),grid_shape=tuple(grid),
        final_fanout=cfg.landmark_tree_v2_final_fanout,
        root_fanout=cfg.landmark_tree_v2_root_fanout,fanout_mode=cfg.landmark_tree_v2_fanout_mode)
    frontier=[(partitioned[:,:leaves*64],rows,torch.zeros_like(rows),leaves)]
    if leaves==1:
        blocks[:,0]=frontier[0][0]; frontier=[]
    level=0
    while frontier:
        grouped={}
        for frag in frontier:
            grouped.setdefault((frag[0].shape[1],frag[3]),[]).append(frag)
        next_frontier=[]
        for (n,budget),fragments in grouped.items():
            ids=torch.cat([f[0] for f in fragments]); heads=torch.cat([f[1] for f in fragments])
            starts=torch.cat([f[2] for f in fragments])
            budgets=hierarchy.budgets(level,budget); caps=tuple(64*b for b in budgets)
            mapped=torch.empty_like(ids)
            if use_fused_node(n,len(caps),128):
                batch_limit=max(1,max_feature_rows//n)
                for start in range(0,ids.shape[0],batch_limit):
                    part=ids[start:start+batch_limit].contiguous()
                    feature=projected_rows(k,factor,part,heads[start:start+batch_limit])
                    count=part.shape[0]
                    local=torch.arange(count*n,device=dev,dtype=torch.int32).view(count,n)
                    mapped[start:start+count]=fused_node_split(feature.reshape(-1,128),local,
                        torch.arange(count,device=dev),t,caps,mode=FAST_PRECISION,
                        midpoint=True,landmarks=32,original_ids=part)
                    del feature
            else:
                landmarks=min(32,n); c=torch.arange(landmarks,device=dev)
                begin=c*n//landmarks; end=(c+1)*n//landmarks
                mids=begin+(end-begin)//2
                centers=projected_rows(k,factor,ids[:,mids].contiguous(),heads)
                weights=(end-begin).to(torch.int32)[None].expand(ids.shape[0],-1).contiguous()
                directions=build_cosine_directions(centers,weights,caps)
                scores=torch.empty((*ids.shape,len(caps)-1),device=dev,dtype=torch.float32)
                # Preserve the reference reduction layout: transform -> BF16
                # tile -> unchanged score kernel. Fusing the reductions changes
                # a few FP16 near-tie scores on SM86.
                head_limit=max(1,max_feature_rows//min(n,4096))
                for first in range(0,ids.shape[0],head_limit):
                    last=min(first+head_limit,ids.shape[0]);count=last-first
                    token_chunk=max(64,(max_feature_rows//count//64)*64)
                    for a in range(0,n,token_chunk):
                        b=min(a+token_chunk,n)
                        feature=projected_rows(k,factor,ids[first:last,a:b].contiguous(),heads[first:last])
                        local=torch.arange(count*(b-a),device=dev,dtype=torch.int32).view(count,b-a)
                        scores[first:last,a:b]=fused_cosine_scores_indexed(feature.reshape(-1,128),local,
                            directions[first:last].contiguous(),FAST_PRECISION)
                        del feature
                arbitrary=(all(b==1 for b in budgets) and cfg.landmark_tree_v2_fanout_mode!='power_of_two_fanout') or cfg.landmark_tree_v2_fanout_mode=='arbitrary_fanout'
                if arbitrary:
                    mapped=partition_scores(scores,ids,caps,max_original_index=t-1,validate=False)
                else:
                    mapped=route_scores_cuda(scores,ids,caps,max_original_index=t-1,
                        partition=True,approximate_key32=True)
                del scores,centers,directions
            offset=leaf_offset=0
            for cap,b in zip(caps,budgets):
                child=mapped[:,offset:offset+cap]
                child_start=starts+leaf_offset
                if b==1: blocks[heads,child_start]=child
                else: next_frontier.append((child,heads,child_start,b))
                offset+=cap;leaf_offset+=b
        frontier=next_frontier;level+=1
    if remainder:perm[:,-remainder:]=partitioned[:,-remainder:]
    return perm.unsqueeze(0),hierarchy
