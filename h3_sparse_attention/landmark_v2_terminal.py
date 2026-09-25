"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


from functools import lru_cache


import torch


@lru_cache(maxsize=256)
def split_topology(capacities):
    """BFS records: weight slot, left/right capacities, child tags, depth.

    Negative tags encode completed children as -1-child. Only unfinished
    branches get internal node IDs. No power-of-two child count is required.
    """
    if not 1 <= len(capacities) <= 64 or any(c <= 0 for c in capacities):
        raise ValueError('expected 1 to 64 positive child capacities')
    pending = [(0, len(capacities), 0, 0)] if len(capacities) > 1 else []
    records = []
    for start, end, slot, depth in pending:
        node = len(records)
        middle = start + (end-start+1)//2
        tags = []
        for a, b, child_slot in ((start, middle, 2*node+1), (middle, end, 2*node+2)):
            if b-a == 1:
                tags.append(-1-a)
            else:
                tags.append(len(pending))
                pending.append((a,b,child_slot,depth+1))
        records.append((slot,sum(capacities[start:middle]),sum(capacities[middle:end]),*tags,depth))
    return tuple(records)


def terminal_topology(leaves):
    """Compatibility helper for a general split into complete 64-token leaves."""
    if type(leaves) is not int or not 1 <= leaves <= 16:
        raise ValueError('terminal nodes require 1 to 16 leaves')
    return split_topology((64,)*leaves)


def node_split_reference(samples, original, centers, weights, capacities, *,
                         distance="cosine", aggregation="linear", order_mode="parent_order"):
    """Independent PyTorch oracle; stable (score, original-id) capacity routing."""
    topology = split_topology(tuple(capacities))
    if sum(capacities) != samples.shape[1]:
        raise ValueError("child capacities must sum to node size")
    if not topology:
        return original.clone()
    x = centers.float()
    unit = x / x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    token_unit = samples.float() / samples.float().norm(dim=-1,keepdim=True).clamp_min(1e-12)
    distances = (1-torch.bmm(unit,unit.transpose(1,2)) if distance == "cosine" else
                 x.square().sum(-1)[:,:,None]+x.square().sum(-1)[:,None,:]-2*torch.bmm(x,x.transpose(1,2)))
    upper = torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
    active_weights = [weights.long()]
    tags = torch.zeros_like(original)
    id_order = original.argsort(dim=1,stable=True)
    scalar_orders = {}
    for node,(slot,lc,rc,lt,rt,_) in enumerate(topology):
        weight = active_weights[slot]
        valid = weight > 0
        pairs = valid[:,:,None] & valid[:,None,:] & upper
        pair = distances.masked_fill(~pairs,-float('inf')).flatten(1).argmax(1)
        first,second = pair//x.shape[1],pair%x.shape[1]
        only = valid.int().argmax(1)
        first = torch.where(valid.sum(1)<2,only,first)
        second = torch.where(valid.sum(1)<2,only,second)
        batch = torch.arange(x.shape[0],device=x.device)
        left,right = x[batch,first],x[batch,second]
        for _ in range(2):
            direction = right/right.norm(dim=-1,keepdim=True).clamp_min(1e-12)-left/left.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            delta = (torch.bmm(unit,direction[:,:,None]).squeeze(-1) if distance == "cosine" else
                     torch.bmm(x,(2*(right-left))[:,:,None]).squeeze(-1)
                     + (left.square().sum(-1)-right.square().sum(-1))[:,None])
            order = delta.argsort(dim=1,stable=True)
            ordered = weight.gather(1,order)
            before = ordered.cumsum(1)-ordered
            take = torch.minimum((lc-before).clamp_min(0),ordered)
            lw = torch.zeros_like(weight).scatter(1,order,take)
            rw = weight-lw
            left = torch.bmm(lw.float()[:,None],x).squeeze(1)/lc
            right = torch.bmm(rw.float()[:,None],x).squeeze(1)/rc
        active_weights.extend((lw,rw))
        direction = right/right.norm(dim=-1,keepdim=True).clamp_min(1e-12)-left/left.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        if distance == "euclidean":
            score = torch.bmm(samples.float(),(2*(right-left))[:,:,None]).squeeze(-1)
            score += (left.square().sum(-1)-right.square().sum(-1))[:,None]
        elif aggregation == "max":
            similarity = torch.bmm(token_unit,unit.transpose(1,2))
            score = (similarity.masked_fill(rw[:,None,:]<=0,-float("inf")).amax(-1)
                     - similarity.masked_fill(lw[:,None,:]<=0,-float("inf")).amax(-1))
        else:
            score = torch.bmm(token_unit,direction[:,:,None]).squeeze(-1)
        member = tags == node
        ordered_score = score.masked_fill(~member,float('inf')).gather(1,id_order)
        order = id_order.gather(1,ordered_score.argsort(dim=1,stable=True))
        if order_mode == "scalar_order":
            if lt < 0:
                scalar_orders[-1-lt] = order[:,:lc]
            if rt < 0:
                scalar_orders[-1-rt] = order[:,lc:lc+rc]
        goes_left = torch.zeros_like(member).scatter(1,order[:,:lc],True)
        tags = torch.where(member,torch.where(goes_left,lt,rt),tags)
    if order_mode == "scalar_order":
        return original.gather(1,torch.cat([scalar_orders[i] for i in range(len(capacities))],dim=1))
    labels = -1-tags
    return original.gather(1,labels.argsort(dim=1,stable=True))


def terminal_split_reference(samples, original, centers, weights):
    """Compatibility wrapper; terminal nodes use the general splitting method."""
    return node_split_reference(samples, original, centers, weights, (64,)*(samples.shape[1]//64))


def route_scores(scores, original, capacities, *, max_original_index=None):
    """Exact capacities for arbitrary early-stop trees; labels are child IDs."""
    topology = split_topology(tuple(capacities))
    if sum(capacities) != original.shape[1]:
        raise ValueError("child capacities must sum to node size")
    if scores.shape != (*original.shape, len(topology)):
        raise ValueError("score/tree shape mismatch")
    if not topology:
        return torch.zeros_like(original)
    if scores.is_cuda:
        from .landmark_v2_route import route_scores_cuda
        return route_scores_cuda(scores, original, capacities, max_original_index=max_original_index)
    tags = torch.zeros_like(original)
    for depth in range(topology[-1][5]+1):
        active_scores = scores.gather(2,tags.clamp_min(0).unsqueeze(-1)).squeeze(-1).contiguous()
        bits = active_scores.view(torch.int32).long() & 0xffffffff
        ordered = torch.where(bits & 0x80000000 != 0, (~bits)&0xffffffff, bits^0x80000000)
        keys = (ordered-0x80000000)*0x100000000 + original
        next_tags = tags
        for node,(_,lc,_,left,right,node_depth) in enumerate(topology):
            if node_depth != depth:
                continue
            member = tags == node
            cutoff = keys.masked_fill(~member,torch.iinfo(torch.int64).max).kthvalue(lc,dim=1).values[:,None]
            next_tags = torch.where(member,torch.where(keys<=cutoff,left,right),next_tags)
        tags = next_tags
    return -1-tags


def partition_scores(scores, original, capacities, *, max_original_index=None, validate=False):
    """Exact routing followed by stable parent-order partition of original IDs."""
    split_topology(tuple(capacities))
    if sum(capacities) != original.shape[1]:
        raise ValueError("child capacities must sum to node size")
    if scores.is_cuda and not validate:
        if scores.shape != (*original.shape, len(capacities)-1):
            raise ValueError("score/tree shape mismatch")
        from .landmark_v2_route import route_scores_cuda
        return route_scores_cuda(scores, original, capacities,
                                 max_original_index=max_original_index, partition=True)
    from .landmark_tree_clustering import _stable_counting_partition
    labels = route_scores(scores, original, capacities, max_original_index=max_original_index)
    return _stable_counting_partition(labels, capacities, validate=validate, source_indices=original)
