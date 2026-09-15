"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


from collections import Counter


from .spark_defaults import SPARK_REWEIGHT_TARGET_BLOCKS, SPARK_REWEIGHT_MIN_BLOCKS, SPARK_REWEIGHT_MAX_BLOCKS


from functools import lru_cache


import math


from .reblock_hierarchy import build_reblock_hierarchy


def _frontiers_for_layout(video_tokens, children, grid_shape, minimum_frames, hierarchy=None):
    if hierarchy is None:
        hierarchy=build_reblock_hierarchy(video_tokens,children,grid_shape=grid_shape,
                                          minimum_frames=minimum_frames)
    if hierarchy.video_tokens != video_tokens:
        raise ValueError('hierarchy and video token count differ')
    return hierarchy.levels,hierarchy.metadata(),hierarchy.children


def _layout_at_level(video_tokens,total_tokens,levels,selected,children,metadata_extra=None):
    leaves=video_tokens//64;cut=leaves*64
    active=tuple((64*s,64*e) for s,e in levels[selected])
    # Excluded video tokens and all context retain the original physical leaf
    # boundaries. They must not be folded into a spatial ancestor by accident.
    tail=tuple((start,min(start+64,total_tokens)) for start in range(cut,total_tokens,64))
    ranges=active+tail
    mapping=[-1]*((total_tokens+63)//64)
    for parent,(start,end) in enumerate(ranges):
        for leaf in range(start//64,(end+63)//64):
            assert mapping[leaf]==-1;mapping[leaf]=parent
    assert all(p>=0 for p in mapping)
    metadata=dict(video_tokens=video_tokens,total_tokens=total_tokens,active_video_tokens=cut,
        video_leaves=leaves,children=list(children) if not isinstance(children,int) else [children],
        selected_completed_split_levels=selected,total_split_levels=len(levels)-1,
        active_virtual_blocks=len(active),total_virtual_blocks=len(ranges),
        active_size_counts=dict(sorted(Counter(e-s for s,e in active).items())),
        frontier_size_counts=[dict(sorted(Counter(64*(e-s) for s,e in level).items())) for level in levels],
        tail_policy='active-video-excluded tokens and packed context keep physical64 query blocks')
    if metadata_extra:
        metadata.update(metadata_extra)
    return dict(ranges=ranges,leaf_to_virtual=tuple(mapping),metadata=metadata)


@lru_cache(maxsize=128)
def virtual_query_layout(video_tokens,total_tokens,levels_up,children=(8,4),*,grid_shape=None,minimum_frames=0,hierarchy=None):
    if not (64<=video_tokens<=total_tokens) or type(levels_up) is not int or levels_up < 0:
        raise ValueError('requires 64<=video_tokens<=total_tokens and nonnegative integer levels_up')
    leaves=video_tokens//64;levels,hierarchy_metadata,children=_frontiers_for_layout(video_tokens,children,grid_shape,minimum_frames,hierarchy)
    selected=max(0,len(levels)-1-levels_up)
    return _layout_at_level(video_tokens,total_tokens,levels,selected,children,dict(
        **hierarchy_metadata,
        levels_up=levels_up,effective_levels_up=len(levels)-1-selected,selection_mode='levels_up',
        global_video_representative=selected==0,
        convention='global frontier before the last levels_up splitting rounds; already terminal64 leaves retained'))


@lru_cache(maxsize=128, typed=True)
def target_virtual_query_layout(video_tokens,total_tokens,target_blocks=SPARK_REWEIGHT_TARGET_BLOCKS,min_blocks=SPARK_REWEIGHT_MIN_BLOCKS,max_blocks=SPARK_REWEIGHT_MAX_BLOCKS,children=(8,4),*,grid_shape=None,minimum_frames=0,hierarchy=None):
    """Choose the global LMv2 frontier closest to a stable physical-block size."""
    if not 64<=video_tokens<=total_tokens:
        raise ValueError('requires 64<=video_tokens<=total_tokens')
    if any(type(x) is not int for x in (target_blocks,min_blocks,max_blocks)):
        raise ValueError('target/min/max blocks must be integers')
    if not 1<=min_blocks<=target_blocks<=max_blocks:
        raise ValueError('requires 1<=min_blocks<=target_blocks<=max_blocks')
    leaves=video_tokens//64;levels,hierarchy_metadata,children=_frontiers_for_layout(video_tokens,children,grid_shape,minimum_frames,hierarchy)
    candidates=[]
    for selected,frontier in enumerate(levels):
        sizes=tuple(end-start for start,end in frontier)
        # Weight by covered leaves, so a tiny early-terminal group cannot
        # dominate the selection for the rest of the video.
        cost=sum(size*math.log2(size/target_blocks)**2 for size in sizes)/leaves
        feasible=min(sizes)>=min_blocks and max(sizes)<=max_blocks
        candidates.append((not feasible,cost,abs(len(frontier)-leaves/target_blocks),selected))
    _,cost,_,selected=min(candidates)
    levels_up=len(levels)-1-selected
    return _layout_at_level(video_tokens,total_tokens,levels,selected,children,dict(
        **hierarchy_metadata,
        levels_up=levels_up,global_video_representative=selected==0,selection_mode='target_blocks',target_parent_blocks=target_blocks,
        min_parent_blocks=min_blocks,max_parent_blocks=max_blocks,selection_cost=cost,
        convention='global frontier minimizing token-weighted log-size error within the requested block range'))

