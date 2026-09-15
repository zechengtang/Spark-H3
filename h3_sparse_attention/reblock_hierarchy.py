"""Spark dependencies adapted from MiniMax-H3-Sparse; see PORT_MANIFEST.json."""
from __future__ import annotations


from dataclasses import dataclass


from functools import lru_cache


import math


from .landmark_tree_clustering import _child_leaf_budgets, _choose_children


def temporal_leaf_roots(grid_shape, minimum_frames):
    """Temporal roots in 64-token leaf coordinates, matching reblock alignment."""
    if type(minimum_frames) is not int or minimum_frames < 0:
        raise ValueError('minimum_frames must be a nonnegative integer')
    if not minimum_frames:
        return None
    if grid_shape is None or len(grid_shape) != 3 or any(type(x) is not int or x <= 0 for x in grid_shape):
        raise ValueError('temporal roots require a positive (frames,height,width) grid')
    frames,height,width=grid_shape
    frame_tokens=height*width
    quantum=64//math.gcd(frame_tokens,64)
    unit=math.ceil(minimum_frames/quantum)*quantum
    if frames < unit or frames % quantum:
        raise ValueError('temporal roots require whole-frame, 64-token-aligned groups')
    count=frames//unit
    lengths=[unit]*(count-1)+[frames-unit*(count-1)]
    ends=[];start=0
    for length in lengths:
        end=start+length*frame_tokens//64
        ends.append((start,end));start=end
    return tuple(ends)


@lru_cache(maxsize=64)
def tree_frontiers(leaves, children=(8,4), roots=None):
    if leaves<1:raise ValueError('at least one complete video leaf is required')
    if isinstance(children,int):children=(children,)
    if not children or any(c not in (2,4,8,16,32) for c in children):raise ValueError('invalid LMv2 child schedule')
    frontier=((0,leaves),) if roots is None else tuple(roots)
    if (not frontier or frontier[0][0] != 0 or frontier[-1][1] != leaves
            or any(start >= end for start,end in frontier)
            or any(a[1] != b[0] for a,b in zip(frontier,frontier[1:]))):
        raise ValueError('roots must partition all video leaves in order')
    levels=[((0,leaves),),frontier] if roots is not None and len(frontier)>1 else [frontier]
    depth=0
    while any(end-start>1 for start,end in frontier):
        next_level=[]
        for start,end in frontier:
            size=end-start
            if size==1:
                next_level.append((start,end));continue
            fanout=_choose_children(size,children[min(depth,len(children)-1)])
            offset=start
            for count in _child_leaf_budgets(size,fanout):
                next_level.append((offset,offset+count));offset+=count
            assert offset==end
        frontier=tuple(next_level);levels.append(frontier);depth+=1
    return tuple(levels)


@dataclass(frozen=True)
class ReblockHierarchy:
    video_tokens: int
    children: tuple[int, ...]
    levels: tuple
    roots: tuple
    split_budgets: tuple
    minimum_frames: int = 0

    def budgets(self, splitting_round, leaves):
        for size, budgets in self.split_budgets[splitting_round]:
            if size == leaves:
                return budgets
        raise ValueError('node is not part of the prepared reblocking hierarchy')

    def metadata(self):
        return dict(temporal_minimum_frames=self.minimum_frames,
                    temporal_root_leaf_ranges=self.roots if self.minimum_frames else None,
                    hierarchy_boundary='video root',
                    temporal_split_counted=len(self.roots)>1,
                    hierarchy_source='reblock_plan')


@lru_cache(maxsize=128)
def build_reblock_hierarchy(video_tokens, children=(8,4), *, grid_shape=None, minimum_frames=0):
    if isinstance(children,int):children=(children,)
    roots=temporal_leaf_roots(grid_shape,minimum_frames)
    if roots is not None and math.prod(grid_shape) != video_tokens:
        raise ValueError('temporal grid must describe all video tokens')
    leaves=video_tokens//64
    levels=tree_frontiers(leaves,children,roots)
    roots=roots if roots is not None else ((0,leaves),)
    # These capacities, not a second fanout calculation, drive token splitting.
    rounds=[]
    offset=1 if len(roots)>1 else 0
    for parents,children_level in zip(levels[offset:],levels[offset+1:]):
        by_size={}
        child_index=0
        for start,end in parents:
            budgets=[]
            while child_index<len(children_level) and children_level[child_index][0]<end:
                a,b=children_level[child_index]
                assert start<=a<b<=end
                budgets.append(b-a);child_index+=1
            if end-start>1:
                previous=by_size.setdefault(end-start,tuple(budgets))
                assert previous==tuple(budgets)
        rounds.append(tuple(sorted(by_size.items())))
    return ReblockHierarchy(video_tokens,tuple(children),levels,roots,tuple(rounds),minimum_frames)

