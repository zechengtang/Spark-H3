"""Immutable hierarchy owned by reblocking and consumed by downstream methods.

Leaf intervals refer to the final packed 64-token block order. This plan drives
actual split capacities, so consumers need not infer topology from settings.
"""
from dataclasses import dataclass
from functools import lru_cache

# None inherits the ordinary fanout. Environment variables do not override it.
# Prepared CUDA graphs retain the explicit settings used at construction.


def normalize_final_fanout(final_fanout):
    if type(final_fanout) is int and 2 <= final_fanout <= 32:
        return final_fanout
    if isinstance(final_fanout, (tuple, list)) and final_fanout:
        values = tuple(final_fanout)
        if all(type(v) is int and 2 <= v <= 32 for v in values):
            return values[0] if len(set(values)) == 1 else values
    raise ValueError('final_fanout must be an integer from 2 to 32 or a nonempty schedule')


def resolve_final_fanout(children, final_fanout=None, *,
                         fanout_mode='arbitrary_fanout', terminal_leaf_blocks=None):
    """None inherits fanout; an explicit legacy keyword remains readable."""
    children = (children,) if isinstance(children, int) else tuple(children)
    if terminal_leaf_blocks is not None:
        if terminal_leaf_blocks not in (0, 16):
            raise ValueError('legacy terminal_leaf_blocks must be 0 or 16')
        if final_fanout is not None:
            raise ValueError('use final_fanout or legacy terminal_leaf_blocks, not both')
    if fanout_mode == 'power_of_two_fanout':
        inherited = normalize_final_fanout(children)
        if final_fanout is not None and normalize_final_fanout(final_fanout) != inherited:
            raise ValueError('a separate final_fanout requires arbitrary_fanout or power_of_two_arbitrary_final mode')
        return inherited
    if final_fanout is not None:
        return normalize_final_fanout(final_fanout)
    if terminal_leaf_blocks is not None:
        return normalize_final_fanout(tuple(max(c, terminal_leaf_blocks) for c in children))
    return normalize_final_fanout(children)


def resolve_root_fanout(children, root_fanout=None):
    """Maximum fanout of the first nonfinal splitting round in each tree."""
    if root_fanout is None:
        root_fanout = children if isinstance(children, int) else children[0]
    if type(root_fanout) is not int or root_fanout not in (2, 4, 8, 16, 32):
        raise ValueError('root_fanout must be 2, 4, 8, 16 or 32, or None')
    return root_fanout


def _at_depth(schedule, depth):
    return schedule if isinstance(schedule, int) else schedule[min(depth, len(schedule)-1)]


def _fanout_alias(children, fanout):
    if fanout is None:
        return children
    if children != (16,) and children != fanout:
        raise ValueError('use fanout or legacy children, not conflicting values')
    return fanout

FANOUT_MODES = ('power_of_two_fanout', 'arbitrary_fanout', 'power_of_two_arbitrary_final')


def normalize_fanout_mode(mode):
    if mode not in FANOUT_MODES:
        raise ValueError(f'fanout_mode must be one of {FANOUT_MODES}')
    return mode

def balanced_child_budgets(leaves, children, depth=0, final_fanout=None, *, root_fanout=None, terminal_leaf_blocks=None):
    """Minimum outer depth, then minimum fanout fitting the remaining rounds.

    fanout limits nonfinal rounds; final_fanout limits the final round.
    The last schedule entry repeats. Capacities differ by at most one block
    and are in descending order.
    """
    final_fanout = resolve_final_fanout(children, final_fanout,
                                       terminal_leaf_blocks=terminal_leaf_blocks)
    root_fanout = resolve_root_fanout(children, root_fanout)
    if leaves == 1:
        return (1,)
    # A tree that finishes in one round uses the final-round setting.
    if leaves <= _at_depth(final_fanout, depth):
        return (1,)*leaves
    def limit(level):
        return root_fanout if level == 0 else children[min(level, len(children)-1)]
    rounds = 1
    while True:
        capacity = _at_depth(final_fanout, depth+rounds-1)
        for level in range(depth+rounds-2, depth-1, -1):
            capacity *= limit(level)
        if capacity >= leaves:
            break
        rounds += 1
    if rounds == 1:
        count = leaves
    else:
        child_capacity = _at_depth(final_fanout, depth+rounds-1)
        for level in range(depth+rounds-2, depth, -1):
            child_capacity *= limit(level)
        count = max(2, (leaves+child_capacity-1)//child_capacity)
    quotient, remainder = divmod(leaves, count)
    return (quotient+1,)*remainder + (quotient,)*(count-remainder)


def power_of_two_child_budgets(leaves, children, depth=0, *, root_fanout=None):
    """Maximum-first power-of-two budgets, also used for strict final rounds."""
    if leaves < 2:
        return (1,)
    limit = resolve_root_fanout(children, root_fanout) if depth == 0 else children[min(depth, len(children)-1)]
    count = 1
    while count * 2 <= min(leaves, limit):
        count *= 2
    frontier = (int(leaves),)
    while len(frontier) < count:
        frontier = tuple(value for budget in frontier
                         for value in ((budget + 1)//2, budget//2))
    return frontier


@lru_cache(maxsize=64)
def tree_frontiers(leaves, children=(16,), roots=None,
                   final_fanout=None, fanout_mode='power_of_two_fanout', *,
                   fanout=None, root_fanout=None, terminal_leaf_blocks=None):
    fanout_mode = normalize_fanout_mode(fanout_mode)
    children = _fanout_alias(children, fanout)
    if leaves<1:raise ValueError('at least one complete video leaf is required')
    if isinstance(children,int):children=(children,)
    if not children or any(c not in (2,4,8,16,32) for c in children):raise ValueError('invalid LMv2 child schedule')
    final_fanout = resolve_final_fanout(children, final_fanout, fanout_mode=fanout_mode,
                                       terminal_leaf_blocks=terminal_leaf_blocks)
    root_fanout = resolve_root_fanout(children, root_fanout)
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
            offset=start
            # Only the arbitrary modes finish non-power-of-two nodes directly.
            # Strict mode revisits unfinished children with fresh landmarks.
            if fanout_mode != 'power_of_two_fanout' and size <= _at_depth(final_fanout, depth):
                budgets = (1,) * size
            elif fanout_mode in ('power_of_two_fanout', 'power_of_two_arbitrary_final'):
                budgets = power_of_two_child_budgets(size, children, depth, root_fanout=root_fanout)
            else:
                budgets = balanced_child_budgets(size, children, depth, final_fanout, root_fanout=root_fanout)
            for count in budgets:
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
    final_fanout: int | tuple[int, ...] | None = None
    fanout_mode: str = 'power_of_two_fanout'
    root_fanout: int | None = None

    def __post_init__(self):
        object.__setattr__(self, 'final_fanout', resolve_final_fanout(
            self.children, self.final_fanout, fanout_mode=self.fanout_mode))
        object.__setattr__(self, 'root_fanout', resolve_root_fanout(self.children, self.root_fanout))

    @property
    def fanout(self):
        return self.children[0] if len(set(self.children)) == 1 else self.children

    def budgets(self, splitting_round, leaves):
        for size, budgets in self.split_budgets[splitting_round]:
            if size == leaves:
                return budgets
        raise ValueError('node is not part of the prepared reblocking hierarchy')

    def metadata(self):
        return dict(hierarchy_boundary='video root',
                    hierarchy_source='reblock_plan',
                    fanout=self.fanout,
                    root_fanout=self.root_fanout,
                    final_fanout=self.final_fanout,
                    fanout_mode=self.fanout_mode)


@lru_cache(maxsize=128)
def build_reblock_hierarchy(video_tokens, children=(16,), *, grid_shape=None,
                            final_fanout=None, fanout_mode='power_of_two_fanout',
                            fanout=None, root_fanout=None, terminal_leaf_blocks=None):
    fanout_mode = normalize_fanout_mode(fanout_mode)
    children = _fanout_alias(children, fanout)
    if isinstance(children,int):children=(children,)
    leaves=video_tokens//64
    final_fanout = resolve_final_fanout(children, final_fanout, fanout_mode=fanout_mode,
                                       terminal_leaf_blocks=terminal_leaf_blocks)
    root_fanout = resolve_root_fanout(children, root_fanout)
    roots=((0,leaves),)
    levels=tree_frontiers(leaves,children,final_fanout=final_fanout,fanout_mode=fanout_mode,root_fanout=root_fanout)
    # These capacities, not a second fanout calculation, drive token splitting.
    rounds=[]
    for parents,children_level in zip(levels,levels[1:]):
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
    return ReblockHierarchy(video_tokens,tuple(children),levels,roots,tuple(rounds),
                            final_fanout,fanout_mode,root_fanout)
