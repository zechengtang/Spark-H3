"""Terminal capacity, early-stop topology, stable ties and fused CUDA checks."""
import pytest
import torch
from h3_sparse_attention.landmark_v2_terminal import terminal_topology, terminal_split_reference
from h3_sparse_attention.landmark_tree_v2 import recursive_landmark_tree_v2_reference
from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy


@pytest.mark.parametrize('leaves',range(1,17))
def test_small_node_finishes_in_one_outer_round(leaves):
    n=leaves*64
    result=recursive_landmark_tree_v2_reference(
        torch.zeros(1,n,4),grid_shape=(1,1,n),max_children=8,
        fanout_mode='arbitrary_fanout', final_fanout=16)
    assert len(result.split_stats)==int(leaves>1)
    if leaves>1:
        assert result.split_stats[0].children==leaves
        assert result.hierarchy.budgets(0,leaves)==(1,)*leaves
    assert torch.equal(result.permutation,torch.arange(n).reshape(1,n))
    assert result.block_indices.shape==(1,leaves,64)
    tree=terminal_topology(leaves)
    assert len(tree)==leaves-1
    assert all(l>0 and r>0 for _,l,r,_,_,_ in tree)
    assert sorted(-tag-1 for row in tree for tag in row[3:5] if tag<0)==list(range(leaves)) if leaves>1 else not tree


def test_unequal_branches_stop_without_empty_or_repeated_splits():
    tree=terminal_topology(3)
    assert tree[0][1:5]==(128,64,1,-3)
    assert tree[1][1:5]==(64,64,-1,-2)
    tree=terminal_topology(12)
    assert [sum(t[5]==depth for t in tree) for depth in range(4)]==[1,2,4,4]
    assert all(t[1]+t[2]>=128 for t in tree)


def test_temporal_hierarchy_publishes_finished_small_nodes():
    h=build_reblock_hierarchy(72576,(16,16),grid_shape=(72,28,36),minimum_frames=10,fanout_mode='arbitrary_fanout')
    assert h.budgets(1,16)==(1,)*16
    assert h.budgets(1,15)==(1,)*15
    assert len(h.levels)==4  # global root, six temporal roots, 15/16-block nodes, leaves
    from h3_sparse_attention.landmark_virtual_q import virtual_query_layout
    data=virtual_query_layout(72576,73565,1,hierarchy=h)
    assert data['metadata']['active_size_counts']=={960:18,1024:54}
    assert data['metadata']['final_fanout']==16
    assert data['metadata']['fanout_mode']=='arbitrary_fanout'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
# Production dispatch fuses only two- and four-child nodes.
@pytest.mark.parametrize('leaves',[2,4])
def test_fused_terminal_ties_and_stable_parent_order(leaves):
    from h3_sparse_attention.landmark_v2_fused_node import fused_node_split
    n=leaves*64;tokens=n+67
    # Scrambled parent order, distinct batch rows, and out-of-node token ids.
    torch.manual_seed(71+leaves)
    source=torch.zeros(2*tokens,128,device='cuda',dtype=torch.bfloat16)
    rows=torch.arange(2,device='cuda')
    original=torch.stack([torch.randperm(tokens,device='cuda')[:n] for _ in range(2)])
    ids=original+rows[:,None]*tokens
    out=fused_node_split(source,ids,rows,tokens,(64,)*leaves,mode='fp16',midpoint=True,terminal=True)
    assert torch.equal(out.sort(1).values,original.sort(1).values)
    ranks=original.argsort(1).argsort(1)
    expected=original.gather(1,(ranks//64).argsort(dim=1,stable=True))
    assert torch.equal(out,expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('leaves,midpoint',[(4,True)])
def test_fused_terminal_random_routing_matches_reference(leaves,midpoint):
    from h3_sparse_attention.landmark_v2_fused_node import fused_node_split
    from h3_sparse_attention.landmark_tree_v2_triton import indexed_interval_means
    n=leaves*64;tokens=n+67;parents=8
    torch.manual_seed(100+leaves)
    source=torch.randn(parents*tokens,128,device='cuda',dtype=torch.bfloat16)
    rows=torch.arange(parents,device='cuda')
    original=torch.stack([torch.randperm(tokens,device='cuda')[:n] for _ in range(parents)])
    ids=(original+rows[:,None]*tokens).contiguous()
    centers,weights=indexed_interval_means(source,ids,32,midpoint=midpoint)
    ref=terminal_split_reference(source[ids],original,centers,weights)
    actual=fused_node_split(source,ids,rows,tokens,(64,)*leaves,mode='fp16',midpoint=midpoint,terminal=True)
    assert torch.equal(actual.sort(1).values,original.sort(1).values)
    ref_leaf=torch.full((parents,tokens),-1,device='cuda',dtype=torch.long)
    labels=(torch.arange(n,device='cuda')//64).expand(parents,-1)
    ref_leaf.scatter_(1,ref,labels)
    agreement=(ref_leaf.gather(1,actual)==labels).float().mean().item()
    assert agreement>.99,agreement


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_terminal_fp16_normalizes_before_cast_and_replays_graph():
    from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    torch.manual_seed(17)
    # Include an excluded tail as well as twelve complete terminal leaves.
    n=12*64+17
    samples=torch.randn(3,n,128,device='cuda',dtype=torch.bfloat16)
    plan=PreparedLandmarkTreeV2Permutation(batch=3,tokens=n,dim=128,grid_shape=(1,1,n),device='cuda')
    plan.run(samples);perm,_=plan.run(samples)
    expected=perm.clone()
    assert plan.graph_active
    for scale in (2**20,2**-20):
        perm,inv=plan.run(samples*scale)
        assert torch.equal(perm,expected)
        assert torch.equal(perm.gather(1,inv),torch.arange(n,device='cuda').expand_as(perm))
