"""Focused checks for balanced scheduling and arbitrary-count node splitting."""
import pytest
import torch
from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy
from h3_sparse_attention.landmark_v2_terminal import node_split_reference, route_scores, split_topology


def test_seventeen_blocks_use_two_balanced_children():
    h = build_reblock_hierarchy(17*64, (16,), fanout_mode='arbitrary_fanout')
    assert h.budgets(0,17) == (9,8)
    assert h.budgets(1,9) == (1,)*9
    assert h.budgets(1,8) == (1,)*8
    assert len(h.levels) == 3


@pytest.mark.parametrize('leaves,fanout,root', [(33,16,(11,11,11)), (17,16,(9,8)), (5,8,(1,)*5)])
def test_arbitrary_fanout_with_matching_final_fanout(leaves, fanout, root):
    h = build_reblock_hierarchy(leaves*64,(fanout,),final_fanout=fanout,fanout_mode='arbitrary_fanout')
    assert h.budgets(0,leaves) == root
    for round_ in h.split_budgets:
        for total, capacities in round_:
            assert sum(capacities) == total
            assert max(capacities)-min(capacities) <= 1
            assert 2 <= len(capacities) <= fanout


def test_ten_second_hierarchy_is_balanced_and_published():
    for fanout, expected_root in [(8,(567,567)), (16,(227,227,227,227,226))]:
        h = build_reblock_hierarchy(72576,(fanout,),final_fanout=16,fanout_mode='arbitrary_fanout')
        assert h.budgets(0,1134) == expected_root
        assert h.levels[-1] == tuple((i,i+1) for i in range(1134))
        for round_ in h.split_budgets:
            for total, capacities in round_:
                assert sum(capacities) == total
                assert max(capacities)-min(capacities) <= 1


def test_named_fanout_modes_control_the_complete_hierarchy():
    power = build_reblock_hierarchy(
        1134*64, (16,), fanout_mode='power_of_two_fanout')
    arbitrary = build_reblock_hierarchy(
        1134*64, (16,), fanout_mode='arbitrary_fanout')
    assert [len(level) for level in power.levels] == [1,16,256,1024,1134]
    assert [len(level) for level in arbitrary.levels] == [1,5,75,1134]
    assert power.budgets(0,1134) == (71,)*7+(70,)+(71,)*7+(70,)
    assert arbitrary.budgets(0,1134) == (227,)*4+(226,)
    assert power.final_fanout == 16
    assert power.metadata()['fanout_mode'] == 'power_of_two_fanout'
    assert arbitrary.metadata()['fanout_mode'] == 'arbitrary_fanout'
    assert arbitrary.final_fanout == 16


@pytest.mark.parametrize('bad', [None, '', 'power2', 'balanced', True])
def test_named_fanout_modes_are_strict(bad):
    with pytest.raises(ValueError, match='fanout_mode'):
        build_reblock_hierarchy(17*64,(16,),fanout_mode=bad)


@pytest.mark.parametrize('children', [3,5,17,32])
def test_general_route_exact_capacities_and_ties(children):
    caps = tuple(2+i%3 for i in range(children))
    n=sum(caps)
    ids=torch.randperm(n)[None]
    labels=route_scores(torch.zeros(1,n,children-1),ids,caps)
    offset=0
    for child,cap in enumerate(caps):
        assert torch.equal(labels == child, (ids >= offset)&(ids < offset+cap))
        offset+=cap
    assert len(split_topology(caps)) == children-1


def test_cpu_recursive_seventeen_block_split():
    from h3_sparse_attention.landmark_tree_v2 import recursive_landmark_tree_v2_reference
    n=17*64
    out=recursive_landmark_tree_v2_reference(torch.zeros(1,n,8),grid_shape=(1,1,n),max_children=16,fanout_mode='arbitrary_fanout')
    assert out.hierarchy.budgets(0,17) == (9,8)
    assert torch.equal(out.permutation,torch.arange(n)[None])


def test_recursive_modes_select_matching_scheduler_and_splitter():
    from h3_sparse_attention.landmark_tree_v2 import recursive_landmark_tree_v2_reference
    n=17*64; samples=torch.zeros(1,n,8)
    power=recursive_landmark_tree_v2_reference(
        samples,grid_shape=(1,1,n),max_children=16,
        fanout_mode='power_of_two_fanout')
    arbitrary=recursive_landmark_tree_v2_reference(
        samples,grid_shape=(1,1,n),max_children=16,
        fanout_mode='arbitrary_fanout')
    assert [item.children for item in power.split_stats] == [16,2]
    assert [item.children for item in arbitrary.split_stats] == [2,9,8]
    assert power.hierarchy.fanout_mode == 'power_of_two_fanout'
    assert arbitrary.hierarchy.fanout_mode == 'arbitrary_fanout'
    assert torch.equal(power.permutation,torch.arange(n)[None])
    assert torch.equal(arbitrary.permutation,torch.arange(n)[None])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_general_scoring_route_and_graph(monkeypatch):
    from h3_sparse_attention.landmark_v2_cosine_fast import build_cosine_directions, fused_cosine_scores
    from h3_sparse_attention.landmark_tree_clustering import _stable_counting_partition
    from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    import h3_sparse_attention.landmark_v2_fused_node as fused
    torch.manual_seed(27)
    caps=(96,80,64)
    samples=torch.randn(1,sum(caps),128,device='cuda',dtype=torch.bfloat16)
    centers=samples[:,:32].contiguous()
    weights=torch.tensor([[8]*16+[7]*16],device='cuda')
    ids=torch.randperm(sum(caps),device='cuda')[None]
    directions=build_cosine_directions(centers,weights,caps)
    scores=fused_cosine_scores(samples,directions,'tf32x3')
    labels=route_scores(scores,ids,caps)
    actual=_stable_counting_partition(labels,caps,validate=True,source_indices=ids)
    expected=node_split_reference(samples,ids,centers,weights,caps)
    assert torch.equal(actual,expected)
    # Exercise large-node dispatch and graph replay without compiling many
    # full-node shapes: full-node fusion is checked separately below.
    monkeypatch.setattr(fused,'FUSED_NODE_ENABLED',False)
    n=17*64
    source=torch.zeros(1,n,128,device='cuda',dtype=torch.bfloat16)
    plan=PreparedLandmarkTreeV2Permutation(batch=1,tokens=n,dim=128,grid_shape=(1,1,n),device='cuda',max_children=16,fanout_mode='arbitrary_fanout')
    plan.run(source)
    perm,inv=plan.run(source)
    assert plan.graph_active
    assert plan.hierarchy.budgets(0,17)==(9,8)
    assert torch.equal(perm,torch.arange(n,device='cuda')[None])
    assert torch.equal(perm.gather(1,inv),perm)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_existing_fused_split_accepts_unequal_three_way_capacities():
    from h3_sparse_attention.landmark_v2_fused_node import fused_node_split
    caps=(96,80,64);n=sum(caps)
    ids=torch.randperm(n,device='cuda')[None]
    source=torch.zeros(n,128,device='cuda',dtype=torch.bfloat16)
    out=fused_node_split(source,ids,torch.zeros(1,device='cuda',dtype=torch.long),n,caps,midpoint=True,mode='fp16')
    offset=0
    for cap in caps:
        assert torch.equal(out[0,offset:offset+cap],ids[0][(ids[0]>=offset)&(ids[0]<offset+cap)])
        offset+=cap
