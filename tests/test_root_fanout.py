"""Root/final defaults and independent first-round fanout control."""
import dataclasses
import pytest
import torch
from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy
from h3_sparse_attention.processor import H3SparseAttentionConfig


@pytest.mark.parametrize('fanout', [2, 4, 8, 16, 32, (8,4)])
@pytest.mark.parametrize('mode', ['arbitrary_fanout', 'power_of_two_fanout'])
def test_none_defaults_inherit_ordinary_fanout(fanout, mode):
    h = build_reblock_hierarchy(1134*64, fanout=fanout, fanout_mode=mode)
    expected_root = fanout if isinstance(fanout,int) else fanout[0]
    explicit = build_reblock_hierarchy(1134*64, fanout=fanout, root_fanout=expected_root,
                                       final_fanout=fanout, fanout_mode=mode)
    assert h == explicit
    assert h.metadata()['root_fanout'] == expected_root
    assert h.metadata()['final_fanout'] == fanout


@pytest.mark.parametrize('root', [2,4,8,16,32])
def test_root_limit_changes_only_the_first_nonfinal_round(root):
    h = build_reblock_hierarchy(513*64, fanout=8, root_fanout=root,
                                fanout_mode='arbitrary_fanout')
    for depth, round_ in enumerate(h.split_budgets):
        for size, budgets in round_:
            assert sum(budgets) == size
            assert len(budgets) <= (8 if all(n==1 for n in budgets) or depth else root)
    assert h.root_fanout == root and h.final_fanout == 8


def test_root_override_changes_depth_and_keeps_final_default_independent():
    a = build_reblock_hierarchy(65*64, fanout=8, root_fanout=2, fanout_mode='arbitrary_fanout')
    b = build_reblock_hierarchy(65*64, fanout=8, root_fanout=16, fanout_mode='arbitrary_fanout')
    assert a.budgets(0,65) == (33,32)
    assert b.budgets(0,65) == (8,8)+(7,)*7
    assert len(a.levels) == 4 and len(b.levels) == 3
    assert a.final_fanout == b.final_fanout == 8


def test_final_round_has_precedence_for_one_round_tree():
    h = build_reblock_hierarchy(14*64, fanout=8, root_fanout=2,
                                final_fanout=16, fanout_mode='arbitrary_fanout')
    assert h.budgets(0,14) == (1,)*14


@pytest.mark.parametrize('bad', [0,1,3,63,True,'8',()])
def test_invalid_root_is_rejected(bad):
    with pytest.raises(ValueError,match='root_fanout'):
        H3SparseAttentionConfig.sol(20,landmark_tree_v2_root_fanout=bad)


def test_configuration_defaults_and_roundtrip():
    cfg = H3SparseAttentionConfig.sol(20)
    assert cfg.landmark_tree_v2_root_fanout is None
    assert cfg.landmark_tree_v2_final_fanout is None
    cfg = dataclasses.replace(cfg,landmark_tree_v2_children=None,landmark_tree_v2_fanout=8,
        landmark_tree_v2_root_fanout=16,landmark_tree_v2_fanout_mode='arbitrary_fanout')
    assert H3SparseAttentionConfig(**dataclasses.asdict(cfg)) == cfg


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_processor_and_graph_forward_root_override():
    from types import SimpleNamespace
    import h3_sparse_attention.processor as proc
    import h3_sparse_attention.spark_integration as integration
    n=65*64
    x=torch.zeros(1,n,2,128,device='cuda',dtype=torch.bfloat16)
    layout=SimpleNamespace(grid=(1,1,n),video_tokens=n,sequence_length=n)
    cfg=H3SparseAttentionConfig.sol(20,sol_landmark_preprocess=True,
        sol_landmark_preprocess_version='v2',landmark_tree_v2_fanout=8,
        landmark_tree_v2_root_fanout=16,
        landmark_tree_v2_fanout_mode='arbitrary_fanout')
    ctrl=proc._Controller(cfg)
    for _ in range(3):
        plan,_,perm,inv=integration._landmark_tree_v2_combined_permutations(ctrl,x,x,layout)
    assert plan.graph_active and plan.root_fanout==16 and plan.final_fanout==8
    assert plan.hierarchy.budgets(0,65)==(8,8)+(7,)*7
    assert torch.equal(perm.gather(1,inv),torch.arange(n,device='cuda').expand_as(perm))
