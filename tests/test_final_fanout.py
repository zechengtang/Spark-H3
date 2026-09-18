"""Explicit final fanout, old-setting translation, and public API propagation."""
import dataclasses
import pytest
import torch
from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy
from h3_sparse_attention.processor import H3SparseAttentionConfig


@pytest.mark.parametrize('fanout', [2, 4, 8, 16, 32, (8, 4), (32, 8)])
@pytest.mark.parametrize('leaves', [1, 7, 14, 17, 33, 1134])
def test_legacy_final_settings_keep_the_same_hierarchy(fanout, leaves):
    schedule = (fanout,) if isinstance(fanout, int) else fanout
    for old_limit in (0, 16):
        final = tuple(max(c, old_limit) for c in schedule)
        old = build_reblock_hierarchy(leaves*64, schedule,
            terminal_leaf_blocks=old_limit, fanout_mode='arbitrary_fanout')
        new = build_reblock_hierarchy(leaves*64, fanout=schedule,
            final_fanout=final, fanout_mode='arbitrary_fanout')
        assert old.levels == new.levels
        assert old.split_budgets == new.split_budgets
        assert old.metadata() == new.metadata()


@pytest.mark.parametrize('final', [4, 8, 16])
def test_explicit_final_limit_is_independent_of_outer_fanout(final):
    h = build_reblock_hierarchy(1134*64, fanout=8, final_fanout=final,
                                fanout_mode='arbitrary_fanout')
    for round_ in h.split_budgets:
        for _, children in round_:
            assert len(children) <= (final if all(n == 1 for n in children) else 8)
    assert h.metadata()['fanout'] == 8
    assert h.metadata()['final_fanout'] == final
    assert 'terminal_leaf_blocks' not in h.metadata()


@pytest.mark.parametrize('bad', [0, 1, 33, True, (), '16'])
def test_invalid_final_fanout_is_rejected(bad):
    with pytest.raises(ValueError, match='final_fanout'):
        build_reblock_hierarchy(1024, fanout=8, final_fanout=bad,
                                fanout_mode='arbitrary_fanout')


def test_tensor_api_final_fanout_changes_outer_rounds():
    from h3_sparse_attention.spark import spark_reblock
    x = torch.zeros(1, 14*64, 8)
    a = spark_reblock(x, grid_shape=(1, 1, 14*64), fanout=8,
                     final_fanout=16, fanout_mode='arbitrary_fanout')
    b = spark_reblock(x, grid_shape=(1, 1, 14*64), fanout=8,
                     final_fanout=8, fanout_mode='arbitrary_fanout')
    assert a.hierarchy.budgets(0, 14) == (1,)*14
    assert b.hierarchy.budgets(0, 14) == (7, 7)
    assert torch.equal(a.permutation, b.permutation)  # Stable ties.


def test_config_names_roundtrip_and_spark_preset_override():
    cfg = H3SparseAttentionConfig.spark(20, landmark_tree_v2_fanout=8,
        landmark_tree_v2_final_fanout=16, landmark_tree_v2_fanout_mode='arbitrary_fanout')
    assert cfg.landmark_tree_v2_children == cfg.landmark_tree_v2_fanout == 8
    assert cfg.landmark_tree_v2_final_fanout == 16
    assert H3SparseAttentionConfig(**dataclasses.asdict(cfg)) == cfg


def test_conflicting_names_are_rejected():
    with pytest.raises(ValueError, match='conflicting'):
        H3SparseAttentionConfig.sol(20, landmark_tree_v2_children=16,
                                    landmark_tree_v2_fanout=4)
    with pytest.raises(ValueError, match='not both'):
        build_reblock_hierarchy(1024, final_fanout=16, terminal_leaf_blocks=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('final', [8, 16])
def test_prepared_graph_preserves_explicit_final_fanout(final):
    from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    x = torch.zeros(1, 14*64, 128, device='cuda', dtype=torch.bfloat16)
    plan = PreparedLandmarkTreeV2Permutation(batch=1, tokens=14*64, dim=128,
        grid_shape=(1, 1, 14*64), device='cuda', fanout=8, final_fanout=final,
        fanout_mode='arbitrary_fanout')
    plan.run(x)
    perm, inv = plan.run(x)
    assert plan.graph_active
    assert plan.hierarchy.final_fanout == final
    assert plan.hierarchy.budgets(0, 14) == ((1,)*14 if final == 16 else (7, 7))
    assert torch.equal(perm.gather(1, inv), torch.arange(14*64, device='cuda')[None])


def test_none_inherits_fanout_even_with_legacy_environment(monkeypatch):
    import h3_sparse_attention.reblock_hierarchy as hierarchy
    monkeypatch.setenv('H3_LMV2_FINAL_FANOUT', '16')
    monkeypatch.setenv('H3_LMV2_TERMINAL_LEAVES', '16')
    assert hierarchy.resolve_final_fanout(8) == 8
    assert hierarchy.resolve_final_fanout(8, 16) == 16
    assert hierarchy.resolve_final_fanout(8, terminal_leaf_blocks=16) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_processor_passes_final_fanout_to_its_cached_plan():
    from types import SimpleNamespace
    import h3_sparse_attention.processor as proc
    import h3_sparse_attention.spark_integration as integration
    x = torch.zeros(1, 896, 2, 128, device='cuda', dtype=torch.bfloat16)
    layout = SimpleNamespace(grid=(1, 1, 896), video_tokens=896, sequence_length=896)
    for final in (8, 16):
        cfg = H3SparseAttentionConfig.sol(20, sol_landmark_preprocess=True,
            sol_landmark_preprocess_version='v2', landmark_tree_v2_fanout=8,
            landmark_tree_v2_final_fanout=final, landmark_tree_v2_fanout_mode='arbitrary_fanout')
        controller = proc._Controller(cfg)
        for _ in range(3):
            plan, _, perm, inv = integration._landmark_tree_v2_combined_permutations(
                controller, x, x, layout)
        assert plan.graph_active and plan.final_fanout == final
        assert plan.hierarchy.metadata()['final_fanout'] == final
        assert torch.equal(perm.gather(1, inv), torch.arange(896, device='cuda').expand_as(perm))
