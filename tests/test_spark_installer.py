import pytest
import torch
from h3_sparse_attention import H3SparseAttentionConfig, install_h3_spark_attn
from h3_sparse_attention.landmark_virtual_q import virtual_query_layout
from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy


def test_installer_defaults_and_global_frontier():
    plugin=install_h3_spark_attn(object(),num_inference_steps=20)
    cfg=plugin.config
    assert cfg.method=='sol' and cfg.total_evaluations==19
    assert cfg.sol_route_topk_ratio==.1 and cfg.sol_route_topk_cutoff_mode=='gemm_radix'
    assert cfg.sol_route_topk_execution=='threshold'
    assert cfg.sol_sparse_video_scope=='target'
    assert cfg.sol_global_anchor_dtype=='bfloat16'
    assert cfg.sol_force_local_blocks is None and not cfg.sol_local_blocks_enabled
    assert cfg.sol_video_tail_mode=='dense'
    assert cfg.sol_landmark_preprocess and cfg.sol_landmark_preprocess_version=='v2'
    assert cfg.sol_virtual_query_target_blocks is None and cfg.sol_virtual_query_levels_up==99
    assert not cfg.sol_local_blocks_enabled
    assert cfg.landmark_tree_v2_children==16 and cfg.landmark_tree_v2_landmark_mode=='midpoint'
    assert cfg.landmark_tree_v2_landmark_count==32
    assert cfg.landmark_tree_v2_midpoint_direction_mode == 'legacy'
    h=build_reblock_hierarchy(72576,cfg.landmark_tree_v2_children,grid_shape=(72,24,42))
    layout=virtual_query_layout(72576,73565,cfg.sol_virtual_query_levels_up,hierarchy=h)
    assert h.roots==((0,1134),)
    assert layout['metadata']['active_size_counts']=={72576:1}
    assert layout['metadata']['active_virtual_blocks']==1
    assert layout['metadata']['global_video_representative']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_paired_headwise_permutation_matches_two_independent_launches():
    from h3_sparse_attention.landmark_tree_v2_triton import (
        headwise_permute_bthd,
        headwise_permute_bthd_low_memory,
        headwise_permute_pair_bthd,
        headwise_permute_pair_bthd_low_memory,
    )

    torch.manual_seed(91)
    first = torch.randn(2, 193, 3, 128, device='cuda', dtype=torch.bfloat16)
    second = torch.randn_like(first)
    video_tokens = 128
    permutation = torch.stack([
        torch.randperm(video_tokens, device='cuda')
        for _ in range(first.shape[0] * first.shape[2])
    ]).reshape(first.shape[0], first.shape[2], video_tokens)
    expected_first = headwise_permute_bthd(
        first, permutation, video_tokens=video_tokens
    )
    expected_second = headwise_permute_bthd(
        second, permutation, video_tokens=video_tokens
    )
    actual_first, actual_second = headwise_permute_pair_bthd(
        first, second, permutation, video_tokens=video_tokens
    )
    torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
    destination = torch.empty_like(first)
    returned = headwise_permute_bthd(
        first, permutation, video_tokens=video_tokens, out=destination
    )
    assert returned.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(returned, expected_first, rtol=0, atol=0)
    low_single_input = first.clone()
    low_single_ptr = low_single_input.data_ptr()
    low_single = headwise_permute_bthd_low_memory(
        low_single_input, permutation, video_tokens=video_tokens
    )
    assert low_single.data_ptr() == low_single_ptr
    torch.testing.assert_close(low_single, expected_first, rtol=0, atol=0)
    low_first_input, low_second_input = first.clone(), second.clone()
    low_first_ptr, low_second_ptr = (
        low_first_input.data_ptr(), low_second_input.data_ptr()
    )
    low_first, low_second = headwise_permute_pair_bthd_low_memory(
        low_first_input, low_second_input, permutation, video_tokens=video_tokens
    )
    assert low_first.data_ptr() == low_first_ptr
    assert low_second.data_ptr() == low_second_ptr
    torch.testing.assert_close(low_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(low_second, expected_second, rtol=0, atol=0)
    shared_workspace = torch.empty(
        first.shape[0], first.shape[2], video_tokens, first.shape[3],
        device='cuda', dtype=first.dtype,
    )
    workspace_first, workspace_second = first.clone(), second.clone()
    workspace_first_ptr = workspace_first.data_ptr()
    workspace_second_ptr = workspace_second.data_ptr()
    workspace_first, workspace_second = headwise_permute_pair_bthd_low_memory(
        workspace_first, workspace_second, permutation,
        video_tokens=video_tokens, workspace=shared_workspace,
    )
    assert workspace_first.data_ptr() == workspace_first_ptr
    assert workspace_second.data_ptr() == workspace_second_ptr
    torch.testing.assert_close(workspace_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(workspace_second, expected_second, rtol=0, atol=0)
    from h3_sparse_attention.virtual_q_permute import permute_with_virtual_anchors
    ranges = torch.tensor([[0, first.shape[1]]], device='cuda', dtype=torch.int64)
    anchor_source = first.clone()
    expected_q, expected_anchor = permute_with_virtual_anchors(
        anchor_source, permutation, ranges, video_tokens=video_tokens
    )
    reused_source = first.clone()
    reused_ptr = reused_source.data_ptr()
    reused_q, reused_anchor = permute_with_virtual_anchors(
        reused_source, permutation, ranges, video_tokens=video_tokens,
        reuse_input=True,
    )
    assert reused_q.data_ptr() == reused_ptr
    torch.testing.assert_close(reused_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(reused_anchor, expected_anchor, rtol=0, atol=0)


def test_long_sequence_low_memory_reblock_is_sm80_only(monkeypatch):
    from types import SimpleNamespace
    from h3_sparse_attention.spark_integration import _sm80_low_memory_reblock

    fake = SimpleNamespace(is_cuda=True, shape=(1, 90_001, 1, 128), device='cuda')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _device: (8, 0))
    assert _sm80_low_memory_reblock(fake)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _device: (12, 0))
    assert not _sm80_low_memory_reblock(fake)
    fake.shape = (1, 90_000, 1, 128)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _device: (8, 0))
    assert not _sm80_low_memory_reblock(fake)


def test_overrides_and_plain_sol_opt_in():
    cfg=H3SparseAttentionConfig.spark(20,sol_virtual_query_levels_up=2,landmark_tree_v2_children=8)
    assert cfg.sol_virtual_query_levels_up==2 and cfg.sol_virtual_query_target_blocks is None
    assert cfg.landmark_tree_v2_children==8
    assert H3SparseAttentionConfig.sol(20).sol_virtual_query_target_blocks is None
    assert H3SparseAttentionConfig.sol(20).sol_route_topk_execution == 'threshold'
    assert H3SparseAttentionConfig.sol(20).sol_local_blocks_enabled
    assert H3SparseAttentionConfig.spark(20).sol_force_local_blocks is None
    assert H3SparseAttentionConfig.spark(
        20, sol_landmark_preprocess=False, sol_virtual_query_levels_up=None
    ).sol_local_blocks_enabled
    assert not H3SparseAttentionConfig.sol(
        20, sol_landmark_preprocess=True, sol_landmark_preprocess_version='v2'
    ).sol_local_blocks_enabled
    assert H3SparseAttentionConfig.spark(20, sol_force_local_blocks=True).sol_local_blocks_enabled
    assert not H3SparseAttentionConfig.sol(20, sol_force_local_blocks=False).sol_local_blocks_enabled
    assert H3SparseAttentionConfig.spark(
        20, sol_route_topk_execution='threshold'
    ).sol_route_topk_execution == 'threshold'
    assert H3SparseAttentionConfig.spark(
        20, landmark_tree_v2_midpoint_direction_mode='fused'
    ).landmark_tree_v2_midpoint_direction_mode == 'fused'
    with pytest.raises(ValueError, match='midpoint_direction_mode'):
        H3SparseAttentionConfig.spark(
            20, landmark_tree_v2_midpoint_direction_mode='invalid'
        )
    target=H3SparseAttentionConfig.spark(20,sol_virtual_query_target_blocks=189)
    assert target.sol_virtual_query_levels_up is None
    assert target.sol_virtual_query_target_blocks==189
    assert (target.sol_virtual_query_min_blocks,target.sol_virtual_query_max_blocks)==(94,284)
    with pytest.raises(ValueError,match="requires Top-K or virtual-query"):
        H3SparseAttentionConfig.sol(20,sol_video_tail_mode='pad')
    with pytest.raises(ValueError,match='choose either'):
        H3SparseAttentionConfig.spark(sol_virtual_query_levels_up=2,sol_virtual_query_target_blocks=189)
    assert H3SparseAttentionConfig.spark(
        20, sol_sparse_video_scope='target_and_condition'
    ).sol_sparse_video_scope == 'target_and_condition'
    with pytest.raises(ValueError, match='sol_sparse_video_scope'):
        H3SparseAttentionConfig.spark(20, sol_sparse_video_scope='all')


def test_exact_block_radius_configuration_and_reblock_guard():
    no_reblock = H3SparseAttentionConfig.sol(
        20, sol_route_topk_ratio=0.1, sol_exact_block_radius=0
    )
    assert no_reblock.sol_local_blocks_enabled
    assert no_reblock.sol_local_block_radius == 0

    for reuse in ("q_from_k", "k_from_q"):
        cfg = H3SparseAttentionConfig.spark(
            20,
            sol_exact_block_radius=2,
            landmark_tree_v2_layout_reuse=reuse,
        )
        assert cfg.sol_local_block_radius == 2

    with pytest.raises(ValueError, match="requires Top-K routing"):
        H3SparseAttentionConfig.sol(20, sol_exact_block_radius=0)
    with pytest.raises(ValueError, match="requires landmark_tree_v2_layout_reuse"):
        H3SparseAttentionConfig.spark(20, sol_exact_block_radius=0)
    with pytest.raises(ValueError, match="not both"):
        H3SparseAttentionConfig.sol(
            20,
            sol_route_topk_ratio=0.1,
            sol_exact_block_radius=0,
            sol_force_local_blocks=False,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        H3SparseAttentionConfig.sol(
            20, sol_route_topk_ratio=0.1, sol_exact_block_radius=True
        )


def test_legacy_local_block_flags_map_to_existing_radius():
    assert H3SparseAttentionConfig.sol(
        20, sol_route_topk_ratio=0.1
    ).sol_local_block_radius == 1
    assert H3SparseAttentionConfig.spark(20).sol_local_block_radius == -1
    assert H3SparseAttentionConfig.spark(
        20, sol_force_local_blocks=True
    ).sol_local_block_radius == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_diffusers_spark_runs_on_sm80(monkeypatch):
    from h3_sparse_attention.processor import PackedLayout, _Controller
    from h3_sparse_attention.spark_integration import spark_attention_bthd

    q = torch.zeros((1, 64, 1, 128), device='cuda', dtype=torch.bfloat16)
    layout = PackedLayout(
        permutation=torch.arange(64, device='cuda'),
        inverse_permutation=torch.arange(64, device='cuda'),
        grid=(1, 8, 8),
        video_tokens=64,
        sequence_length=64,
        video_positions=torch.zeros((64, 3), device='cuda'),
    )
    controller = _Controller(H3SparseAttentionConfig.spark(3))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _device=None: (8, 0))
    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', '1')
    actual = spark_attention_bthd(controller, q, q, q, layout, 0)
    assert actual.shape == q.shape
    assert torch.isfinite(actual).all()
    assert controller.sol_backend == 'sm80_fused_virtual_query'


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason='SM80 CUDA device required',
)
def test_sm80_forced_fused_virtual_query_is_selected(monkeypatch):
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_q_backend

    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', '1')
    q = torch.empty((1, 64, 1, 128), device='cuda', dtype=torch.bfloat16)
    assert virtual_q_backend(q) == 'sm80_fused_virtual_query'
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('fused', ['0', '1'])
@pytest.mark.parametrize('tail_mode', ['dense', 'pad'])
def test_spark_inference_matches_dense_when_all_blocks_exact(monkeypatch, fused, tail_mode):
    from test_sol_spark import TinyTransformer
    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', fused)
    if fused == '1' and torch.cuda.get_device_capability() not in ((8, 0), (9, 0), (10, 0), (12, 0)):
        pytest.skip('fused kernel requires SM80/SM90/SM100/SM120')
    torch.manual_seed(27)
    model = TinyTransformer().to(device='cuda', dtype=torch.bfloat16).eval()
    attn = model.transformer_blocks[0].attn
    original = attn.get_processor()
    # Include a partial video block and context so sink alignment is exercised.
    x = torch.randn(1, 577, 128, device='cuda', dtype=torch.bfloat16)
    tags = torch.cat([torch.ones(63), torch.zeros(514)]).to(device='cuda', dtype=torch.long)
    pos = torch.zeros(577, 3, device='cuda', dtype=torch.long)
    pos[63:, 2] = torch.arange(514, device='cuda')
    with torch.no_grad():
        expected = model(x)
        with install_h3_spark_attn(model, num_inference_steps=3, warmup_percent=0,
                                   sol_dense_layers=0, sol_route_topk_ratio=1.0,
                                   sol_video_tail_mode=tail_mode) as plugin:
            actual = model(x, token_tags=tags, position_ids=pos)
            torch.testing.assert_close(actual, expected, atol=.008, rtol=.025)
            stats = plugin.summary()
            assert stats['processor_calls']['sol_landmark_preprocess_calls'] == 1
            assert stats['processor_calls']['sol_virtual_query_calls'] == 1
            assert stats['sol_virtual_query_layout'] is not None
            assert stats['processor_calls']['sol_dense_context_queries'] == 1
            # Exercise cached plan/graph reuse, then reset before a new prompt.
            repeated = model(x, token_tags=tags, position_ids=pos)
            torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
            plugin.reset()
            assert not plugin.controller.rope_sol_key_clustering_static
            assert plugin.summary()['completed_evaluations'] == 0
        assert attn.get_processor() is original
        assert not model._forward_pre_hooks
        assert not plugin.controller.rope_sol_key_clustering_static
        assert not plugin.controller._virtual_query_layout_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("topk_ratio", [0.1, None])
def test_spark_default_routes_merge_with_reweight(monkeypatch, topk_ratio):
    from h3_sparse_attention.processor import _Controller, _packed_layout, _sol_attention
    from h3_sparse_attention.sol_numerator_virtual_q import (
        build_virtual_anchors, virtual_summaries,
    )
    from h3_sparse_attention import spark_integration as integration
    # Compare the full wrapper to an explicit mathematical exact/skipped merge.
    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', '0')
    torch.manual_seed(28)
    tokens, video = 1159, 1152
    q, k, v = [torch.randn(1, tokens, 1, 128, device='cuda', dtype=torch.bfloat16)
               for _ in range(3)]
    tags = torch.cat([torch.zeros(video), torch.ones(tokens-video)]).cuda().long()
    positions = torch.zeros(tokens, 3, device='cuda', dtype=torch.long)
    positions[:video] = torch.cartesian_prod(torch.arange(72, device='cuda'),
                                            torch.arange(4, device='cuda'),
                                            torch.arange(4, device='cuda'))
    layout = _packed_layout(tags, positions)
    cfg = H3SparseAttentionConfig.spark(3, warmup_percent=0, sol_dense_layers=0,
                                        sol_route_topk_ratio=topk_ratio)
    controller = _Controller(cfg)
    # Capture the source's exact branch route; use it to build the independent oracle.
    import h3_sparse_attention.sol_numerator_virtual_q as virtual
    exact = virtual.exact_attention
    captured = {}
    builder = integration._landmark_tree_v2_qk_block_permutations
    def capture_permutations(*args, **kwargs):
        result = builder(*args, **kwargs)
        captured['inverse'] = result[1]
        return result
    monkeypatch.setattr(integration, '_landmark_tree_v2_qk_block_permutations', capture_permutations)
    def capture(*args, **kwargs):
        result = exact(*args, **kwargs)
        captured['args'] = args
        captured['route'] = result[2]
        return result
    monkeypatch.setattr(virtual, 'exact_attention', capture)
    actual = _sol_attention(controller, q.transpose(1,2), k.transpose(1,2),
                            v.transpose(1,2), layout, 0).transpose(1,2)
    qr, kr, vr = captured['args'][:3]
    ranges, mapping, _ = next(iter(controller._virtual_query_layout_cache.values()))
    anchors = build_virtual_anchors(qr, ranges)
    ak, av, lm = virtual_summaries(anchors, kr, vr)
    route = captured['route'][0,:,0].bool()
    assert (~route[:video//64, :video//64]).any(), 'test must exercise skipped blocks'
    out = torch.empty_like(qr)
    for block in range(video//64):
        rows = slice(block*64, (block+1)*64)
        parent = int(mapping[block])
        exact_tokens = route[block].repeat_interleave(64)[:tokens]
        approx = ~route[block]
        keys = torch.cat([kr[0, exact_tokens, 0].float(), ak[0,parent,0,approx].float()])
        values = torch.cat([vr[0, exact_tokens, 0].float(), av[0,parent,0,approx].float()])
        logits = qr[0,rows,0].float() @ keys.T / (128**.5)
        logits[:, int(exact_tokens.sum()):] += lm[0,parent,0,approx]
        out[0,rows,0] = (logits.softmax(-1) @ values).to(qr.dtype)
    # Context output is separately dense; restore video query ordering for comparison.
    out[:,video:] = torch.nn.functional.scaled_dot_product_attention(
        qr[:,video:].transpose(1,2), kr.transpose(1,2), vr.transpose(1,2)).transpose(1,2)
    inv = captured['inverse']
    expected = integration._headwise_permute_video_tokens(out, inv, video_tokens=video)
    torch.testing.assert_close(actual, expected, atol=.008, rtol=.025)
    if torch.cuda.get_device_capability() in ((9, 0), (10, 0), (12, 0)):
        monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', '1')
        fused = _sol_attention(_Controller(cfg), q.transpose(1,2), k.transpose(1,2),
                               v.transpose(1,2), layout, 0).transpose(1,2)
        torch.testing.assert_close(fused, expected, atol=.008, rtol=.025)
