import pytest
import torch
from h3_sparse_attention import H3SparseAttentionConfig, install_h3_spark_attn
from h3_sparse_attention.landmark_virtual_q import target_virtual_query_layout
from h3_sparse_attention.reblock_hierarchy import build_reblock_hierarchy


def test_installer_defaults_and_target_frontier():
    plugin=install_h3_spark_attn(object(),num_inference_steps=20)
    cfg=plugin.config
    assert cfg.method=='sol' and cfg.total_evaluations==19
    assert cfg.sol_route_topk_ratio==.1 and cfg.sol_route_topk_cutoff_mode=='gemm_radix'
    assert cfg.sol_landmark_preprocess and cfg.sol_landmark_preprocess_version=='v2'
    assert cfg.sol_virtual_query_target_blocks==189 and cfg.sol_virtual_query_levels_up is None
    assert not cfg.sol_local_blocks_enabled
    assert cfg.landmark_tree_v2_children==16 and cfg.landmark_tree_v2_landmark_mode=='midpoint'
    assert cfg.landmark_tree_v2_landmark_count==32
    h=build_reblock_hierarchy(72576,cfg.landmark_tree_v2_children,grid_shape=(72,24,42))
    layout=target_virtual_query_layout(72576,73565,hierarchy=h)
    assert h.roots==((0,1134),)
    assert layout['metadata']['active_size_counts']=={4480:2,4544:14}


def test_overrides_and_plain_sol_opt_in():
    cfg=H3SparseAttentionConfig.spark(20,sol_virtual_query_levels_up=2,landmark_tree_v2_children=8)
    assert cfg.sol_virtual_query_levels_up==2 and cfg.sol_virtual_query_target_blocks is None
    assert cfg.landmark_tree_v2_children==8
    assert H3SparseAttentionConfig.sol(20).sol_virtual_query_target_blocks is None
    with pytest.raises(ValueError,match='choose either'):
        H3SparseAttentionConfig.spark(sol_virtual_query_levels_up=2,sol_virtual_query_target_blocks=189)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('fused', ['0', '1'])
def test_spark_inference_matches_dense_when_all_blocks_exact(monkeypatch, fused):
    from test_sol_spark import TinyTransformer
    monkeypatch.setenv('H3_SPARK_REWEIGHT_FUSED', fused)
    if fused == '1' and torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (12, 0)):
        pytest.skip('fused kernel requires SM90/SM100/SM120')
    torch.manual_seed(27)
    model = TinyTransformer().to(device='cuda', dtype=torch.bfloat16).eval()
    attn = model.transformer_blocks[0].attn
    original = attn.get_processor()
    # Include a partial video block and context so sink alignment is exercised.
    x = torch.randn(1, 521, 128, device='cuda', dtype=torch.bfloat16)
    tags = torch.cat([torch.ones(7), torch.zeros(514)]).to(device='cuda', dtype=torch.long)
    pos = torch.zeros(521, 3, device='cuda', dtype=torch.long)
    pos[7:, 2] = torch.arange(514, device='cuda')
    with torch.no_grad():
        expected = model(x)
        with install_h3_spark_attn(model, num_inference_steps=3, warmup_percent=0,
                                   sol_dense_layers=0, sol_route_topk_ratio=1.0) as plugin:
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
