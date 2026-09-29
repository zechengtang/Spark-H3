"""Precision-only summary ablations; INT8 attention consumption is orthogonal."""
import math

import pytest
import torch


def test_summary_numeric_modes_validate_and_default_bf16_anchor():
    from h3_sparse_attention.processor import H3SparseAttentionConfig

    baseline = H3SparseAttentionConfig.spark(20)
    assert baseline.sol_global_anchor_dtype == "bfloat16"
    assert baseline.sol_reweight_summary_math == "tensorcore"
    assert baseline.sol_reweight_logmass_key == "stored"
    assert baseline.sol_reweight_components == "full"
    assert H3SparseAttentionConfig.spark(
        20, sol_global_anchor_dtype="float32").sol_global_anchor_dtype == "float32"
    experimental = H3SparseAttentionConfig.spark(
        20, sol_reweight_summary_math="comfy_fp32",
        sol_reweight_logmass_key="pre_round")
    assert experimental.sol_reweight_summary_math == "comfy_fp32"
    with pytest.raises(ValueError, match="sol_reweight_summary_math"):
        H3SparseAttentionConfig.spark(20, sol_reweight_summary_math="unknown")
    with pytest.raises(ValueError, match="sol_reweight_logmass_key"):
        H3SparseAttentionConfig.spark(20, sol_reweight_logmass_key="unknown")
    with pytest.raises(ValueError, match="sol_reweight_components"):
        H3SparseAttentionConfig.spark(20, sol_reweight_components="unknown")
    assert H3SparseAttentionConfig.spark(
        20, sol_reweight_components="bias_only").sol_reweight_components == "bias_only"
    with pytest.raises(ValueError, match="require virtual query summaries"):
        H3SparseAttentionConfig.sol(20, sol_reweight_components="bias_only")
    with pytest.raises(ValueError, match="require virtual query summaries"):
        H3SparseAttentionConfig.sol(20, sol_reweight_summary_math="comfy_fp32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("anchor_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("logmass_key", ["stored", "pre_round"])
def test_comfy_fp32_summary_matches_fp32_reference(anchor_dtype, logmass_key):
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_summaries

    torch.manual_seed(20260926)
    # A short final block also validates partial-block softmax normalization.
    b, t, h, p, d = 1, 91, 2, 2, 128
    k = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    a = torch.randn(b, p, h, d, device="cuda", dtype=torch.float32).to(anchor_dtype)
    ak, av, lm = virtual_summaries(
        a, k, v, summary_math="comfy_fp32", logmass_key=logmass_key)
    assert ak.dtype == av.dtype == torch.bfloat16
    assert lm.dtype == torch.float32
    for parent in range(p):
        for head in range(h):
            for block in range(2):
                lo, hi = block * 64, min((block + 1) * 64, t)
                anchor = a[0, parent, head].float()
                keys = k[0, lo:hi, head].float()
                values = v[0, lo:hi, head].float()
                logits = (keys * anchor).sum(-1) / math.sqrt(d)
                weights = torch.softmax(logits, 0)
                key_fp32 = (weights[:, None] * keys).sum(0)
                value_fp32 = (weights[:, None] * values).sum(0)
                expected_key = key_fp32.to(torch.bfloat16)
                expected_value = value_fp32.to(torch.bfloat16)
                key_for_shift = key_fp32 if logmass_key == "pre_round" else expected_key.float()
                expected_lm = torch.logsumexp(logits, 0) - (anchor * key_for_shift).sum() / math.sqrt(d)
                assert torch.allclose(ak[0, parent, head, block].float(), expected_key.float(), atol=0.008)
                assert torch.allclose(av[0, parent, head, block].float(), expected_value.float(), atol=0.008)
                assert torch.allclose(lm[0, parent, head, block], expected_lm, atol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("summary_math", ["tensorcore", "comfy_fp32"])
def test_logmass_mode_changes_only_bias_not_weighted_key_value(summary_math):
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_summaries

    torch.manual_seed(20260927)
    a = torch.randn(1, 1, 1, 128, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    rounded = virtual_summaries(a, k, v, summary_math=summary_math, logmass_key="stored")
    unrounded = virtual_summaries(a, k, v, summary_math=summary_math, logmass_key="pre_round")
    assert torch.equal(rounded[0], unrounded[0])
    assert torch.equal(rounded[1], unrounded[1])
    assert not torch.equal(rounded[2], unrounded[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("summary_math", ["tensorcore", "comfy_fp32"])
def test_weight_and_mass_ablation_is_independent(summary_math):
    from h3_sparse_attention.sol_numerator_virtual_q import virtual_summaries

    torch.manual_seed(20260930)
    anchor = torch.randn(1, 1, 1, 128, device="cuda", dtype=torch.float32)
    keys = torch.randn(1, 91, 1, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    modes = {
        mode: virtual_summaries(anchor, keys, values, summary_math=summary_math,
                                logmass_key="stored", reweight_components=mode)
        for mode in ("full", "weights_only", "bias_only", "none")
    }
    for part in (0, 1):
        assert torch.equal(modes["full"][part], modes["weights_only"][part])
        assert torch.equal(modes["bias_only"][part], modes["none"][part])
    for block, (start, stop) in enumerate(((0, 64), (64, 91))):
        expected_k = keys[:, start:stop].float().mean(1).to(torch.bfloat16)
        expected_v = values[:, start:stop].float().mean(1).to(torch.bfloat16)
        assert torch.allclose(modes["none"][0][:, 0, :, block], expected_k, atol=0.008)
        assert torch.allclose(modes["none"][1][:, 0, :, block], expected_v, atol=0.008)
        expected_logmass = math.log(stop - start)
        assert torch.allclose(modes["none"][2][..., block],
                              torch.full_like(modes["none"][2][..., block], expected_logmass),
                              atol=1e-5)
        assert torch.allclose(modes["weights_only"][2][..., block],
                              modes["none"][2][..., block], atol=1e-5)
        assert not torch.allclose(modes["full"][2][..., block],
                                  modes["weights_only"][2][..., block], atol=1e-4)
        assert not torch.allclose(modes["bias_only"][2][..., block],
                                  modes["none"][2][..., block], atol=1e-4)
        if summary_math == "comfy_fp32":
            logits = (keys[0, start:stop, 0].float() * anchor[0, 0, 0]).sum(-1) / math.sqrt(128)
            mean_key = expected_k[0, 0].float()
            expected_bias = torch.logsumexp(logits, 0) - (
                anchor[0, 0, 0] * mean_key).sum() / math.sqrt(128)
            assert torch.allclose(modes["bias_only"][2][0, 0, 0, block],
                                  expected_bias, atol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reweight_components", ["full", "weights_only", "bias_only", "none"])
def test_comfy_summary_runs_through_fused_query_attention(monkeypatch, reweight_components):
    from h3_sparse_attention.sol_numerator_virtual_q import (
        build_virtual_anchors, reduce_virtual_key_centroids, virtual_q_attention)

    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (12, 0)):
        pytest.skip("fused virtual-query kernel unavailable")
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    torch.manual_seed(20260928)
    q, k, v = [torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    ranges = torch.tensor([[0, 128]], device="cuda", dtype=torch.int64)
    mapping = torch.zeros(2, device="cuda", dtype=torch.int64)
    anchor = build_virtual_anchors(q, ranges, dtype=torch.float32)
    key_centroids = reduce_virtual_key_centroids(k)
    threshold = torch.full((1, 2, 1), 5.0, device="cuda")
    output = virtual_q_attention(
        q, k, v, virtual_ranges=ranges, leaf_to_virtual=mapping,
        virtual_anchors=anchor, key_centroids=key_centroids, threshold=threshold,
        force_local_blocks=False, summary_math="comfy_fp32",
        logmass_key="pre_round", anchor_dtype=torch.float32,
        reweight_components=reweight_components)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reweight_components", ["full", "bias_only", "none"])
def test_comfy_summary_runs_through_block_tail_fallback(monkeypatch, reweight_components):
    from h3_sparse_attention.sol_numerator_virtual_q import (
        build_virtual_anchors, virtual_q_attention)
    from sol_attn.preprocess import _reduce_kv

    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (12, 0)):
        pytest.skip("Sol exact fallback unavailable")
    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "0")
    torch.manual_seed(20260929)
    q, k, v = [torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    ranges = torch.tensor([[0, 128]], device="cuda", dtype=torch.int64)
    mapping = torch.zeros(2, device="cuda", dtype=torch.int64)
    anchor = build_virtual_anchors(q, ranges, dtype=torch.float32)
    key_centroids, value_sums = _reduce_kv(k, v)
    threshold = torch.full((1, 2, 1), 5.0, device="cuda")
    output = virtual_q_attention(
        q, k, v, virtual_ranges=ranges, leaf_to_virtual=mapping,
        virtual_anchors=anchor, key_centroids=key_centroids,
        value_sums=value_sums, threshold=threshold, force_local_blocks=False,
        tail_granularity="block", summary_math="comfy_fp32",
        logmass_key="pre_round", anchor_dtype=torch.float32,
        reweight_components=reweight_components)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()
