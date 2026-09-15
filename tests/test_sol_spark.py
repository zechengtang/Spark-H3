import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from h3_sparse_attention import (
    H3SparseAttentionConfig, install_h3_sol_attn, spark_reblock, spark_reweight,
)
from h3_sparse_attention.landmark_tree_v2 import recursive_landmark_tree_v2_reference
from h3_sparse_attention.processor import _packed_layout

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("tokens", [64, 130, 512])
@pytest.mark.parametrize("distance", ["cosine", "euclidean"])
def test_reblock_reference_and_inverse(tokens, distance):
    torch.manual_seed(11)
    x = torch.randn(2, tokens, 8)
    kwargs = dict(grid_shape=(1, 1, tokens), distance=distance, initial_order="hilbert_hwt")
    actual = spark_reblock(x, **kwargs)
    expected = recursive_landmark_tree_v2_reference(x, **kwargs)
    assert torch.equal(actual.permutation, expected.permutation)
    assert torch.equal(actual.permutation.sort(-1).values, torch.arange(tokens).expand(2, -1))
    assert torch.equal(actual.inverse_permutation.gather(-1, actual.permutation),
                       torch.arange(tokens).expand(2, -1))
    assert actual.num_excluded == tokens % 64


@cuda
@pytest.mark.parametrize("distance,group_size,aggregation", [
    ("cosine", 1, "linear"), ("cosine", (8, 4), "linear"),
    ("cosine", 1, "max"), ("euclidean", 1, "linear"),
])
def test_cuda_reblock(distance, group_size, aggregation):
    torch.manual_seed(12)
    x = torch.randn(2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    options = dict(grid_shape=(1, 1, 1024), distance=distance,
                   group_size=group_size, aggregation=aggregation)
    a, b = spark_reblock(x, **options), spark_reblock(x, **options)
    assert torch.equal(a.permutation, b.permutation)
    assert torch.equal(a.inverse_permutation.gather(-1, a.permutation),
                       torch.arange(1024, device="cuda").expand(2, -1))


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("parents", [3, 19])
def test_reweight_against_softmax(dtype, parents):
    torch.manual_seed(13)
    # Strided parent dimension exercises the supported noncontiguous anchors.
    a = torch.randn(2, parents * 2, 2, 128, device="cuda", dtype=dtype)[:, ::2]
    k = torch.randn(2, 137, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    ak, av, lm = spark_reweight(a, k, v)
    for block, start in enumerate(range(0, 137, 64)):
        kb, vb = k[:, start:start + 64], v[:, start:start + 64]
        logits = torch.einsum("bphd,bthd->bpht", a.float(), kb.float()) / math.sqrt(128)
        prob = logits.softmax(-1).to(dtype).float()
        expected_k = torch.einsum("bpht,bthd->bphd", prob, kb.float()).to(dtype)
        expected_v = torch.einsum("bpht,bthd->bphd", prob, vb.float()).to(dtype)
        expected_lm = logits.logsumexp(-1) - (a.float() * ak[:, :, :, block].float()).sum(-1) / math.sqrt(128)
        torch.testing.assert_close(ak[:, :, :, block], expected_k, atol=0.008, rtol=0.015)
        torch.testing.assert_close(av[:, :, :, block], expected_v, atol=0.008, rtol=0.015)
        torch.testing.assert_close(lm[:, :, :, block], expected_lm, atol=0.003, rtol=0.003)


def test_reweight_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        spark_reweight(torch.zeros(1, 1, 1, 128), torch.zeros(1, 64, 1, 128),
                       torch.zeros(1, 64, 1, 128))


@cuda
@pytest.mark.parametrize("sink_start,sink_tokens", [(0, 137), (97, 40)])
def test_sol_exact_sinks(sink_start, sink_tokens):
    from sol_attn import sol_attn
    torch.manual_seed(14)
    q, k, v = [torch.randn(1, 137, 2, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    actual = sol_attn(q, k, v, sink_start=sink_start, sink_tokens=sink_tokens)
    # Three blocks are all exact for the middle query block via the local band.
    rows = slice(None) if sink_tokens == 137 else slice(64, 128)
    expected = F.scaled_dot_product_attention(q[:, rows].transpose(1, 2),
                                               k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(actual[:, rows], expected, atol=0.008, rtol=0.015)


def test_target_suffix_selection():
    tags = torch.tensor([0, 0, 1, 2, 0, 0])
    positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0],
                              [0, 0, 0], [0, 0, 0], [0, 0, 1]])
    layout = _packed_layout(tags, positions, video_indices=torch.tensor([0, 1, 4, 5]),
                            timestep_indices=torch.zeros(6, dtype=torch.long),
                            text_indices=torch.tensor([2]))
    assert layout.video_tokens == 2
    assert layout.permutation.tolist() == [4, 5, 0, 1, 2, 3]


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention
        self.transformer_blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({"attn": MiniMaxH3Attention(128, 1, 128)})])
        # H3 blocks expose attention as an attribute.

    def forward(self, hidden_states, token_tags=None, position_ids=None):
        return self.transformer_blocks[0].attn(hidden_states)


@cuda
def test_h3_processor_context_and_restoration():
    torch.manual_seed(15)
    model = TinyTransformer().to(device="cuda", dtype=torch.bfloat16).eval()
    attn = model.transformer_blocks[0].attn
    original = attn.get_processor()
    x = torch.randn(1, 259, 128, device="cuda", dtype=torch.bfloat16)
    tags = torch.cat([torch.ones(3), torch.zeros(256)]).to(device="cuda", dtype=torch.long)
    pos = torch.zeros(259, 3, device="cuda", dtype=torch.long)
    pos[3:, 2] = torch.arange(256, device="cuda")
    with torch.no_grad():
        expected = model(x)
        plugin = install_h3_sol_attn(model, num_inference_steps=3, warmup_percent=0,
                                     sol_dense_layers=0)
        with pytest.raises(RuntimeError, match="test cleanup"):
            with plugin:
                actual = model(x, token_tags=tags, position_ids=pos)
                torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=0.005, rtol=0.015)
                assert plugin.summary()["processor_calls"]["sparse:sol"] == 1
                assert torch.isfinite(actual).all()
                raise RuntimeError("test cleanup")
        assert attn.get_processor() is original
        assert not model._forward_pre_hooks
        with plugin:
            model(x, token_tags=tags, position_ids=pos)
            assert plugin.summary()["completed_evaluations"] == 1


def test_schedule_and_method_validation():
    assert H3SparseAttentionConfig.sol(20).dense_evaluations == 4
    assert H3SparseAttentionConfig.sol(50).total_evaluations == 49
    with pytest.raises(ValueError, match="only"):
        H3SparseAttentionConfig(method="unsupported")


from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation

@cuda
def test_prepared_cuda_graph_replays_deterministically():
    tokens = 1024
    samples = torch.randn(2, tokens, 128, device="cuda", dtype=torch.bfloat16)
    plan = PreparedLandmarkTreeV2Permutation(
        batch=2,
        tokens=tokens,
        dim=128,
        grid_shape=(1, 1, tokens),
        initial_order="flat",
        device=torch.device("cuda"),
    )
    plan.run(samples)
    first = plan.run(samples)
    second = plan.replay()
    assert plan.graph_active
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@cuda
def test_sol_triton_fallback(monkeypatch):
    import sol_attn.interface as interface
    monkeypatch.setattr(interface, "_cute_runtime_available", lambda: False)
    q, k, v = [torch.randn(1, 137, 1, 128, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    output = interface.sol_attn(q, k, v, sink_start=0, sink_tokens=137,
                                force_local_blocks=False)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(output, expected, atol=0.008, rtol=0.015)
