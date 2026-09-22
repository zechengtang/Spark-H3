import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_sparse_attention import Fp8Linear, convert_linear_to_fp8, install_fp8
from h3_sparse_attention.fp8_linear import (
    Fp8AttnProcessor, Fp8SwiGLUFeedForward, forward_qkv_shared,
    quantize_rows, quantize_rows_reference,
    quantize_tensor, quantize_tensor_reference,
    swiglu_quantize, swiglu_quantize_tensor,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_quantize_rows_matches_reference():
    torch.manual_seed(21)
    x = torch.randn(517, 4099, device="cuda", dtype=torch.bfloat16) * 3
    y, scale = quantize_rows(x)
    y_ref, scale_ref = quantize_rows_reference(x)
    assert y.dtype == torch.float8_e4m3fn
    assert torch.equal(scale, scale_ref)
    assert torch.equal(y.view(torch.uint8), y_ref.view(torch.uint8))


@cuda
def test_quantize_tensor_matches_reference():
    torch.manual_seed(22)
    x = torch.randn(517, 4099, device="cuda", dtype=torch.bfloat16) * 3
    y, scale = quantize_tensor(x)
    y_ref, scale_ref = quantize_tensor_reference(x)
    assert scale.shape == (1, 1)
    assert torch.equal(scale, scale_ref)
    assert torch.equal(y.view(torch.uint8), y_ref.view(torch.uint8))


def _assert_fp8_matches_activation(y, scale, activation):
    """The kernel evaluates a * g * sigmoid(g) in one fp32 expression while eager goes
    through bf16 F.silu with two roundings, so neither the scales nor the codes are
    bitwise -- the checkable property is that dequantisation lands within fp8's own
    quantisation error of the true activation."""
    dequant = y.float() * scale
    ref = activation.float()
    cosine = F.cosine_similarity(dequant.flatten(), ref.flatten(), dim=0)
    assert cosine > 0.999
    assert (dequant - ref).abs().max() <= ref.abs().max() * 2**-3


@cuda
def test_swiglu_quantize_matches_fused_reference():
    torch.manual_seed(23)
    h = torch.randn(256, 4104 * 2, device="cuda", dtype=torch.bfloat16)
    y, scale = swiglu_quantize(h)
    a, gate = h.chunk(2, dim=-1)
    _assert_fp8_matches_activation(y, scale, a * F.silu(gate))
    ref_scale = (a.float() * F.silu(gate.float())).abs().amax(dim=1, keepdim=True) / 448.0
    assert torch.allclose(scale, ref_scale.clamp_min(1e-12), rtol=2**-7)


@cuda
def test_swiglu_quantize_tensor_matches_fused_reference():
    torch.manual_seed(24)
    h = torch.randn(256, 4104 * 2, device="cuda", dtype=torch.bfloat16)
    y, scale = swiglu_quantize_tensor(h)
    a, gate = h.chunk(2, dim=-1)
    act = a * F.silu(gate)  # the kernel quantises this bf16 rounding of the activation
    _assert_fp8_matches_activation(y, scale, act)
    ref_scale = (act.float().abs().amax() / 448.0).clamp_min(1e-12).reshape(1, 1)
    assert torch.equal(scale, ref_scale)


@cuda
def test_fp8_linear_close_to_bf16():
    torch.manual_seed(25)
    linear = nn.Linear(5376, 7168, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(259, 5376, device="cuda", dtype=torch.bfloat16)
    expected = linear(x)
    actual = Fp8Linear(linear)(x)
    assert actual.shape == expected.shape
    cosine = F.cosine_similarity(expected.flatten().float(), actual.flatten().float(), dim=0)
    assert cosine > 0.998


def _fake_model(num_blocks=10, width=16, narrow=4):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.wide = nn.Linear(width, width, bias=False)
            self.narrow = nn.Linear(narrow, width, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(Block() for _ in range(num_blocks))
            # Wide, but outside the blocks: must stay bf16 (context_embedder analogue).
            self.context_embedder = nn.Linear(width, width, bias=False)

    return Model()


def test_convert_scopes_to_transformer_blocks():
    model = _fake_model()
    swapped = convert_linear_to_fp8(model, min_width=8, skip_end_blocks=2)
    assert swapped == 6  # one wide Linear per middle block (10 - 2*2)
    for index, block in enumerate(model.transformer_blocks):
        if index < 2 or index >= 8:
            assert type(block.wide) is nn.Linear
        else:
            assert type(block.wide) is Fp8Linear
        assert type(block.narrow) is nn.Linear
    assert type(model.context_embedder) is nn.Linear


def test_convert_rejects_model_without_blocks():
    with pytest.raises(ValueError, match="transformer_blocks"):
        convert_linear_to_fp8(nn.Linear(16, 16))


def test_install_fp8_accepts_pipeline_or_transformer():
    class Pipe:
        def __init__(self):
            self.transformer = _fake_model()

    pipe = Pipe()
    assert install_fp8(pipe, min_width=8, skip_end_blocks=2) == 6
    assert type(pipe.transformer.transformer_blocks[4].wide) is Fp8Linear

    model = _fake_model()
    assert install_fp8(model, min_width=8, skip_end_blocks=0) == 10


@cuda
def test_forward_qkv_shared_bitwise_and_fallback():
    from types import SimpleNamespace

    torch.manual_seed(26)
    projections = [Fp8Linear(nn.Linear(64, 64, bias=False, device="cuda",
                                       dtype=torch.bfloat16))
                   for _ in range(3)]
    attn = SimpleNamespace(to_q=projections[0], to_k=projections[1], to_v=projections[2])
    x = torch.randn(1, 130, 64, device="cuda", dtype=torch.bfloat16)
    shared = forward_qkv_shared(attn, x)
    for projection, shared_out in zip(projections, shared):
        assert torch.equal(shared_out, projection(x))

    attn.to_k = nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    assert forward_qkv_shared(attn, x) is None


class _FfBlock(nn.Module):
    def __init__(self, dim=16, inner=32):
        super().__init__()
        from diffusers.models.attention import FeedForward
        self.ff = FeedForward(dim, inner_dim=inner, activation_fn="swiglu", bias=False)


class _FfModel(nn.Module):
    def __init__(self, num_blocks=4, **kwargs):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            _FfBlock(**kwargs) for _ in range(num_blocks))


def test_convert_fuses_swiglu_feedforward():
    model = _FfModel()
    convert_linear_to_fp8(model, min_width=8, skip_end_blocks=1)
    for index, block in enumerate(model.transformer_blocks):
        if index in (0, 3):
            assert type(block.ff) is not Fp8SwiGLUFeedForward
        else:
            assert type(block.ff) is Fp8SwiGLUFeedForward
            assert type(block.ff.proj) is Fp8Linear
            assert type(block.ff.down) is Fp8Linear


@cuda
def test_fused_feedforward_close_to_bf16():
    torch.manual_seed(27)
    model = _FfModel(num_blocks=1, dim=5376, inner=14336).to("cuda", torch.bfloat16).eval()
    ff_ref = _FfBlock(dim=5376, inner=14336).to("cuda", torch.bfloat16).eval()
    ff_ref.ff.load_state_dict(model.transformer_blocks[0].ff.state_dict())
    x = torch.randn(1, 259, 5376, device="cuda", dtype=torch.bfloat16) * 3
    with torch.no_grad():
        expected = ff_ref.ff(x)
        convert_linear_to_fp8(model, min_width=8, skip_end_blocks=0)
        actual = model.transformer_blocks[0].ff(x)
    assert actual.shape == expected.shape
    # Two fp8 GEMMs with an fp8 activation between them compound: ~0.998 per GEMM.
    cosine = F.cosine_similarity(expected.flatten().float(), actual.flatten().float(), dim=0)
    assert cosine > 0.995


@cuda
def test_install_fp8_attention_processor():
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    class AttnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([
                nn.ModuleDict({"attn": MiniMaxH3Attention(128, 1, 128)})])

    torch.manual_seed(28)
    model = AttnModel().to("cuda", torch.bfloat16).eval()
    attn = model.transformer_blocks[0].attn
    x = torch.randn(1, 259, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = attn(x)
        install_fp8(model, min_width=8, skip_end_blocks=0)
        assert type(attn.get_processor()) is Fp8AttnProcessor
        actual = attn(x)
    # fp8 q/k/v feed a bf16 softmax and the fp8 to_out: compounding as above.
    cosine = F.cosine_similarity(expected.flatten().float(), actual.flatten().float(), dim=0)
    assert cosine > 0.995


@cuda
def test_fp8_processor_composes_with_sparse_plugin():
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    from h3_sparse_attention import install_h3_sol_attn
    from h3_sparse_attention.processor import _H3SparseProcessor

    class AttnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([
                nn.ModuleDict({"attn": MiniMaxH3Attention(128, 1, 128)})])

        def forward(self, hidden_states, token_tags=None, position_ids=None):
            return self.transformer_blocks[0].attn(hidden_states)

    torch.manual_seed(29)
    model = AttnModel().to("cuda", torch.bfloat16).eval()
    attn = model.transformer_blocks[0].attn
    install_fp8(model, min_width=8, skip_end_blocks=0)
    fp8_processor = attn.get_processor()

    plugin = install_h3_sol_attn(model, num_inference_steps=3, warmup_percent=0,
                                 sol_dense_layers=0)
    with plugin:
        sparse_processor = attn.get_processor()
        assert type(sparse_processor) is _H3SparseProcessor
        # The plugin adopts the fp8 processor as its dense fallback...
        assert sparse_processor.original is fp8_processor
        x = torch.randn(1, 259, 128, device="cuda", dtype=torch.bfloat16)
        tags = torch.cat([torch.ones(3), torch.zeros(256)]).to("cuda", dtype=torch.long)
        pos = torch.zeros(259, 3, device="cuda", dtype=torch.long)
        pos[3:, 2] = torch.arange(256, device="cuda")
        with torch.no_grad():
            model(x, token_tags=tags, position_ids=pos)
    # ... and restores it on exit.
    assert attn.get_processor() is fp8_processor
