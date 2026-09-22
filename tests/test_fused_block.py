import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_sparse_attention import install_fp8, install_fused_blocks
from h3_sparse_attention.fp8_linear import Fp8SwiGLUFeedForward

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_block(hidden=128, heads=1, head_dim=128, ffn=256, temb_dim=64):
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock,
    )
    return MiniMaxH3TransformerBlock(
        hidden_size=hidden, num_attention_heads=heads, attention_head_dim=head_dim,
        ffn_dim=ffn, time_embed_dim=temb_dim, norm_eps=1e-5, qk_norm_eps=1e-5)


def _block_inputs(t=259, hidden=128, temb_dim=64, timesteps=2, seed=31):
    torch.manual_seed(seed)
    x = torch.randn(1, t, hidden, device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(timesteps, temb_dim, device="cuda", dtype=torch.float32)
    indices = torch.randint(0, timesteps * 3, (t,), device="cuda")
    return x, temb, indices


def _rel(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


@cuda
def test_fused_block_matches_eager():
    block = _make_block().to("cuda", torch.bfloat16).eval()
    x, temb, indices = _block_inputs()
    with torch.no_grad():
        expected = block(x, temb, indices, None)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([block])

    assert install_fused_blocks(Model()) == 1
    with torch.no_grad():
        actual = block(x, temb, indices, None)
    # Fused is non-bitwise (fp32 accumulation, one rounding): close, not equal.
    assert _rel(actual, expected) < 1e-2
    assert F.cosine_similarity(actual.flatten().float(),
                               expected.flatten().float(), dim=0) > 0.9999


@cuda
def test_fused_block_composes_with_fp8_either_order():
    for fp8_first in (False, True):
        block = _make_block().to("cuda", torch.bfloat16).eval()
        reference = _make_block().to("cuda", torch.bfloat16).eval()
        reference.load_state_dict(block.state_dict())
        x, temb, indices = _block_inputs()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([block])

        model = Model()
        if fp8_first:
            install_fp8(model, min_width=8, skip_end_blocks=0)
            install_fused_blocks(model)
        else:
            install_fused_blocks(model)
            install_fp8(model, min_width=8, skip_end_blocks=0)
        assert type(block.ff) is Fp8SwiGLUFeedForward
        assert "_fast_block_forward" in repr(block.forward)
        with torch.no_grad():
            expected = reference(x, temb, indices, None)
            actual = block(x, temb, indices, None)
        # fp8 in every GEMM plus the non-bitwise fusion: trajectory-level closeness.
        assert F.cosine_similarity(actual.flatten().float(),
                                   expected.flatten().float(), dim=0) > 0.99


def test_install_fused_blocks_rejects_model_without_blocks():
    with pytest.raises(ValueError, match="transformer_blocks"):
        install_fused_blocks(nn.Linear(16, 16))
