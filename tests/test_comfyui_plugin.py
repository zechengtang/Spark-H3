from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import pytest

COMFY = Path(__file__).resolve().parents[2] / "ComfyUI"
if str(COMFY) not in sys.path:
    sys.path.insert(0, str(COMFY))

from comfyui_nodes import (
    LoadMiniMaxH3AVLatentCache,
    MiniMaxH3SolAttentionSM120,
    MiniMaxH3SparkAttentionSM120,
    SaveMiniMaxH3AVLatentCache,
    _RunState,
    _make_spark_layout,
)


def test_h3_av_latent_cache_roundtrip(tmp_path, monkeypatch):
    import sys

    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

        def unbind(self):
            return self.tensors

    fake_comfy = SimpleNamespace(nested_tensor=SimpleNamespace(NestedTensor=NestedTensor))
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", fake_comfy.nested_tensor)
    video = torch.randn(1, 24, 3, 4, 5, dtype=torch.bfloat16)
    audio = torch.randn(1, 32, 2, 7, dtype=torch.bfloat16)
    cache = tmp_path / "av.safetensors"
    latent = {"samples": NestedTensor((video, audio))}
    result = SaveMiniMaxH3AVLatentCache().save(latent, str(cache))
    assert result["result"][0] is latent
    loaded = LoadMiniMaxH3AVLatentCache().load(str(cache))[0]["samples"].unbind()
    torch.testing.assert_close(loaded[0], video)
    torch.testing.assert_close(loaded[1], audio)


def test_comfy_layout_moves_target_video_first_and_roundtrips():
    positions = torch.zeros(11, 3, dtype=torch.float64)
    positions[3:11] = torch.cartesian_prod(
        torch.arange(2), torch.arange(2), torch.arange(2)
    )
    layout = SimpleNamespace(
        segments=[(0, 3, "text"), (3, 11, "video")],
        position_ids=positions,
    )
    spark = _make_spark_layout(layout, 11, torch.device("cpu"))
    assert spark.grid == (2, 2, 2)
    assert spark.video_tokens == 8
    assert spark.permutation.tolist() == list(range(3, 11)) + list(range(3))
    values = torch.arange(11)
    restored = values[spark.permutation][spark.inverse_permutation]
    torch.testing.assert_close(restored, values)


def test_run_state_uses_comfy_model_evaluation_count_and_resets():
    state = _RunState.create(20, 10.0, 0.1)
    assert state.controller.config.total_evaluations == 20
    assert state.warmup_evaluations == 2
    state.begin_evaluation({"sigmas": torch.tensor([1.0])})
    assert state.controller.evaluation_index == 0 and state.is_warmup
    state.begin_evaluation({"sigmas": torch.tensor([0.9])})
    assert state.controller.evaluation_index == 1 and state.is_warmup
    state.begin_evaluation({"sigmas": torch.tensor([0.8])})
    assert state.controller.evaluation_index == 2 and not state.is_warmup
    state.begin_evaluation({"sigmas": torch.tensor([1.0])})
    assert state.controller.evaluation_index == 0


class _FakeAttention:
    heads = 2
    head_dim = 128

    def forward(self, *args, **kwargs):
        raise AssertionError("not called during patch installation")


class MiniMaxH3Model:
    def __init__(self):
        self.blocks = [SimpleNamespace(attn=_FakeAttention()) for _ in range(3)]


class _FakePatcher:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.object_patches = {}
        self.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
        self.block_patches = {}
        self.callbacks = []

    def clone(self):
        clone = _FakePatcher(self.diffusion_model)
        clone.object_patches = dict(self.object_patches)
        clone.block_patches = dict(self.block_patches)
        clone.callbacks = list(self.callbacks)
        return clone

    def get_model_object(self, path):
        obj = self
        for part in path.split("."):
            if part == "diffusion_model":
                obj = self.diffusion_model
            elif part == "model_sampling":
                obj = self.model_sampling
            elif part == "blocks":
                obj = obj.blocks
            elif part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.block_patches[name, block_name, number, transformer_index] = patch

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.append((call_type, key, callback))


def test_spark_node_uses_comfyui_native_block_patches_and_cleanup():
    model = _FakePatcher(MiniMaxH3Model())
    (patched,) = MiniMaxH3SparkAttentionSM120().patch(
        model, True, 20, 10.0, 0.1, 1, 4096, True
    )
    assert patched is not model
    assert sorted(patched.block_patches) == [
        ("dit", "double_block", i, None) for i in range(3)
    ]
    assert patched.object_patches == {}
    assert len(patched.callbacks) == 1
    assert patched.callbacks[0][1] == "spark_h3_native_attention"


def test_sol_node_uses_base_sol_config():
    model = _FakePatcher(MiniMaxH3Model())
    (patched,) = MiniMaxH3SolAttentionSM120().patch(
        model, True, 20, 20.0, 1, 4096, True, 1.0
    )
    assert patched is not model
    forward = patched.object_patches["diffusion_model.blocks.0.attn.forward"]
    state = next(
        cell.cell_contents
        for cell in forward.__closure__
        if isinstance(cell.cell_contents, _RunState)
    )
    assert state.controller.config.sol_tau == 1.0
    assert state.controller.config.sol_route_topk_ratio is None
    assert not state.controller.config.sol_landmark_preprocess


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 CUDA device required",
)
def test_comfy_adapter_all_exact_matches_dense_on_sm120():
    from comfyui_nodes import _ActivationLog, _make_attention_forward

    torch.manual_seed(31)
    heads, dim, hidden = 2, 128, 256

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = dim
            self.qkv_proj = torch.nn.Linear(hidden, 3 * hidden, bias=False)
            self.q_norm = torch.nn.RMSNorm(dim, eps=1e-5)
            self.k_norm = torch.nn.RMSNorm(dim, eps=1e-5)
            self.out_proj = torch.nn.Linear(hidden, hidden, bias=False)

    attn = Attention().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    text_tokens, video_tokens = 7, 512
    sequence_length = text_tokens + video_tokens
    x = torch.randn(sequence_length, hidden, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(sequence_length, 3, dtype=torch.float64)
    positions[text_tokens:] = torch.cartesian_prod(
        torch.arange(8), torch.arange(8), torch.arange(8)
    )
    layout = SimpleNamespace(
        segments=[(0, text_tokens, "text"), (text_tokens, sequence_length, "video")],
        position_ids=positions,
    )
    state = _RunState.create(2, 0.0, 1.0)
    wrapped = _make_attention_forward(
        attn,
        attn.forward,
        layer=0,
        dense_layers=0,
        min_tokens=0,
        strict=True,
        state=state,
        activation_log=_ActivationLog(),
    )

    with torch.no_grad():
        q, k, v = attn.qkv_proj(x).chunk(3, dim=-1)
        q = attn.q_norm(q.view(1, sequence_length, heads, dim)).transpose(1, 2)
        k = attn.k_norm(k.view(1, sequence_length, heads, dim)).transpose(1, 2)
        v = v.view(1, sequence_length, heads, dim).transpose(1, 2)
        dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        dense = attn.out_proj(dense.transpose(1, 2).reshape(sequence_length, hidden))
        actual = wrapped(
            x,
            transformer_options={
                "sigmas": torch.tensor([1.0], device="cuda"),
                "minimax_h3_layout": layout,
            },
        )
    torch.testing.assert_close(actual, dense, atol=0.012, rtol=0.035)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 CUDA device required",
)
def test_comfy_native_chunked_qkv_matches_monolithic_projection():
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.minimax.model import rope_rotation_table
    from comfyui_nodes import _native_spark_qkv

    torch.manual_seed(37)
    heads, dim, hidden, tokens = 2, 128, 256, 73

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = dim
            self.qkv_proj = torch.nn.Linear(hidden, 3 * hidden, bias=False)
            self.q_norm = torch.nn.RMSNorm(dim, eps=1e-5)
            self.k_norm = torch.nn.RMSNorm(dim, eps=1e-5)

    attn = Attention().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    permutation = torch.randperm(tokens, device="cuda")
    rope = rope_rotation_table(
        torch.randn(tokens, dim, device="cuda"), torch.bfloat16
    )

    with torch.no_grad():
        actual = _native_spark_qkv(attn, x, rope, permutation)
        projected = attn.qkv_proj(x.index_select(0, permutation))
        q, k, v = projected.split(hidden, dim=-1)
        q = q.view(1, tokens, heads, dim)
        k = k.view(1, tokens, heads, dim)
        v = v.view(1, tokens, heads, dim)
        comfy.quant_ops.ck.rms_rope_split_half_(
            q,
            k,
            rope.index_select(1, permutation),
            comfy.model_management.cast_to(attn.q_norm.weight, device=x.device),
            comfy.model_management.cast_to(attn.k_norm.weight, device=x.device),
            epsilon=attn.q_norm.eps,
            rot_dim=dim,
        )
    for got, expected in zip(actual, (q, k, v)):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
