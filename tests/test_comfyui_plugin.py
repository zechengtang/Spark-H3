from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
    state = _RunState.create(20, 0.1, 0.1)
    assert state.steps == 20
    assert state.warmup_evaluations == 2
    state.begin_evaluation({"sigmas": torch.tensor([1.0])})
    assert state.controller.evaluation_index == 0 and state.is_warmup
    state.begin_evaluation({"sigmas": torch.tensor([0.9])})
    assert state.controller.evaluation_index == 1 and state.is_warmup
    state.begin_evaluation({"sigmas": torch.tensor([0.8])})
    assert state.controller.evaluation_index == 2 and not state.is_warmup
    state.begin_evaluation({"sigmas": torch.tensor([1.0])})
    assert state.controller.evaluation_index == 0


def test_spark_warmup_steps_mode_and_limits():
    fixed = _RunState.create(20, 0.1, 0.1, warmup_mode="warmup_steps", warmup_steps=4)
    assert fixed.warmup_evaluations == 4
    assert _RunState.create(3, 0.1, 0.1, warmup_mode="warmup_steps", warmup_steps=4).warmup_evaluations == 3
    assert _RunState.create(20, 0.1, 0.1, warmup_mode="warmup_steps", warmup_steps=0).warmup_evaluations == 0
    assert _RunState.create(20, 0.1, 0.1).warmup_evaluations == 2
    assert _RunState.create(20, 0.2, 0.2).warmup_evaluations == 4
    for index in range(5):
        fixed.begin_evaluation({"sigmas": torch.tensor([1.0 - index * 0.1])})
        assert fixed.is_warmup == (index < 4)
    inputs = MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]
    assert inputs["warmup_mode"][0] == ["warmup_ratio", "warmup_steps"]
    assert inputs["warmup_ratio"][1]["default"] == 0.2
    assert list(inputs).index("warmup_mode") < list(inputs).index("warmup_ratio") < list(inputs).index("warmup_steps")
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["topk_ratio"][1]["default"] == 0.2
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["min_tokens"][1]["default"] == 12288
    assert "warmup_mode" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["required"]
    with pytest.raises(ValueError, match="warmup_mode"):
        _RunState.create(20, 0.1, 0.1, warmup_mode="invalid")
    with pytest.raises(ValueError, match="warmup_steps"):
        _RunState.create(20, 0.1, 0.1, warmup_steps=-1)
    with pytest.raises(ValueError, match="warmup_ratio"):
        _RunState.create(20, 20.0, 0.1)


def test_spark_ablation_modes_are_orthogonal():
    full = _RunState.create(20, 0.2, 0.1, "full").controller
    optimized = _RunState.create(20, 0.2, 0.1, "full_group841").controller
    reuse = _RunState.create(20, 0.2, 0.1, "full_reuse2").controller

    assert full.config.topk_ratio == 0.1
    assert full.config.topk_mode == "topk_ratio"
    assert full.config.topk_blocks == 228
    fixed = _RunState.create(20, 0.2, 0.1, topk_mode="topk_blocks", topk_blocks=12)
    assert fixed.controller.config.topk_mode == "topk_blocks"
    assert fixed.controller.config.topk_blocks == 12
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["topk_mode"][0] == [
        "topk_ratio", "topk_blocks",
    ]
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["topk_mode"][1]["default"] == "topk_ratio"
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["topk_blocks"][1]["default"] == 228
    assert list(MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]).index("topk_mode") < list(MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]).index("topk_ratio") < list(MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]).index("topk_blocks")
    assert "topk_blocks" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["required"]
    assert full.config.landmark_tree_v2_group_size == 1
    assert optimized.config.landmark_tree_v2_group_size == (8, 4, 1)
    assert reuse.reblock_reuse_layers == 2
    assert reuse.config.landmark_tree_v2_group_size == 1
    assert full.config.video_tail_mode == "dense"
    assert full.config.landmark_tree_v2_midpoint_direction_mode == "fused"
    assert "midpoint_direction_mode" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    assert "midpoint_direction_mode" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]
    assert full.config.global_anchor_dtype == "float32"
    assert "global_anchor_dtype" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    assert "global_anchor_dtype" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]
    fused_midpoint = _RunState.create(20, 0.2, 0.1, midpoint_direction_mode="fused")
    assert fused_midpoint.controller.config.landmark_tree_v2_midpoint_direction_mode == "fused"
    with pytest.raises(ValueError, match="midpoint_direction_mode"):
        _RunState.create(20, 0.2, 0.1, midpoint_direction_mode="invalid")
    for legacy in ("full_target189", "topk10_base", "reblock_only", "reweight_only"):
        with pytest.raises(ValueError, match="only comfy-kitchen global modes"):
            _RunState.create(20, 0.2, 0.1, legacy)
    with pytest.raises(ValueError, match="topk_mode"):
        _RunState.create(20, 0.2, 0.1, topk_mode="invalid")
    with pytest.raises(ValueError, match="topk_blocks"):
        _RunState.create(20, 0.2, 0.1, topk_blocks=0)


def test_comfy_video_tail_rejects_unimplemented_pad():
    from comfyui_backend import ComfySparkConfig

    assert "video_tail_mode" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    assert "ablation_mode" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    for mode in ("pad", "invalid"):
        with pytest.raises(ValueError, match="only supports 'dense'"):
            ComfySparkConfig(video_tail_mode=mode)
        with pytest.raises(ValueError, match="only supports 'dense'"):
            _RunState.create(20, .2, .1, video_tail_mode=mode)


def test_spark_tail_granularity_is_independent():
    for grain in ("block", "query"):
        state = _RunState.create(20, .2, .1, tail_granularity=grain)
        assert state.controller.config.tail_granularity == grain
        assert state.controller.config.global_anchor_dtype == "float32"
    assert _RunState.create(20, .2, .1).controller.config.tail_granularity == "query"
    with pytest.raises(ValueError, match="tail_granularity"):
        _RunState.create(20, .2, .1, tail_granularity="invalid")
    assert "tail_granularity" in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]


def test_spark_local_exact_override_validation():
    from comfyui_backend import ComfySparkConfig

    assert ComfySparkConfig().force_local_blocks is None
    assert ComfySparkConfig(force_local_blocks=True).force_local_blocks is True
    assert ComfySparkConfig(force_local_blocks=False).force_local_blocks is False
    with pytest.raises(ValueError, match="force_local_blocks"):
        ComfySparkConfig(force_local_blocks=1)
    assert "tail_granularity" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]


@pytest.mark.parametrize("video_tokens,total_tokens,topk_mode,topk_blocks,expected_ratio", [
    (64, 70, "topk_ratio", 128, 0.2),
    (65, 70, "topk_blocks", 1, 1.0),
    (640, 646, "topk_blocks", 3, 0.3),
    (640, 646, "topk_blocks", 20, 1.0),
    (64, 64, "topk_ratio", 128, 0.2),
])
def test_spark_only_target_video_queries_use_sparse_output(
    monkeypatch, video_tokens, total_tokens, topk_mode, topk_blocks, expected_ratio
):
    import comfyui_reblock_plan
    from comfyui_backend import (
        ComfyPackedLayout, ComfySparkConfig, ComfySparkController,
        comfy_kitchen_spark_attention,
    )

    generator = torch.Generator().manual_seed(42)
    q, k, v = (
        torch.randn(1, total_tokens, 2, 128, generator=generator).to(torch.bfloat16)
        for _ in range(3)
    )
    identity = torch.arange(video_tokens).view(1, 1, -1).expand(1, 2, -1).contiguous()
    monkeypatch.setattr(
        comfyui_reblock_plan, "build_comfy_reblock_permutations",
        lambda controller, query, key, layout: (identity, None, identity, None),
    )
    call = {}

    def fake_spark(query, key, value, **kwargs):
        call.update(kwargs)
        return torch.full_like(query, 7)

    cuda_module = ModuleType("comfy_kitchen.backends.cuda")
    cuda_module.spark_attn = fake_spark
    monkeypatch.setitem(sys.modules, "comfy_kitchen.backends.cuda", cuda_module)
    layout = ComfyPackedLayout(
        permutation=torch.arange(total_tokens),
        inverse_permutation=torch.arange(total_tokens),
        grid=(1, 1, video_tokens),
        video_tokens=video_tokens,
        sequence_length=total_tokens,
        video_positions=torch.zeros(video_tokens, 3),
    )
    controller = ComfySparkController(ComfySparkConfig(
        topk_mode=topk_mode, topk_blocks=topk_blocks,
    ))
    output = comfy_kitchen_spark_attention(controller, q, k, v, layout, layer=1)

    assert call["topk_ratio"] == pytest.approx(expected_ratio)
    assert call["sink_q"] == [video_tokens // 64, (total_tokens + 63) // 64]
    assert call["sink_blocks"] == [video_tokens // 64, (total_tokens + 63) // 64]
    torch.testing.assert_close(output[:, :video_tokens], torch.full_like(q[:, :video_tokens], 7))
    if video_tokens < total_tokens:
        torch.testing.assert_close(output[:, video_tokens:], torch.full_like(q[:, video_tokens:], 7))
    assert "sink_query_mode" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]


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
        model, True, 20, 0.1, 0.1, 1, 4096, True
    )
    assert patched is not model
    assert sorted(patched.block_patches) == [
        ("dit", "double_block", i, None) for i in range(3)
    ]
    assert patched.object_patches == {}
    assert len(patched.callbacks) == 1
    assert patched.callbacks[0][1] == "spark_h3_native_attention"


def test_sol_node_delegates_to_comfyui_official_patch(monkeypatch):
    model = _FakePatcher(MiniMaxH3Model())
    model.model_options = {"transformer_options": {}}
    captured = {}

    def official(model_arg, **kwargs):
        captured.update(kwargs)
        return model_arg.clone()

    import comfy_extras.nodes_sparse_attention as official_module
    monkeypatch.setattr(official_module, "apply_block_sparse_attention", official)
    (patched,) = MiniMaxH3SolAttentionSM120().patch(
        model, True, 20, 0.2, 1, 4096, True, 1.0
    )
    assert patched is not model
    assert captured["tau"] == 1.0
    assert captured["topk_ratio"] == 0.0
    assert captured["start_percent"] == 0.2
    assert captured["sink_conditioning"] == "exact_kv_and_rows"
    assert captured["extra_tokens"] == 0


def test_comfyui_module_has_no_legacy_attention_interface():
    source = (Path(__file__).resolve().parents[1] / "comfyui_nodes.py").read_text()
    assert "_sol_attention" not in source
    assert "_make_attention_forward" not in source
    assert "import triton" not in source
    assert "cutlass.cute" not in source


def test_comfyui_pipeline_does_not_import_diffusers_pipeline():
    root = Path(__file__).resolve().parents[1]
    node_source = (root / "comfyui_nodes.py").read_text()
    backend_source = (root / "comfyui_backend.py").read_text()
    assert "h3_sparse_attention.processor" not in node_source
    assert "spark_attention_bthd" not in node_source
    assert "spark_attention_bthd(" not in backend_source
    assert "spark_integration" not in backend_source
    assert "H3SparseAttentionConfig" not in backend_source
    assert "_Controller" not in backend_source


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
