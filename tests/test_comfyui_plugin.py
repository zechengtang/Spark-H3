import os
from collections import Counter
from pathlib import Path
import subprocess
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
    MiniMaxH3SparkAttentionSM86,
    MiniMaxH3SparkAttentionSM89,
    MiniMaxH3SparkAttentionSM120,
    SaveMiniMaxH3AVLatentCache,
    _RunState,
    _dense_attention_override,
    _dense_block_args,
    _make_spark_layout,
    _make_kj_low_vram_attention_override,
    _make_kj_low_vram_spark_block_patch,
    _spark_mode_is_dense,
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


def test_comfy_target_scope_keeps_released_layout_path():
    positions = torch.zeros(12, 3, dtype=torch.float64)
    positions[2:6] = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
        dtype=torch.float64,
    )
    positions[8:12] = torch.tensor(
        [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[
            (0, 2, "text"), (2, 6, "ref_img"), (6, 8, "audio"),
            (8, 12, "video"),
        ],
        position_ids=positions,
    )

    implicit = _make_spark_layout(layout, 12, torch.device("cpu"))
    explicit = _make_spark_layout(
        layout, 12, torch.device("cpu"), sparse_scope="target-video"
    )
    assert explicit.grid == implicit.grid == (1, 2, 2)
    assert explicit.video_tokens == implicit.video_tokens == 4
    assert explicit.video_start == implicit.video_start == 8
    assert explicit.permutation.tolist() == implicit.permutation.tolist() == (
        list(range(8, 12)) + list(range(8))
    )


def test_comfy_ref_target_scope_packs_visual_references_and_restores_order():
    positions = torch.zeros(12, 3, dtype=torch.float64)
    positions[2:6] = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]],
        dtype=torch.float64,
    )
    positions[8:12] = torch.tensor(
        [[2, 0, 0], [2, 0, 1], [3, 0, 0], [3, 0, 1]],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[
            (0, 2, "text"), (2, 6, "ref_img"), (6, 8, "audio"),
            (8, 12, "video"),
        ],
        position_ids=positions,
    )

    spark = _make_spark_layout(
        layout, 12, torch.device("cpu"), sparse_scope="joint-video"
    )
    assert spark.grid == (4, 1, 2)
    assert spark.video_tokens == 8
    assert spark.video_start is None
    assert spark.permutation.tolist() == list(range(2, 6)) + list(range(8, 12)) + [0, 1, 6, 7]
    values = torch.arange(12)
    torch.testing.assert_close(
        values[spark.permutation][spark.inverse_permutation], values
    )


def test_comfy_ref_target_scope_uses_flat_grid_for_mixed_resolutions():
    positions = torch.zeros(10, 3, dtype=torch.float64)
    positions[1:3] = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float64)
    positions[6:10] = torch.tensor(
        [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[
            (0, 1, "text"), (1, 3, "ref_img"), (3, 5, "ref_audio"),
            (5, 6, "audio"), (6, 10, "video"),
        ],
        position_ids=positions,
    )

    spark = _make_spark_layout(
        layout, 10, torch.device("cpu"), sparse_scope="joint-video"
    )
    assert spark.grid == (1, 1, 6)
    assert spark.video_tokens == 6
    assert spark.permutation.tolist() == [1, 2, 6, 7, 8, 9, 0, 3, 4, 5]


def test_comfy_ref_target_scope_keeps_static_reference_images_exact():
    positions = torch.zeros(14, 3, dtype=torch.float64)
    positions[2:4] = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.float64)
    positions[4:8] = torch.tensor(
        [[1, 0, 0], [1, 0, 1], [2, 0, 0], [2, 0, 1]],
        dtype=torch.float64,
    )
    positions[10:14] = torch.tensor(
        [[3, 0, 0], [3, 0, 1], [4, 0, 0], [4, 0, 1]],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[
            (0, 2, "text"), (2, 4, "ref_img"), (4, 8, "ref_img"),
            (8, 10, "audio"), (10, 14, "video"),
        ],
        position_ids=positions,
    )

    spark = _make_spark_layout(
        layout, 14, torch.device("cpu"), sparse_scope="joint-video"
    )
    assert spark.video_tokens == 8
    assert spark.permutation.tolist() == (
        list(range(4, 8)) + list(range(10, 14)) + [0, 1, 2, 3, 8, 9]
    )
    assert 2 not in spark.permutation[:spark.video_tokens]
    values = torch.arange(14)
    torch.testing.assert_close(
        values[spark.permutation][spark.inverse_permutation], values
    )


def test_comfy_ref_target_without_reference_is_identical_to_target():
    positions = torch.zeros(6, 3, dtype=torch.float64)
    positions[2:6] = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
        dtype=torch.float64,
    )
    layout = SimpleNamespace(
        segments=[(0, 2, "text"), (2, 6, "video")],
        position_ids=positions,
    )
    target = _make_spark_layout(layout, 6, torch.device("cpu"), "target")
    combined = _make_spark_layout(layout, 6, torch.device("cpu"), "joint-video")
    assert combined.grid == target.grid
    assert combined.video_start == target.video_start
    assert torch.equal(combined.permutation, target.permutation)


def test_comfy_sparse_scope_validation():
    with pytest.raises(ValueError, match="sparse_scope"):
        _RunState.create(20, 0.2, 0.2, sparse_scope="invalid")
    optional = MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    assert optional["sparse_scope"][0] == ["target-video", "joint-video"]
    assert optional["sparse_scope"][1]["default"] == "target-video"
    assert list(optional) == [
        "dense_backend", "tail_granularity", "reblock_layout", "sparse_scope",
        "producer_chunk",
    ]
    assert "sparse_scope" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]


def test_comfy_producer_chunk_validation():
    optional = MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    producer_chunk = optional["producer_chunk"]
    assert producer_chunk[0] == "INT"
    assert producer_chunk[1] == {
        "default": 16384,
        "min": 64,
        "max": 262144,
        "step": 64,
        "tooltip": producer_chunk[1]["tooltip"],
    }
    assert _RunState.create(20, 0.2, 0.2).producer_chunk == 16384
    assert _RunState.create(20, 0.2, 0.2, producer_chunk=8192).producer_chunk == 8192
    with pytest.raises(ValueError, match="producer_chunk"):
        _RunState.create(20, 0.2, 0.2, producer_chunk=0)
    with pytest.raises(ValueError, match="producer_chunk"):
        _RunState.create(20, 0.2, 0.2, producer_chunk=1000)
    assert "producer_chunk" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]


def test_legacy_target_sparse_scope_maps_to_target_video():
    state = _RunState.create(20, 0.2, 0.2, sparse_scope="target")
    assert state.sparse_scope == "target-video"


def test_legacy_ref_target_sparse_scope_maps_to_joint_video():
    state = _RunState.create(20, 0.2, 0.2, sparse_scope="ref+target")
    assert state.sparse_scope == "joint-video"


def test_ref2va_12fps_480p_layout_matches_validated_diffusers_shape():
    from comfy.ldm.minimax.model import PackedLayout

    refs = [
        {
            "kind": "video_audio",
            "latent_t": 17,
            "latent_h": 30,
            "latent_w": 52,
            "ref_audio_t": 207,
        },
        {"kind": "audio", "ref_audio_t": 207},
    ]
    layout = PackedLayout(2807, 37, 48, 84, 207, refs=refs)
    assert layout.seq_len == 47975
    reference_start, reference_stop, kind = layout.segments[2]
    assert kind == "ref_img"
    assert reference_stop - reference_start == 6630
    target_start, target_stop, target_kind = layout.segments[-1]
    assert target_kind == "video"

    spark = _make_spark_layout(
        layout, layout.seq_len, torch.device("cpu"), sparse_scope="joint-video"
    )
    assert spark.video_tokens == (reference_stop - reference_start) + (
        target_stop - target_start
    )
    assert spark.grid == (1, 1, spark.video_tokens)
    assert spark.video_start is None
    assert spark.permutation[:reference_stop - reference_start].tolist() == list(
        range(reference_start, reference_stop)
    )



def test_ref2va_mixed_image_video_layout_sparsifies_only_reference_video():
    from comfy.ldm.minimax.model import PackedLayout

    refs = [
        {"kind": "image", "latent_h": 30, "latent_w": 52},
        {
            "kind": "video_audio", "latent_t": 17,
            "latent_h": 30, "latent_w": 52, "ref_audio_t": 207,
        },
    ]
    layout = PackedLayout(128, 37, 48, 84, 207, refs=refs)
    visual_refs = [segment for segment in layout.segments if segment[2] == "ref_img"]
    assert len(visual_refs) == 2
    image_start, image_stop, _ = visual_refs[0]
    video_start, video_stop, _ = visual_refs[1]
    target_start, target_stop, _ = next(
        segment for segment in layout.segments if segment[2] == "video"
    )

    spark = _make_spark_layout(
        layout, layout.seq_len, torch.device("cpu"), sparse_scope="joint-video"
    )
    expected_sparse = list(range(video_start, video_stop)) + list(
        range(target_start, target_stop)
    )
    assert spark.video_tokens == len(expected_sparse)
    assert spark.permutation[:spark.video_tokens].tolist() == expected_sparse
    assert not any(
        image_start <= index < image_stop
        for index in spark.permutation[:spark.video_tokens].tolist()
    )


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
    assert MiniMaxH3SparkAttentionSM86.TARGET_CAPABILITY == (8, 6)
    assert MiniMaxH3SparkAttentionSM89.TARGET_CAPABILITY == (8, 9)
    assert MiniMaxH3SparkAttentionSM120.TARGET_CAPABILITY == (12, 0)
    assert inputs["mode"][0] == ["fidelity", "speed", "custom", "dense"]
    assert inputs["mode"][1]["default"] == "fidelity"
    assert list(inputs).index("mode") == 1
    assert "enabled" not in inputs
    assert inputs["warmup_mode"][0] == ["warmup_ratio", "warmup_steps"]
    assert inputs["warmup_ratio"][1]["default"] == 0.2
    assert inputs["warmup_steps"][1]["default"] == 4
    assert list(inputs).index("warmup_mode") < list(inputs).index("warmup_ratio") < list(inputs).index("warmup_steps")
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["topk_ratio"][1]["default"] == 0.2
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["dense_layers"][1]["default"] == 0
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["required"]["min_tokens"][1]["default"] == 12288
    assert MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]["dense_backend"] == (
        ["inherit", "kitchen", "sage", "sdpa"],
        {
            "default": "inherit",
            "tooltip": "Dense 预热、固定 dense 层和 dense 回退所用的 attention 后端。inherit 不强制覆盖，沿用模型或 ComfyUI 当前设置；kitchen、sage、sdpa 强制使用对应后端。",
        },
    )
    assert MiniMaxH3SolAttentionSM120.INPUT_TYPES()["required"]["mode"][0] == [
        "fidelity", "speed", "custom", "dense",
    ]
    assert "warmup_mode" in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["required"]
    assert "dense_backend" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]
    with pytest.raises(ValueError, match="warmup_mode"):
        _RunState.create(20, 0.1, 0.1, warmup_mode="invalid")
    with pytest.raises(ValueError, match="warmup_steps"):
        _RunState.create(20, 0.1, 0.1, warmup_steps=-1)
    with pytest.raises(ValueError, match="warmup_ratio"):
        _RunState.create(20, 20.0, 0.1)


def test_spark_public_modes_validate_and_only_dense_bypasses():
    assert not _spark_mode_is_dense("fidelity")
    assert not _spark_mode_is_dense("speed")
    assert not _spark_mode_is_dense("custom")
    assert _spark_mode_is_dense("dense")
    with pytest.raises(ValueError, match="mode"):
        _spark_mode_is_dense("invalid")


def test_dense_backend_override_uses_comfy_registry_and_preserves_inherit(monkeypatch):
    from comfy.ldm.modules import attention as comfy_attention

    calls = []

    def selected(*args, **kwargs):
        calls.append((args, kwargs))
        return "selected"

    def selected_containers(*args, **kwargs):
        return "containers"

    selected.container_function = selected_containers
    requested = []

    def get_attention_function(name, default):
        requested.append((name, default))
        return selected

    monkeypatch.setattr(comfy_attention, "get_attention_function", get_attention_function)
    assert _dense_attention_override("inherit") is None
    override = _dense_attention_override("kitchen")
    assert requested == [("comfy_kitchen_int8", None)]
    assert override("previous", 1, flag=True) == "selected"
    assert calls == [((1,), {"flag": True})]
    assert override.container_function is selected_containers
    assert override.dense_backend == "kitchen"


def test_dense_backend_validation_and_unavailable_dependency(monkeypatch):
    from comfy.ldm.modules import attention as comfy_attention

    with pytest.raises(ValueError, match="dense_backend"):
        _dense_attention_override("unknown")
    monkeypatch.setattr(
        comfy_attention, "get_attention_function", lambda _name, _default: None
    )
    with pytest.raises(RuntimeError, match="sageattention"):
        _dense_attention_override("sage")
    with pytest.raises(RuntimeError, match="INT8 attention"):
        _dense_attention_override("kitchen")


def test_dense_backend_is_scoped_to_dense_block_call():
    original_override = object()
    selected_override = object()
    transformer_options = {
        "sigmas": torch.tensor([1.0]),
        "optimized_attention_override": original_override,
    }
    args = {"img": torch.empty(1), "transformer_options": transformer_options}

    assert _dense_block_args(args, None) is args
    overridden = _dense_block_args(args, selected_override)
    assert overridden is not args
    assert overridden["transformer_options"] is not transformer_options
    assert overridden["transformer_options"]["optimized_attention_override"] is selected_override
    assert transformer_options["optimized_attention_override"] is original_override


def test_spark_ablation_modes_are_orthogonal():
    full = _RunState.create(20, 0.2, 0.1, "full").controller
    topk_only = _RunState.create(20, 0.2, 0.1, "topk_only").controller
    reblock_only = _RunState.create(20, 0.2, 0.1, "reblock_only").controller
    reweight_only = _RunState.create(20, 0.2, 0.1, "reweight_only").controller
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
    assert full.config.landmark_tree_v2_layout_reuse == "q_from_k"
    independent = _RunState.create(
        20, 0.2, 0.1, reblock_layout="independent"
    ).controller
    assert independent.config.landmark_tree_v2_layout_reuse == "independent"
    assert optimized.config.landmark_tree_v2_group_size == (8, 4, 1)
    assert reuse.reblock_reuse_layers == 2
    assert reuse.config.landmark_tree_v2_group_size == 1
    assert (full.reblock_enabled, full.reweight_enabled) == (True, True)
    assert (topk_only.reblock_enabled, topk_only.reweight_enabled) == (False, False)
    assert (reblock_only.reblock_enabled, reblock_only.reweight_enabled) == (True, False)
    assert (reweight_only.reblock_enabled, reweight_only.reweight_enabled) == (False, True)
    assert full.config.video_tail_mode == "dense"
    assert full.config.landmark_tree_v2_midpoint_direction_mode == "legacy"
    assert "midpoint_direction_mode" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    assert "midpoint_direction_mode" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]
    assert full.config.global_anchor_dtype == "float32"
    assert "global_anchor_dtype" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]
    assert "global_anchor_dtype" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]
    reblock_layout = MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]["reblock_layout"]
    assert reblock_layout[0] == ["q_reuse_k", "independent"]
    assert reblock_layout[1]["default"] == "q_reuse_k"
    assert "reblock_layout" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]
    with pytest.raises(ValueError, match="fused.*disabled"):
        _RunState.create(20, 0.2, 0.1, midpoint_direction_mode="fused")
    with pytest.raises(ValueError, match="midpoint_direction_mode"):
        _RunState.create(20, 0.2, 0.1, midpoint_direction_mode="invalid")
    for legacy in ("full_target189", "topk10_base"):
        with pytest.raises(ValueError, match="only comfy-kitchen global modes"):
            _RunState.create(20, 0.2, 0.1, legacy)
    with pytest.raises(ValueError, match="topk_mode"):
        _RunState.create(20, 0.2, 0.1, topk_mode="invalid")
    with pytest.raises(ValueError, match="topk_blocks"):
        _RunState.create(20, 0.2, 0.1, topk_blocks=0)
    with pytest.raises(ValueError, match="reblock_layout"):
        _RunState.create(20, 0.2, 0.1, reblock_layout="invalid")


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
    with pytest.raises(ValueError, match="layout_reuse"):
        ComfySparkConfig(landmark_tree_v2_layout_reuse="k_from_q")
    assert "tail_granularity" not in MiniMaxH3SolAttentionSM120.INPUT_TYPES()["optional"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_size", [1, (8, 4, 1)])
def test_comfy_q_reuse_k_aliases_independent_key_layout_bitwise(
    group_size, monkeypatch
):
    import comfyui_reblock_plan
    from comfyui_backend import (
        ComfyPackedLayout,
        ComfySparkConfig,
        ComfySparkController,
        comfy_kitchen_spark_attention,
    )
    from comfyui_reblock_plan import build_comfy_reblock_permutations

    device = torch.device("cuda")
    video_tokens = 2048
    tokens = video_tokens + 64
    heads = 2
    ids = torch.arange(tokens, device=device)
    layout = ComfyPackedLayout(
        permutation=ids,
        inverse_permutation=ids.clone(),
        grid=(1, 32, 64),
        video_tokens=video_tokens,
        sequence_length=tokens,
        video_positions=torch.zeros(video_tokens, 3, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(20261008)
    query = torch.randn(
        1, tokens, heads, 128,
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    key = torch.randn(
        query.shape,
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    value = torch.randn(
        query.shape,
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    independent_controller = ComfySparkController(ComfySparkConfig(
        landmark_tree_v2_layout_reuse="independent",
        landmark_tree_v2_group_size=group_size,
    ))
    shared_controller = ComfySparkController(ComfySparkConfig(
        landmark_tree_v2_layout_reuse="q_from_k",
        landmark_tree_v2_group_size=group_size,
    ))
    # ComfyUI keeps its default cudaMallocAsync allocator, so its reblock
    # planner must remain eager even across repeated calls.
    for _ in range(3):
        independent = build_comfy_reblock_permutations(
            independent_controller, query, key, layout
        )
        shared = build_comfy_reblock_permutations(
            shared_controller, query, key, layout
        )

        assert torch.equal(shared[0], independent[2])
        assert torch.equal(shared[2], independent[2])
        assert shared[0].data_ptr() == shared[2].data_ptr()
        assert shared[1] is None

    plans = [
        value
        for controller in (independent_controller, shared_controller)
        for value in controller.rope_sol_key_clustering_static.values()
        if hasattr(value, "graph_active")
    ]
    assert plans
    assert all(not plan.graph_active for plan in plans)
    if tuple(torch.cuda.get_device_capability()) not in ((8, 9), (12, 0)):
        return

    expected_layout = (independent[2], None, independent[2], None)
    monkeypatch.setattr(
        comfyui_reblock_plan,
        "build_comfy_reblock_permutations",
        lambda *_args, **_kwargs: expected_layout,
    )
    expected = comfy_kitchen_spark_attention(
        independent_controller, query, key, value, layout, layer=1
    )
    reused_layout = (shared[0], None, shared[2], None)
    monkeypatch.setattr(
        comfyui_reblock_plan,
        "build_comfy_reblock_permutations",
        lambda *_args, **_kwargs: reused_layout,
    )
    actual = comfy_kitchen_spark_attention(
        shared_controller, query, key, value, layout, layer=1
    )
    assert torch.equal(actual, expected)
    assert independent_controller.backend == "comfy-kitchen-spark-global"
    assert shared_controller.backend == "comfy-kitchen-spark-global"
    assert independent_controller.counts["comfy_kitchen_calls"] == 1
    assert shared_controller.counts["comfy_kitchen_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_comfy_reblock_is_safe_with_cuda_malloc_async():
    """Exercise the allocator in a fresh process, where its backend is fixed."""

    probe = r"""
import torch
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations

device = torch.device("cuda")
tokens = 2048
heads = 2
ids = torch.arange(tokens, device=device)
layout = ComfyPackedLayout(
    permutation=ids,
    inverse_permutation=ids.clone(),
    grid=(1, 32, 64),
    video_tokens=tokens,
    sequence_length=tokens,
    video_positions=torch.zeros(tokens, 3, device=device),
)
generator = torch.Generator(device=device).manual_seed(20261009)
query = torch.randn(
    1, tokens, heads, 128,
    generator=generator, device=device, dtype=torch.bfloat16,
)
key = torch.randn_like(query)
controller = ComfySparkController(ComfySparkConfig(
    landmark_tree_v2_layout_reuse="q_from_k",
    landmark_tree_v2_group_size=1,
))
for _ in range(3):
    result = build_comfy_reblock_permutations(controller, query, key, layout)
    torch.cuda.synchronize()
    assert result[0].data_ptr() == result[2].data_ptr()
plans = [
    value for value in controller.rope_sol_key_clustering_static.values()
    if hasattr(value, "graph_active")
]
assert plans and all(not plan.graph_active for plan in plans)
"""
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        timeout=120,
    )


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
    # A joint-video composite has no single original video span. Even when the
    # controller requests direct output, the backend must return packed order
    # so the Python caller can restore it with inverse_permutation.
    controller.direct_output = True
    output = comfy_kitchen_spark_attention(controller, q, k, v, layout, layer=1)

    assert call["topk_ratio"] == pytest.approx(expected_ratio)
    assert call["sink_q"] == [video_tokens // 64, (total_tokens + 63) // 64]
    assert call["sink_blocks"] == [video_tokens // 64, (total_tokens + 63) // 64]
    assert call["video_start"] == 0
    assert controller.output_in_original_layout is False
    torch.testing.assert_close(output[:, :video_tokens], torch.full_like(q[:, :video_tokens], 7))
    if video_tokens < total_tokens:
        torch.testing.assert_close(output[:, video_tokens:], torch.full_like(q[:, video_tokens:], 7))
    assert "sink_query_mode" not in MiniMaxH3SparkAttentionSM120.INPUT_TYPES()["optional"]


def test_target_scope_keeps_backend_direct_output_path(monkeypatch):
    from comfyui_backend import (
        ComfyPackedLayout, ComfySparkConfig, ComfySparkController,
        comfy_kitchen_spark_attention,
    )

    captured = {}
    cuda_module = ModuleType("comfy_kitchen.backends.cuda")

    def fake_spark(query, _key, _value, **kwargs):
        captured.update(kwargs)
        return query

    cuda_module.spark_attn = fake_spark
    monkeypatch.setitem(sys.modules, "comfy_kitchen.backends.cuda", cuda_module)
    tokens, video_tokens, video_start = 70, 64, 6
    q = torch.zeros(1, tokens, 1, 128, dtype=torch.bfloat16)
    layout = ComfyPackedLayout(
        permutation=torch.arange(tokens),
        inverse_permutation=torch.arange(tokens),
        grid=(1, 1, video_tokens),
        video_tokens=video_tokens,
        sequence_length=tokens,
        video_positions=torch.zeros(video_tokens, 3),
        video_start=video_start,
    )
    controller = ComfySparkController(ComfySparkConfig())
    controller.reblock_enabled = False
    controller.reweight_enabled = False
    controller.direct_output = True

    comfy_kitchen_spark_attention(controller, q, q, q, layout, layer=1)

    assert captured["video_start"] == video_start
    assert controller.output_in_original_layout is True


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
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}
        self.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
        self.block_patches = {}
        self.callbacks = []

    def clone(self):
        clone = _FakePatcher(self.diffusion_model)
        clone.model_options = {
            "transformer_options": dict(
                self.model_options.get("transformer_options", {})
            )
        }
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
        model, "fidelity", 20, "warmup_ratio", 0.2, 4,
        0.1, 1, 4096, True, dense_backend="sdpa"
    )
    assert patched is not model
    assert sorted(patched.block_patches) == [
        ("dit", "double_block", i, None) for i in range(3)
    ]
    assert patched.object_patches == {}
    assert not any(
        getattr(patch, "spark_kj_low_vram", False)
        for patch in patched.block_patches.values()
    )
    assert len(patched.callbacks) == 1
    assert patched.callbacks[0][1] == "spark_h3_native_attention"

    captured = {}

    def original_block(args):
        captured.update(args["transformer_options"])
        return "dense"

    block_patch = patched.block_patches[("dit", "double_block", 0, None)]
    result = block_patch(
        {
            "img": torch.empty(8, 16),
            "rope_freqs": None,
            "transformer_options": {"sigmas": torch.tensor([1.0])},
        },
        {"original_block": original_block},
    )
    assert result == "dense"
    override = captured["optimized_attention_override"]
    assert override.dense_backend == "sdpa"

    assert MiniMaxH3SparkAttentionSM120().patch(
        model, "dense", 20, "warmup_ratio", 0.2, 4, 0.1, 1, 4096, True
    ) == (model,)


def test_spark_node_selects_kj_low_vram_patch_only_at_install_time():
    model = _FakePatcher(MiniMaxH3Model())

    def low_vram_forward(*_args, **_kwargs):
        raise AssertionError("compose hook is not called while installing")

    low_vram_forward._uses_optimized_attention = True
    model.model_options["transformer_options"]["sol_take_forward"] = low_vram_forward
    (patched,) = MiniMaxH3SparkAttentionSM120().patch(
        model, "speed", 20, "warmup_ratio", 0.0, 4,
        0.1, 0, 4096, True,
    )
    assert all(
        getattr(patch, "spark_kj_low_vram", False)
        for patch in patched.block_patches.values()
    )

    plain = _FakePatcher(MiniMaxH3Model())
    plain.model_options["transformer_options"]["sol_take_forward"] = lambda: None
    with pytest.raises(RuntimeError, match="incompatible sol_take_forward"):
        MiniMaxH3SparkAttentionSM120().patch(
            plain, "speed", 20, "warmup_ratio", 0.0, 4,
            0.1, 0, 4096, True,
        )


def test_spark_preserves_kjnodes_chunk_feedforward_object_patches():
    model = _FakePatcher(MiniMaxH3Model())
    sentinels = {}
    for index in range(3):
        key = f"diffusion_model.blocks.{index}.mlp.forward"
        sentinels[key] = object()
    model.object_patches.update(sentinels)

    (patched,) = MiniMaxH3SparkAttentionSM120().patch(
        model, "speed", 20, "warmup_ratio", 0.0, 4,
        0.1, 0, 4096, True,
    )
    assert patched.object_patches == sentinels
    assert all(
        not getattr(patch, "spark_kj_low_vram", False)
        for patch in patched.block_patches.values()
    )


def test_kj_low_vram_override_converts_layout_and_restores_native_order(monkeypatch):
    from comfyui_backend import ComfyPackedLayout
    import comfyui_nodes

    permutation = torch.tensor([2, 3, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)
    layout = ComfyPackedLayout(
        permutation=permutation,
        inverse_permutation=inverse,
        grid=(1, 1, 2),
        video_tokens=2,
        sequence_length=4,
        video_positions=torch.zeros(2, 3),
        video_start=2,
    )
    controller = SimpleNamespace(
        counts=Counter(), output_in_original_layout=False,
    )
    state = SimpleNamespace(
        controller=controller,
        spark_layout=lambda *_args: layout,
    )
    activation = SimpleNamespace(hit=lambda *_args: None)
    captured = {}

    def backend(_controller, q, k, v, got_layout, got_layer):
        captured.update(q=q.clone(), k=k.clone(), v=v.clone(),
                        layout=got_layout, layer=got_layer)
        return q.clone()

    monkeypatch.setattr(comfyui_nodes, "comfy_kitchen_spark_attention", backend)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (12, 0))
    options = {"minimax_h3_layout": object()}
    override = _make_kj_low_vram_attention_override(
        7, state, activation, options
    )
    q = torch.arange(1 * 2 * 4 * 128, dtype=torch.bfloat16).reshape(1, 2, 4, 128)
    out = override(None, q, q + 1, q + 2, 2, skip_reshape=True)

    expected_bthd = q.transpose(1, 2).index_select(1, permutation).contiguous()
    torch.testing.assert_close(captured["q"], expected_bthd)
    assert captured["layout"] is layout and captured["layer"] == 7
    expected = expected_bthd.index_select(1, inverse).reshape(1, 4, 256)
    torch.testing.assert_close(out, expected)
    assert controller.counts["sparse:spark_comfy_kj_low_vram"] == 1


def test_kj_low_vram_block_keeps_attention_owned_by_kjnodes(monkeypatch):
    import comfyui_nodes

    monkeypatch.setattr(
        comfyui_nodes, "_native_spark_eligible", lambda *_args: (None, False)
    )
    block = SimpleNamespace(attn=_FakeAttention())
    policy = SimpleNamespace(log_once=lambda *_args: None)
    state = SimpleNamespace(
        controller=SimpleNamespace(
            evaluation_index=0, counts=Counter(), output_in_original_layout=False,
        ),
        warmup_evaluations=0,
    )
    activation = SimpleNamespace(fallbacks=set(), miss=lambda *_args: None)
    patch = _make_kj_low_vram_spark_block_patch(
        block, 3, policy, state, activation, True, None
    )
    captured = {}

    def original_block(got):
        captured.update(got)
        return "sparse-low-vram"

    args = {
        "img": torch.empty(8, 16),
        "rope_freqs": object(),
        "transformer_options": {"minimax_h3_layout": object()},
    }
    assert patch(args, {"original_block": original_block}) == "sparse-low-vram"
    assert "attention" not in captured
    assert captured["transformer_options"][
        "optimized_attention_override"
    ].spark_kj_low_vram


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
        model, "fidelity", 20, "warmup_steps", 0.9, 4, 1, 4096, True, 1.0
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
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((8, 6), (8, 9), (12, 0)),
    reason="SM86, SM89, or SM120 CUDA device required",
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
