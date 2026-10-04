"""Model-free regression checks for the resumed three-task experiment queue."""
from pathlib import Path
import dataclasses
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("steps", [3, 20])
def test_fp32_anchor_changes_only_one_field(steps):
    import run_fp32_anchor_full50_20261004 as fp
    before = dataclasses.asdict(fp.ORIGINAL_CONFIG("legacy_threshold", steps))
    after = dataclasses.asdict(fp.config(fp.METHOD, steps))
    assert [k for k in before if before[k] != after[k]] == ["sol_global_anchor_dtype"]
    assert before["sol_global_anchor_dtype"] == "bfloat16"
    assert after["sol_global_anchor_dtype"] == "float32"


def test_zero_origin_sampler_and_12fps_regression():
    import run_ref2va_8fps_phase_bias_20261004 as ref
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep as Setup
    marker = np.zeros((145, 1, 1, 3), dtype=np.uint8)
    marker[:, 0, 0, 0] = np.arange(145)
    a = Setup._normalize_video_condition(marker, 24, 48, 1, 1, 1, 8)[:, 0, 0, 0]
    c = ref.zero_sample(marker, 24, 145, 1, 1, 1, 8)[:, 0, 0, 0]
    twelve = Setup._normalize_video_condition(marker, 24, 73, 1, 1, 1, 12)[:, 0, 0, 0]
    assert np.array_equal(a, np.arange(1, 145, 3))
    assert len(c) == 39 and np.array_equal(c, np.arange(0, 117, 3))
    assert np.array_equal(twelve, np.arange(0, 145, 2))


def test_audio_roundtrip_without_soundfile_dependency(tmp_path):
    import run_ref2va_8fps_phase_bias_20261004 as ref
    expected = np.random.default_rng(42).normal(size=(2, 3200)).astype(np.float32)
    path = tmp_path / "audio.wav"
    ref.write_audio(path, expected, 32000)
    actual, rate = ref.read_audio(path)
    assert rate == 32000 and np.array_equal(actual, expected.T)


def test_decoder_audio_tensor_conversion():
    import run_ref2va_8fps_phase_bias_20261004 as ref
    source = torch.linspace(-1, 1, 40, dtype=torch.bfloat16).reshape(2, 20)
    actual = ref.audio_numpy(source)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, source.float().numpy())


def test_model_precision_keeps_official_fp32_input_head():
    import run_ref2va_8fps_phase_bias_20261004 as ref
    model = torch.nn.Module()
    model._keep_in_fp32_modules = ["proj_in"]
    model.proj_in = torch.nn.Linear(4, 4, dtype=torch.float32)
    model.block = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    audit = ref.audit_model_precision(model)
    assert audit["parameter_tensor_counts"] == {"bfloat16": 2, "float32": 2}
    model.block.float()
    with pytest.raises(AssertionError):
        ref.audit_model_precision(model)


def test_capture_noise_uses_pipeline_outputs_and_preserves_rng():
    import run_ref2va_8fps_phase_bias_20261004 as ref
    from types import SimpleNamespace
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLatentsStep as Noise, MiniMaxH3PrepareConditionLatentsStep as Condition)
    components = SimpleNamespace(_execution_device="cpu", patch_size=(1, 2, 2), vae_latent_channels=24,
        audio_channels=2, audio_latent_channels=32, keyframe_noise_aug=.999,
        scheduler=SimpleNamespace(scale_noise=lambda sample, sigma, noise: sample * sigma + noise * (1 - sigma)))
    def state():
        return PipelineState(values=dict(condition_latents=[torch.zeros(1, 24, 2, 4, 4)],
            num_condition_video_rows=8, generator=torch.Generator(device="cpu").manual_seed(42),
            num_latent_frames=2, latent_height=4, latent_width=4, num_audio_latents=3))
    original, captured = state(), state()
    Condition()(components, original)
    Noise()(components, original)
    bundle = {}
    with ref.capture_noise(bundle):
        Condition()(components, captured)
        Noise()(components, captured)
    assert set(bundle) == {"condition_noise_0", "condition_rows", "latents", "audio_latents"}
    for key in ("condition_rows", "latents", "audio_latents"):
        assert torch.equal(bundle[key], captured.get(key))
        assert torch.equal(original.get(key), captured.get(key))


@pytest.mark.parametrize("audio_frames,expected_delta", [(414, 0.0), (4, 5 / 3)])
def test_bias_only_layout_and_patch_restoration(audio_frames, expected_delta):
    import run_ref2va_8fps_phase_bias_20261004 as ref
    from types import SimpleNamespace
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep as Layout
    args = (torch.ones(3, dtype=torch.long),
            [SimpleNamespace(kind="video", has_audio=True), SimpleNamespace(kind="audio")],
            [torch.zeros(1, 24, 12, 4, 4)], [torch.zeros(audio_frames, 32), torch.zeros(6, 32)],
            2, 4, 4, 2, (1, 2, 2), 2, 2, 0)
    original = Layout.build_ref2va_packed_sequence
    a, audit = ref.transform_layout(args, "A"), {}
    with ref.phase_layout("B", audit):
        b = Layout.build_ref2va_packed_sequence(*args)
    assert Layout.build_ref2va_packed_sequence is original
    assert audit["downstream_delta"] == pytest.approx(expected_delta)
    expected = a[0].clone()
    indices = a[2][:a[5]]
    expected[indices, 0] += 5 / 3
    expected[int(indices[-1]) + 1:, 0] += expected_delta
    assert torch.equal(expected, b[0])
    assert torch.equal(a[0][:, 1:], b[0][:, 1:])


def test_queue_runs_required_stages_in_order(tmp_path, monkeypatch):
    import run_three_task_queue_4gpu_20261004 as queue
    monkeypatch.setattr(queue, "ROOT", tmp_path / "queue")
    monkeypatch.setattr(queue.study, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(queue, "process_is_route", lambda pid: False)
    monkeypatch.setattr(queue, "rgb_complete", lambda name: name == queue.ROUTE_NAME)
    calls = []
    matrix = dict(status="complete", full50=dict(status="complete", combined_manifests={"legacy_threshold": "baseline"}))
    for duration in (5, 10):
        queue.study.base.write(queue.result_path(queue.ROUTE_NAME, duration), matrix)

    def command(stage, argv):
        calls.append(stage)
        if stage == "task2_ref2va":
            queue.study.base.write(tmp_path / "experiments/ref2va_8fps_phase_bias_20261004/results.json", dict(status="complete"))
        if stage == "task3_fp32_anchor_generation_and_RGB":
            for duration in (5, 10):
                queue.study.base.write(queue.result_path(queue.ANCHOR_NAME, duration),
                    dict(full50=dict(candidate_manifest="fp32", status="complete")))
            monkeypatch.setattr(queue, "rgb_complete", lambda name: True)
    monkeypatch.setattr(queue, "run_command", command)
    monkeypatch.setattr(queue, "vbench", lambda name, duration, manifests, stage, **unused: calls.append(stage))
    monkeypatch.setattr(queue.route, "reference_manifest", lambda duration: Path("dense"))
    queue.main(0)
    assert calls == ["task1_vbench_5s", "task1_vbench_10s", "task2_ref2va",
                     "task3_fp32_anchor_generation_and_RGB", "task3_vbench_5s", "task3_vbench_10s"]
    assert queue.study.base.read(tmp_path / "queue/results.json")["status"] == "complete"
