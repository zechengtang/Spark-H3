#!/usr/bin/env python3
"""Experiment-local 8fps phase ablation: Dense A/B/C, then corrected C Spark.

A preserves the historical 1,4,7,... sampling and physical-time scale3.
B changes only reference-video time bias (+5/3), using the exact A cache.
C samples 0,3,6,... and rebuilds conditioning with truthful 8fps metadata.
No production defaults or spatial coordinate construction are modified.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent / "MiniMax-H3-Benchmark/scripts"))
import ref2va_official_case as official
import run_ref2va_condition_fps_rope_20261003 as old

NAME = "ref2va_8fps_phase_bias_20261004"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
DURATIONS = (5, 10)
ARMS = ("A", "B", "C")
BIAS = 5.0 / 3.0
GPU_IDS = tuple(int(x) for x in os.environ.get("H3_EXPERIMENT_GPUS", "0,1,2,3").split(","))
BASELINE = Path("/autodl-fs/data/h3_outputs/ref2va_768_480_dense_spark_20261003/original768")
QUALITY_SCRIPT = REPO.parent / "MiniMax-H3-Benchmark/scripts/h3_quality_video_pair.py"
FV_EVAL = REPO.parent / "FastVideo/examples/inference/eval"


def configure_load_lock():
    """The old official helper has one global lock, irrespective of slot env."""
    def acquire():
        import fcntl
        slot = int(os.environ.get("H3_WEIGHT_LOAD_LOCK_SLOT", "0"))
        assert 0 <= slot < len(GPU_IDS)
        folder = Path("/root/h3_local")
        folder.mkdir(parents=True, exist_ok=True)
        handle = (folder / f".ref_phase_weight_load_slot{slot}.lock").open("a")
        print("waiting for Ref2VA weight-load slot", slot, flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        print("acquired Ref2VA weight-load slot", slot, flush=True)
        return handle
    official.acquire_weight_load_lock = acquire


def write_audio(path, audio, rate):
    from scipy.io import wavfile
    wavfile.write(str(path), int(rate), np.asarray(audio, dtype=np.float32).T)


def read_audio(path):
    from scipy.io import wavfile
    rate, audio = wavfile.read(str(path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    return audio, int(rate)


def audio_numpy(audio):
    # The audio decoder returns a CUDA tensor even when output_type='np' was
    # requested for video. BF16 must also be converted before numpy().
    import torch
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def audit_model_precision(model):
    """BF16 block stack with the checkpoint's deliberate FP32 IO/time heads.

    Checking the first parameter is invalid: proj_in is intentionally FP32 in
    the official model. Validate every parameter against its released policy,
    without casting any module or changing inference precision.
    """
    import torch
    counts = {"bfloat16": 0, "float32": 0}
    keep = tuple(model._keep_in_fp32_modules)
    for name, value in model.named_parameters():
        expected = torch.float32 if any(key in name for key in keep) else torch.bfloat16
        assert value.dtype == expected, (name, value.dtype, expected)
        counts[str(value.dtype).removeprefix("torch.")] += 1
    assert counts["bfloat16"] > 0
    return dict(requested_dtype="bfloat16", official_keep_in_fp32_modules=list(keep), parameter_tensor_counts=counts)


def accept_runner_repair():
    """Audited infrastructure-only migration before ANY new inference output."""
    path = ROOT / "protocol.json"
    protocol = official.read(path)
    current = source_hashes()
    changed = [p for p, digest in protocol["source_sha256"].items() if current.get(p) != digest]
    assert changed == [str(Path(__file__))], changed
    for arm in ARMS:
        assert not list((OUT / arm).glob("**/latents.pt")), "cannot migrate after inference started"
    old_hash = protocol["source_sha256"][str(Path(__file__))]
    official.write(ROOT / f"protocol.before_runner_repair.{old_hash[:12]}.json", protocol)
    protocol["source_sha256"] = current
    protocol.setdefault("runtime_repairs", []).append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_runner_sha256=old_hash, current_runner_sha256=current[str(Path(__file__))],
        scope="Audio IO adapters; isolated conditioning; four load slots; release compile owner; verify official mixed-precision policy instead of first-parameter dtype",
        experiment_parameters_unchanged=True, core_source_hashes_unchanged=True,
        generated_latents_before_repair=0, existing_conditioning_preserved=True))
    official.write(path, protocol)
    shutil.copy2(__file__, ROOT / f"runner_repaired_source.{current[str(Path(__file__))][:12]}.py")


def write(path, value):
    # Identical decoded outputs legitimately give infinite PSNR. Keep valid
    # JSON while recording that special case explicitly in the comparison.
    def finite(item):
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return "Infinity" if item > 0 else "-Infinity" if item < 0 else None
        return item
    official.write(path, finite(value))


def tensor_sha(value):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def source_hashes():
    import diffusers.modular_pipelines.minimax_h3.before_denoise as bd
    import diffusers.modular_pipelines.minimax_h3.before_encoder as be
    import diffusers.modular_pipelines.minimax_h3.encoders as en
    import diffusers.modular_pipelines.minimax_h3.references as refs
    paths = [Path(__file__), Path(official.__file__), Path(old.__file__),
             Path(bd.__file__), Path(be.__file__), Path(en.__file__), Path(refs.__file__),
             REPO / "h3_sparse_attention/processor.py", REPO / "h3_sparse_attention/spark_integration.py",
             REPO / "h3_sparse_attention/spark_reweight_sm120.py", REPO / "h3_sparse_attention/sol_numerator_virtual_q.py"]
    return {str(p): official.sha(p) for p in paths}


def configure_cache(arm):
    if arm in ("A", "B"):
        old.configure(8)
    else:
        official.CONDITIONING_CACHE_SCHEMA = 408
        official.CACHE_DIR = OUT / "conditioning_C"
        official.CASE_DIR = ROOT / "conditioning_C"


def zero_sample(frames, source_fps, num_frames, multiple, short_edge, max_pixels, target_fps):
    """Keep the old raw slot-count rule, but select the zero-origin source frame.

    Retaining the slot-count and VAE-crop policy controls coverage independently
    of phase: this official case has 39 encoded 8fps frames for both durations.
    """
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    count = math.floor(len(frames) * 8 / source_fps + 0.5)
    indices = np.floor(np.arange(count) * source_fps / 8 + 1e-9).astype(np.int64)
    indices = indices[indices < len(frames)]
    requested = round(num_frames * 8 / 24)
    selected = frames[indices][:requested]
    selected = selected[:old.valid_frames(len(selected))]
    return MiniMaxH3Ref2VASetupStep._normalize_video_condition(
        selected, 8.0, len(selected), multiple, short_edge, max_pixels, 8.0
    )


@contextlib.contextmanager
def corrected_conditioning(observations):
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep as Setup
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep as Text
    original_normalize, original_setup, original_gather = Setup._normalize_video_condition, Setup.__call__, Text._gather_vision_features

    def normalize(frames, source_fps, num_frames, multiple, short_edge, max_pixels, ignored_fps):
        count = math.floor(len(frames) * 8 / source_fps + 0.5)
        indices = np.floor(np.arange(count) * source_fps / 8 + 1e-9).astype(np.int64)
        indices = indices[indices < len(frames)][:round(num_frames * 8 / 24)]
        indices = indices[:old.valid_frames(len(indices))]
        output = original_normalize(frames[indices], 8.0, len(indices), multiple, short_edge, max_pixels, 8.0)
        observations.setdefault("sampling", []).append(dict(source_fps=source_fps, source_indices=indices.tolist(),
            source_times_seconds=(indices / source_fps).tolist(), actual_frames=len(output), reference_fps=8))
        return output

    def setup(self, components, state):
        components, state = original_setup(self, components, state)
        for reference in state.values["normalized_references"]:
            if reference.kind == "video":
                reference.fps = 8.0
        return components, state

    def gather(self, processor, references, ignored_fps):
        assert all(r.fps == 8 for r in references if r.kind == "video")
        result = original_gather(self, processor, references, 8.0)
        observations.setdefault("visual_llm", []).append(dict(reference_fps=8, block_timestamps=result[3],
            rendered_labels=[[f"<{x:.1f} seconds>" for x in timestamps] for timestamps in result[3]]))
        return result

    Setup._normalize_video_condition = staticmethod(normalize)
    Setup.__call__, Text._gather_vision_features = setup, gather
    try:
        yield
    finally:
        Setup._normalize_video_condition = staticmethod(original_normalize)
        Setup.__call__, Text._gather_vision_features = original_setup, original_gather


def transform_layout(args, arm, audit=None):
    """Start from historical physical-time A, and change only B's phase."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep as Layout
    with old.rope_patch(8, "physical_time"):
        result = list(Layout.build_ref2va_packed_sequence(*args))
    positions, video_indices = result[0], result[2]
    condition_indices = video_indices[:result[5]]
    latent_frames = int(args[2][0].shape[2])
    origin = float(positions[condition_indices[0], 0])
    video_span = 3 * sum(BIAS * (1, 4, 4, 4, 4)[i % 5] for i in range(latent_frames))
    audio_span = args[3][0].shape[0] / args[9]
    phase = BIAS if arm == "B" else 0.0
    delta = max(audio_span, video_span + phase) - max(audio_span, video_span)
    after_video = int(condition_indices[-1]) + 1
    before = positions.clone()
    if arm == "B":
        positions[condition_indices, 0] += phase
        positions[after_video:, 0] += delta
    assert torch.equal(positions[:, 1:], before[:, 1:])
    attached_audio = result[3][:args[3][0].shape[0]]
    assert torch.equal(positions[attached_audio], before[attached_audio])
    assert math.isclose(float(positions[after_video, 0]), origin + max(audio_span, video_span + phase), abs_tol=1e-10)
    if audit is not None:
        spatial_rows = condition_indices.numel() // latent_frames
        anchors = positions[condition_indices[::spatial_rows], 0]
        audit.update(origin=origin, scale=3.0, bias_rope_units=phase, phase_seconds=phase / 40,
            latent_frames=latent_frames, video_end=origin + video_span + phase,
            attached_audio_start=origin, attached_audio_end=origin + audio_span,
            next_block_start=origin + max(audio_span, video_span + phase), downstream_delta=delta,
            latent_anchor_seconds=((anchors - origin) / 40).tolist(),
            spatial_coordinates_unchanged=True, attached_audio_coordinates_unchanged=True)
        audit["tensors"] = dict(position_ids=positions.clone(), physical_A_position_ids=before,
            video_indices=result[2], audio_indices=result[3], text_indices=result[4], token_tags=result[1],
            condition_video_indices=condition_indices)
    return tuple(result)


@contextlib.contextmanager
def phase_layout(arm, audit):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep as Layout
    original = Layout.build_ref2va_packed_sequence
    def corrected(*args, **kwargs):
        if kwargs:
            raise ValueError("audit expects positional Ref2VA layout arguments")
        # Restore the unwrapped builder while transform_layout invokes the old physical patch.
        Layout.build_ref2va_packed_sequence = staticmethod(original)
        try:
            return transform_layout(args, arm, audit)
        finally:
            Layout.build_ref2va_packed_sequence = staticmethod(corrected)
    Layout.build_ref2va_packed_sequence = staticmethod(corrected)
    try:
        yield
    finally:
        Layout.build_ref2va_packed_sequence = staticmethod(original)


def layout_arguments(state):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames, video_latent_num_frames
    v = state.values
    return (v["text_token_tags"], v["normalized_references"], v["condition_latents"], v["audio_condition_latents"],
            video_latent_num_frames(v["num_frames"], 17, 5), 48, 84, audio_latent_num_frames(v["num_frames"]),
            (1, 2, 2), 2, 2, 0)


def coordinate_audit():
    """Model-free checks, including cached real conditioning and short-audio counterexample."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep as Setup
    marker = np.zeros((145, 1, 1, 3), dtype=np.uint8)
    marker[:, 0, 0, 0] = np.arange(145)
    sampled_a = Setup._normalize_video_condition(marker, 24, 48, 1, 1, 1, 8)[:, 0, 0, 0]
    sampled_c = zero_sample(marker, 24, 145, 1, 1, 1, 8)[:, 0, 0, 0]
    sampled_12 = Setup._normalize_video_condition(marker, 24, 73, 1, 1, 1, 12)[:, 0, 0, 0]
    assert np.array_equal(sampled_a, np.arange(1, 145, 3))
    assert np.array_equal(sampled_c, np.arange(0, 3 * len(sampled_c), 3))
    assert np.array_equal(sampled_12, np.arange(0, 145, 2))
    result = dict(status="passed", models_loaded=False, A_B_source_indices=sampled_a.tolist(),
                  C_source_indices=sampled_c.tolist(), regression_12fps_source_indices=sampled_12.tolist(), durations={})
    raw = official.decode_and_resample_references(240)[0]
    for duration in DURATIONS:
        configure_cache("A")
        state, path, summary = official.load_conditioning(duration)
        observations = official.read(old.ROOT / "conditioning_observations.json")[f"8fps_{duration}s"]
        assert official.sha(path) == observations["cache_sha256"]
        cached = state.values["normalized_references"][0]
        requested = round(state.values["num_frames"] * 8 / 24)
        expected = Setup._normalize_video_condition(raw.frames, raw.fps, requested, 32, 768, 768 * 1344, 8)
        expected = expected[:old.valid_frames(len(expected))]
        assert np.array_equal(expected, cached.frames)
        a, b = {}, {}
        args = layout_arguments(state)
        layout_a, layout_b = transform_layout(args, "A", a), transform_layout(args, "B", b)
        indices = layout_a[2][:layout_a[5]]
        expect = layout_a[0].clone()
        expect[indices, 0] += BIAS
        expect[int(indices[-1]) + 1:, 0] += b["downstream_delta"]
        assert torch.equal(expect, layout_b[0])
        assert all(torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y for x, y in zip(layout_a[1:], layout_b[1:]))
        # Temporal anchors are the beginning of each nonuniform VAE segment, not evenly spaced latent ordinals.
        anchors = np.cumsum([0] + [(1, 4, 4, 4, 4)[i % 5] for i in range(b["latent_frames"] - 1)])
        assert np.allclose(b["latent_anchor_seconds"], anchors / 8 + 1 / 24, atol=1e-12)
        for arm, audit in [("A", a), ("B", b)]:
            tensors = audit.pop("tensors")
            official.atomic_torch_save(tensors, ROOT / "coordinate_audit" / f"{arm}_{duration}s.pt")
        result["durations"][str(duration)] = dict(conditioning_path=str(path), conditioning_sha256=official.sha(path),
            actual_reference_frames=len(cached.frames), reference_coverage_seconds=len(cached.frames) / 8,
            reference_vae_crop_frames=len(cached.frames), reference_vae_crop_duration_seconds=len(cached.frames) / 8,
            last_sample_A_seconds=(1 + 3 * (len(cached.frames) - 1)) / 24,
            last_sample_C_seconds=3 * (len(cached.frames) - 1) / 24, A=a, B=b, summary=summary,
            A_B_conditioning_identical=True, A_B_only_expected_coordinates_changed=True)
    # Ensure max(end_audio, end_video) really handles a video-dominant reference, even though this real case is audio-dominant.
    small = (torch.ones(3, dtype=torch.long), [SimpleNamespace(kind="video", has_audio=True), SimpleNamespace(kind="audio")],
             [torch.zeros(1, 24, 12, 4, 4)], [torch.zeros(4, 32), torch.zeros(6, 32)], 2, 4, 4, 2, (1, 2, 2), 2, 2, 0)
    short = {}
    transform_layout(small, "B", short)
    assert math.isclose(short["downstream_delta"], BIAS, abs_tol=1e-12)
    short.pop("tensors")
    result["video_dominant_counterexample"] = short
    write(ROOT / "coordinate_audit.json", result)
    return result


def prepare():
    if (ROOT / "protocol.json").exists():
        protocol = official.read(ROOT / "protocol.json")
        if protocol["source_sha256"] != source_hashes():
            raise ValueError("Ref2VA resume source mismatch")
        if (ROOT / "conditioning_C_observations.json").exists() and (ROOT / "coordinate_audit.json").exists():
            configure_cache("C")
            for duration in DURATIONS:
                _, path, _ = official.load_conditioning(duration)
                assert official.sha(path) == official.read(ROOT / "conditioning_C_observations.json")[str(duration)]["sha256"]
            return
    else:
        if OUT.exists() or (ROOT.exists() and any(p.name not in ("coordinate_audit", "coordinate_audit.json") for p in ROOT.iterdir())):
            raise FileExistsError("refusing incomplete root without protocol")
        ROOT.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        protocol = dict(status="preflight", source_sha256=source_hashes(), reference_sha256=official.reference_sha256(),
        model=str(official.MODEL), model_fingerprint=official.model_fingerprint(),
        prompt_sha256=official.sha(official.PROMPT_PATH), seed=42, dtype="bfloat16", requested_steps=20, evaluations=19,
        target_fps=24, target_resolution=[1344, 768], reference_fps=8, durations=list(DURATIONS),
        gpus=list(GPU_IDS), scale=3, bias_rope_units=BIAS, patch_scope="experiment-local 8fps physical_time; H/W unchanged",
        A_reuse="Regenerate A: historical cache and runner hashes are checked, but old inference lacks full core-source provenance",
        A_B_conditioning="same exact verified historical cache file", C_conditioning="rebuilt from zero-origin samples",
        old_runner_sha256=official.read(old.ROOT / "protocol.json")["runner_sha256"],
            current_old_runner_sha256=official.sha(Path(old.__file__)))
    assert protocol["old_runner_sha256"] == protocol["current_old_runner_sha256"], "historical A runner provenance changed"
    write(ROOT / "protocol.json", protocol)
    coordinate_audit()
    observations = {}
    configure_cache("C")
    with corrected_conditioning(observations):
        official.condition()
    official.release_cpu_arenas()
    for duration in DURATIONS:
        state, path, summary = official.load_conditioning(duration)
        ref = state.values["normalized_references"][0]
        assert ref.fps == 8 and ref.frames.shape[1:3] == (768, 1344)
        assert len(ref.frames) == official.read(ROOT / "coordinate_audit.json")["durations"][str(duration)]["actual_reference_frames"]
        raw = official.decode_and_resample_references(duration * 24)[0]
        expected = zero_sample(raw.frames, raw.fps, state.values["num_frames"], 32, 768, 768 * 1344, 8)
        assert np.array_equal(ref.frames, expected), "C actual frames do not match zero-origin sampling"
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep as Text
        _, labels = Text._sample_video_condition_frames(ref.frames, 8.0, 2.0, 2)
        indices = list(range(0, len(ref.frames) * 3, 3))
        audit = {}
        transform_layout(layout_arguments(state), "C", audit)
        official.atomic_torch_save(audit.pop("tensors"), ROOT / "coordinate_audit" / f"C_{duration}s.pt")
        observations[str(duration)] = dict(path=str(path), sha256=official.sha(path), actual_frames=len(ref.frames),
            reference_fps=ref.fps, reference_coverage_seconds=len(ref.frames) / 8,
            reference_vae_crop_frames=len(ref.frames), reference_vae_crop_duration_seconds=len(ref.frames) / 8,
            source_indices=indices, source_times_seconds=[i / 24 for i in indices],
            visual_llm_timestamps=labels, visual_llm_labels=[f"<{x:.1f} seconds>" for x in labels],
            video_source_start_seconds=0.0, attached_audio_source_start_seconds=0.0, layout=audit, summary=summary)
    write(ROOT / "conditioning_C_observations.json", observations)
    shutil.copy2(__file__, ROOT / "runner_source.py")
    write(ROOT / "status.json", dict(status="running", stage="dense_ABC"))


@contextlib.contextmanager
def capture_noise(bundle):
    import diffusers.modular_pipelines.minimax_h3.before_denoise as bd
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLatentsStep as Noise
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareConditionLatentsStep as Condition
    originals = {Noise: Noise.__call__, Condition: Condition.__call__}
    def wrapped(cls):
        def call(self, components, state):
            original_randn = bd.randn_tensor
            def randn(*args, **kwargs):
                result = original_randn(*args, **kwargs)
                key = f"condition_noise_{sum(k.startswith('condition_noise_') for k in bundle)}"
                bundle[key] = result.detach().cpu().clone()
                return result
            if cls is Condition:
                bd.randn_tensor = randn
            try:
                components, state = originals[cls](self, components, state)
            finally:
                bd.randn_tensor = original_randn
            keys = ("latents", "audio_latents") if cls is Noise else ("condition_rows",)
            for key in keys:
                # get_block_state reconstructs INPUTS only; condition_rows is
                # an output, written by the original step into PipelineState.
                bundle[key] = state.get(key).detach().cpu().clone()
            return components, state
        return call
    for cls in originals:
        cls.__call__ = wrapped(cls)
    try:
        yield
    finally:
        for cls, original in originals.items():
            cls.__call__ = original


def spark_config(steps):
    from h3_sparse_attention import H3SparseAttentionConfig
    return H3SparseAttentionConfig.spark(steps, warmup_percent=20 if steps == 20 else 33,
        sol_dense_layers=1, sol_route_topk_ratio=0.1, sol_sparse_video_scope="target_and_condition", sol_log_density=False)


def load_denoiser():
    import torch
    from diffusers import ComponentsManager
    workflow = official.get_workflow()
    for name in ("before_encode", "text_encoder", "vae_encoder", "decode.video", "decode.audio"):
        workflow.sub_blocks.pop(name)
    lock = official.acquire_weight_load_lock()
    try:
        manager = ComponentsManager()
        pipe = workflow.init_pipeline(str(official.MODEL), components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path={"default": str(official.MODEL)},
                             disable_mmap={"transformer_ref": False})
    finally:
        official.release_weight_load_lock(lock)
    pipe.transformer_ref.to("cuda")
    torch.cuda.synchronize()
    print("MODEL PRECISION AUDIT", audit_model_precision(pipe.transformer_ref), flush=True)
    official.release_cpu_arenas()
    return pipe, manager


def generate(pipe, arm, duration, sparse=False, warmup=False):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention
    configure_cache(arm)
    state, cache, summary = official.load_conditioning(duration)
    steps = 3 if warmup else 20
    destination = OUT / arm / f"{duration}s" / ("spark" if sparse else "dense")
    if not warmup and (destination / "meta.json").exists():
        meta = official.read(destination / "meta.json")
        if official.sha(meta["latent_path"]) != meta["latent_sha256"]:
            raise ValueError("resumed latent hash mismatch")
        return
    prepared = official.clone_state(state)
    prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
    coordinate, noise, evaluations = {}, {}, []
    hook = pipe.transformer_ref.register_forward_pre_hook(lambda *unused: evaluations.append(1))
    context = install_h3_sparse_attention(pipe.transformer_ref, spark_config(steps)) if sparse else contextlib.nullcontext(None)
    try:
        with phase_layout(arm, coordinate), capture_noise(noise), context as plugin, torch.inference_mode():
            if plugin:
                plugin.reset()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipe(state=prepared, num_frames=state.values["num_frames"], height=768, width=1344,
                          num_inference_steps=steps, generator=torch.Generator(device="cpu").manual_seed(42),
                          output=["latents", "audio_latents"])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            attention = plugin.summary() if plugin else None
    finally:
        hook.remove()
    assert len(evaluations) == steps - 1
    if sparse:
        assert attention["dense_evaluations"] == (1 if warmup else 4)
        assert attention["sol_sparse_video_scope"] == "target_and_condition"
    if warmup:
        print("WARMUP", arm, duration, sparse, seconds, flush=True)
        return
    payload = {k: result[k].detach().cpu().contiguous() for k in ("latents", "audio_latents")}
    assert all(torch.isfinite(v).all() for v in payload.values())
    latent_path = destination / "latents.pt"
    official.atomic_torch_save(payload, latent_path)
    official.atomic_torch_save(noise, destination / "initial_noise.pt")
    official.atomic_torch_save(coordinate.pop("tensors"), destination / "position_coordinates.pt")
    meta = dict(status="generated", arm=arm, sparse=sparse, duration=duration, seed=42, model_dtype="bfloat16",
        requested_steps=20, evaluations=len(evaluations), requested_frames=duration * 24,
        aligned_frames=state.values["num_frames"], latent_sha256=official.sha(latent_path), latent_path=str(latent_path),
        denoise_seconds=seconds, conditioning_path=str(cache), conditioning_sha256=official.sha(cache),
        conditioning_summary=summary, frames_sha256=tensor_sha(state.values["normalized_references"][0].frames),
        prompt_embeds_sha256=tensor_sha(state.values["prompt_embeds"]),
        initial_noise_sha256={k: tensor_sha(v) for k, v in noise.items()}, coordinate=coordinate,
        source_sha256=source_hashes(), model_precision_audit=audit_model_precision(pipe.transformer_ref), attention_summary=attention,
        sparse_config=__import__("dataclasses").asdict(spark_config(20)) if sparse else None)
    write(destination / "meta.json", meta)
    print("GENERATED", arm, duration, sparse, seconds, flush=True)


def load_decoder():
    import torch
    from diffusers import ComponentsManager
    workflow = official.get_workflow()
    for name in list(workflow.sub_blocks):
        if not name.startswith("decode."):
            workflow.sub_blocks.pop(name)
    lock = official.acquire_weight_load_lock()
    try:
        manager = ComponentsManager()
        decoder = workflow.init_pipeline(str(official.MODEL), components_manager=manager)
        decoder.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path={"default": str(official.MODEL)},
                                disable_mmap={"vae": True, "audio_vae": True})
    finally:
        official.release_weight_load_lock(lock)
    decoder.vae.to("cuda")
    decoder.audio_vae.to("cuda")
    return decoder, manager


def decode(decoder, arm, duration, sparse=False):
    import torch
    from diffusers.utils.export_utils import encode_video
    sys.path.insert(0, str(FV_EVAL))
    from fasth3_vbench_archive import archive
    destination = OUT / arm / f"{duration}s" / ("spark" if sparse else "dense")
    if arm == "baseline24":
        source = BASELINE / f"{duration}s/dense"
        meta = official.read(source / "meta.json")
        assert official.sha(meta["latent_path"]) == meta["latent_sha256"]
        original_protocol = official.read(Path("/autodl-fs/data/h3_experiments/ref2va_768_480_dense_spark_20261003/protocol.json"))
        original_runner = Path("/autodl-fs/data/h3_experiments/ref2va_768_480_dense_spark_20261003/runner_source.py")
        assert original_protocol["runner_sha256"] == official.sha(original_runner)
        assert original_protocol["seed"] == 42 and original_protocol["requested_steps"] == 20
        assert original_protocol["actual_evaluations"] == 19 and original_protocol["target_resolution"] == [1344, 768]
        assert meta["method"] == "dense" and meta["variant"] == "original768" and meta["duration"] == duration
        assert meta["video"]["frames"] == duration * 24 and (meta["video"]["width"], meta["video"]["height"]) == (1344, 768)
        assert meta["compile_wrapped_blocks"] == 50
        assert official.sha(meta["conditioning_cache"]) == meta["conditioning_sha256"]
        meta = dict(meta, status="generated", arm=arm, duration=duration,
                    source_meta_sha256=official.sha(source / "meta.json"),
                    historical_protocol=original_protocol,
                    provenance_note="Historical 24fps Dense runner hash and configuration verified; historical full core-source hashes unavailable")
    else:
        meta = official.read(destination / "meta.json")
    if meta.get("status") == "complete" and (destination / "video.mkv").exists():
        assert official.sha(destination / "video.mkv") == meta["video_sha256"]
        assert official.sha(destination / "audio.wav") == meta["audio_sha256"]
        return
    payload = torch.load(meta["latent_path"], map_location="cpu", weights_only=True)
    with torch.inference_mode():
        result = decoder(latents=payload["latents"].cuda(), audio_latents=payload["audio_latents"].cuda(),
                         output_type="np", output=["videos", "audio", "sampling_rate"])
    count, rate = duration * 24, int(result["sampling_rate"])
    full_frames = int(result["videos"][0].shape[0])
    frames = result["videos"][0][:count]
    rgb = np.clip(np.round(frames * 255), 0, 255).astype(np.uint8)
    audio = audio_numpy(result["audio"][0])[..., :duration * rate]
    assert rgb.shape == (count, 768, 1344, 3) and audio.shape[-1] == duration * rate
    destination.mkdir(parents=True, exist_ok=True)
    archive_meta = archive(rgb, audio, rate, destination / "video.mkv", fps=24, threads=8)
    write_audio(destination / "audio.wav", audio, rate)
    encode_video(frames, fps=24, output_path=str(destination / "video.mp4"),
                 audio=torch.from_numpy(audio), audio_sample_rate=rate)
    meta.update(status="complete", video_path=str(destination / "video.mkv"), video_sha256=official.sha(destination / "video.mkv"),
        preview_path=str(destination / "video.mp4"), audio_path=str(destination / "audio.wav"), audio_sha256=official.sha(destination / "audio.wav"),
        full_vae_decoded_frames=full_frames, metric_frames=count, decoded_crop_duration_seconds=count / 24,
        audio_samples=audio.shape[-1], audio_sampling_rate=rate, archive=archive_meta,
        decoded_rgb_sha256=tensor_sha(rgb), decoded_audio_sha256=tensor_sha(audio))
    meta["temporal_diagnostics"] = temporal_diagnostics(rgb, audio, rate, destination)
    write(destination / "meta.json", meta)
    print("DECODED", arm, duration, sparse, flush=True)


def temporal_diagnostics(rgb, audio, rate, destination):
    """Objective timestamp checks and review assets, not a semantic-sync score."""
    from PIL import Image, ImageDraw
    video = destination / "video.mkv"
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video)], text=True))
    streams = {s["codec_type"]: s for s in probe["streams"]}
    assert float(streams["video"].get("start_time", 0)) == 0
    assert float(streams["audio"].get("start_time", 0)) == 0
    assert int(streams["audio"]["sample_rate"]) == rate
    count = len(rgb)
    small = np.stack([np.asarray(Image.fromarray(frame).resize((168, 96)), dtype=np.float32) / 255 for frame in rgb])
    motion = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2, 3))
    boundaries = np.round(np.arange(count + 1) * rate / 24).astype(int)
    rms = [float(np.sqrt(np.mean(audio[..., boundaries[i]:boundaries[i + 1]].astype(np.float64) ** 2))) for i in range(count)]
    times = np.linspace(0, count - 1, 10).round().astype(int)
    sheet = Image.new("RGB", (5 * 336, 2 * 210), "white")
    draw = ImageDraw.Draw(sheet)
    for k, index in enumerate(times):
        x, y = k % 5 * 336, k // 5 * 210
        sheet.paste(Image.fromarray(rgb[index]).resize((336, 192)), (x, y))
        draw.text((x + 4, y + 194), f"t={index / 24:.3f}s / frame {index}", fill="black")
    sheet.save(destination / "temporal_contact_sheet.jpg")
    return dict(stream_probe=probe, video_duration_seconds=count / 24, audio_duration_seconds=audio.shape[-1] / rate,
        stream_start_times_zero=True, video_audio_lengths_match=math.isclose(count / 24, audio.shape[-1] / rate),
        motion_envelope=motion.tolist(), audio_rms_per_video_frame=rms,
        contact_sheet=str(destination / "temporal_contact_sheet.jpg"),
        interpretation="Envelope and PTS checks are diagnostics only; motion, identity preservation and semantic A/V sync need visual/audio review")


def worker(arm, sparse=False, duration=None):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    durations = (duration,) if duration else DURATIONS
    if arm != "baseline24":
        pipe, manager = load_denoiser()
        with official.compile_ref2va_transformer(pipe) as acceleration:
            assert official._impl_bootstrap.verify_transformer_compile(acceleration, pipe.transformer_ref, requested=True) == 50
            for length in durations:
                generate(pipe, arm, length, sparse, warmup=True)
                generate(pipe, arm, length, sparse)
        del pipe, manager, acceleration
        torch._dynamo.reset()
        official.release_cpu_arenas()
    decoder, manager = load_decoder()
    for length in durations:
        decode(decoder, arm, length, sparse)
    del decoder, manager
    official.release_cpu_arenas()


def spawn(jobs):
    processes = []
    for i, (arm, sparse, duration) in enumerate(jobs):
        gpu = GPU_IDS[i % len(GPU_IDS)]
        log = (ROOT / f"worker_{arm}_{'spark' if sparse else 'dense'}_{duration or 'both'}s.log").open("a")
        args = [sys.executable, str(Path(__file__)), "worker", arm, "1" if sparse else "0", str(duration or 0)]
        proc = subprocess.Popen(args, env={**os.environ, **official.ENV, "CUDA_VISIBLE_DEVICES": str(gpu),
            "H3_WEIGHT_LOAD_LOCK_SLOT": str(i % len(GPU_IDS))}, stdout=log, stderr=subprocess.STDOUT)
        processes.append((proc, log))
    pending = {p for p, _ in processes}
    failed = False
    while pending:
        finished = {p for p in pending if p.poll() is not None}
        if any(p.returncode != 0 for p in finished):
            failed = True
            for p in pending - finished:
                p.terminate()
        pending -= finished
        if pending:
            time.sleep(1)
    codes = [p.wait() for p, _ in processes]
    for _, log in processes:
        log.close()
    if failed or any(codes):
        raise RuntimeError(f"Ref2VA workers failed: {codes}")


def score(stage):
    comparisons = []
    for duration in DURATIONS:
        if stage == "dense":
            comparisons.extend((duration, a, b, False, False) for a, b in [("B", "A"), ("C", "A"),
                ("A", "baseline24"), ("B", "baseline24"), ("C", "baseline24")])
        else:
            comparisons.extend([(duration, "C", "C", True, False), (duration, "C", "baseline24", True, False)])
    refs, candidates = [], []
    for index, (duration, a, b, sparse_a, sparse_b) in enumerate(comparisons, 1):
        for arm, sparse, rows in [(a, sparse_a, candidates), (b, sparse_b, refs)]:
            path = OUT / arm / f"{duration}s" / ("spark" if sparse else "dense") / "meta.json"
            meta = official.read(path)
            rows.append(dict(index=index, output_path=meta["video_path"], sha256=meta["video_sha256"]))
    work = ROOT / "quality" / stage
    refpath, candpath = work / "reference_manifest.json", work / "candidate_manifest.json"
    write(refpath, dict(records=refs))
    write(candpath, dict(records=candidates))
    method = f"ref2va_{stage}"
    cfg = work / "config.json"
    write(cfg, dict(work_dir=str(work), reference_manifest=str(refpath), candidate_manifest=str(candpath),
                    method=method, cases=list(range(1, len(comparisons) + 1)), workers=len(GPU_IDS)))
    subprocess.run([sys.executable, str(QUALITY_SCRIPT), "run", "--config", str(cfg)], check=True,
        env={**os.environ, **official.ENV, "CUDA_VISIBLE_DEVICES": ",".join(map(str, GPU_IDS)), "H3_NUM_GPUS": str(len(GPU_IDS))})
    records = []
    for index, (duration, a, b, sparse_a, sparse_b) in enumerate(comparisons, 1):
        quality = official.read(work / f"{method}_{index:02}.json")
        apath = OUT / a / f"{duration}s" / ("spark" if sparse_a else "dense")
        bpath = OUT / b / f"{duration}s" / ("spark" if sparse_b else "dense")
        aa, rate_a = read_audio(apath / "audio.wav")
        ab, rate_b = read_audio(bpath / "audio.wav")
        assert aa.shape == ab.shape and rate_a == rate_b
        audio = dict(sample_rate=rate_a, samples=len(aa), duration_seconds=len(aa) / rate_a,
                     waveform_mse=float(np.mean((aa.astype(np.float64) - ab) ** 2)),
                     waveform_correlation=float(np.corrcoef(aa.ravel(), ab.ravel())[0, 1]),
                     note="Waveform fidelity and matched stream length do not prove semantic audiovisual synchronization")
        records.append(dict(duration=duration, candidate=f"{a}_{'spark' if sparse_a else 'dense'}",
            reference=f"{b}_{'spark' if sparse_b else 'dense'}", quality=quality, audio=audio))
    write(ROOT / f"{stage}_comparison.json", dict(status="complete", records=records,
        metric_definition="Lossless RGB decoded from saved latents; pooled PSNR, Gaussian valid-window SSIM, LPIPS AlexNet v0.1",
        interpretation="Fidelity to reference, not a claim of semantic quality improvement"))
    return records


def check_pairing():
    for duration in DURATIONS:
        a = official.read(OUT / "A" / f"{duration}s/dense/meta.json")
        b = official.read(OUT / "B" / f"{duration}s/dense/meta.json")
        for key in ("conditioning_sha256", "frames_sha256", "prompt_embeds_sha256", "initial_noise_sha256"):
            assert a[key] == b[key], (duration, key)
    write(ROOT / "A_B_pairing_audit.json", dict(status="passed", conditioning_identical=True,
          reference_frames_identical=True, text_embeddings_identical=True, actual_initial_noise_identical=True))


def main():
    os.environ.update(official.ENV)
    configure_load_lock()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "accept_runner_repair":
        accept_runner_repair()
        return
    if command == "prepare":
        prepare()
        return
    if command == "worker":
        worker(sys.argv[2], bool(int(sys.argv[3])), int(sys.argv[4]) or None)
        return
    if command == "coordinate_audit":
        ROOT.mkdir(parents=True, exist_ok=True)
        coordinate_audit()
        return
    if command != "run":
        raise ValueError(command)
    for duration in DURATIONS:
        route = Path("/autodl-fs/data/h3_experiments") / f"route_2x2_full50_4gpu_20261004_{duration}s768p/results.json"
        assert official.read(route).get("full50", {}).get("status") == "complete", "finish task1 first"
    # A separate process guarantees conditioning models/allocator memory are
    # gone before GPU0's denoiser starts, even if third-party managers retain a
    # reference cycle. No numerical or conditioning setting changes here.
    subprocess.run([sys.executable, str(Path(__file__)), "prepare"], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU_IDS[0]), "H3_WEIGHT_LOAD_LOCK_SLOT": "0"})
    spawn([(arm, False, None) for arm in ARMS] + [("baseline24", False, None)])
    check_pairing()
    dense = score("dense")
    write(ROOT / "status.json", dict(status="running", stage="C_spark"))
    spawn([("C", True, duration) for duration in DURATIONS])
    spark = score("spark")
    write(ROOT / "results.json", dict(status="complete", dense_comparisons=dense, spark_comparisons=spark,
        coordinates=official.read(ROOT / "coordinate_audit.json"), pairing=official.read(ROOT / "A_B_pairing_audit.json")))
    protocol = official.read(ROOT / "protocol.json")
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    write(ROOT / "status.json", dict(status="complete", stage="complete"))
    print("REF2VA PHASE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
