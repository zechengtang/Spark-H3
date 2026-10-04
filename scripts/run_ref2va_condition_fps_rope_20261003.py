#!/usr/bin/env python3
"""Ref2VA 5s/10s dense ablation: 12/8fps reference video, legacy vs physical-time RoPE."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import re

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
sys.path.insert(0, str(BENCH / "scripts"))
import ref2va_official_case as official  # noqa: E402

NAME = "ref2va_condition_fps_rope_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
FPS_VALUES = tuple(int(x) for x in os.environ.get("H3_CONDITION_FPS", "12,8").split(",") if x)
DURATIONS = tuple(int(x) for x in os.environ.get("H3_DURATIONS", "5,10").split(",") if x)
ROPE_MODES = ("legacy", "physical_time")


def write(path, value):
    official.write(path, value)


def valid_frames(n):
    return max(5, max(1, (n - 5) // 17) * 17 + 5)


def configure(fps):
    official.CONDITIONING_CACHE_SCHEMA = 100 + fps
    official.CACHE_DIR = OUT / f"conditioning_{fps}fps"
    official.CASE_DIR = ROOT / f"conditioning_{fps}fps"


@contextlib.contextmanager
def conditioning_fps_patch(fps):
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    original_normalize = MiniMaxH3Ref2VASetupStep._normalize_video_condition
    original_gather = MiniMaxH3Ref2VATextEncoderStep._gather_vision_features

    def normalize(frames, source_fps, num_frames, canvas_multiple, canvas_short_edge,
                  canvas_max_pixels, target_fps):
        requested = round(num_frames * fps / official.FPS)
        result = original_normalize(
            frames, source_fps, requested, canvas_multiple, canvas_short_edge, canvas_max_pixels, fps
        )
        # Give the visual LLM exactly the same temporal extent that the reference VAE can encode.
        return result[:valid_frames(len(result))]

    def gather(self, processor, references, ignored_fps):
        return original_gather(self, processor, references, fps)

    MiniMaxH3Ref2VASetupStep._normalize_video_condition = staticmethod(normalize)
    MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = gather
    try:
        yield
    finally:
        MiniMaxH3Ref2VASetupStep._normalize_video_condition = staticmethod(original_normalize)
        MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = original_gather


@contextlib.contextmanager
def rope_patch(fps, mode):
    if mode == "legacy":
        yield
        return
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep

    original = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
    ratio = official.FPS / fps

    def corrected(*args, **kwargs):
        result = list(original(*args, **kwargs))
        references = args[1]
        condition_latents = args[2]
        audio_condition_latents = args[3]
        audio_channels = args[9]
        if len(references) != 2 or references[0].kind != "video" or references[1].kind != "audio":
            raise RuntimeError("physical-time patch is scoped to the official [video, audio] Ref2VA case")
        position_ids, video_indices = result[0], result[2]
        num_condition_video_rows = result[5]
        condition_video_indices = video_indices[:num_condition_video_rows]
        frames = condition_latents[0].shape[2]
        _, patch_h, patch_w = args[8]
        spatial_rows = (condition_latents[0].shape[3] // patch_h) * (condition_latents[0].shape[4] // patch_w)
        assert condition_video_indices.numel() == frames * spatial_rows
        origin = position_ids[condition_video_indices[0], 0].clone()
        position_ids[condition_video_indices, 0] = (
            origin + (position_ids[condition_video_indices, 0] - origin) * ratio
        )
        pattern = (1, 4, 4, 4, 4)
        video_span = sum((5.0 / 3.0) * pattern[i % len(pattern)] for i in range(frames))
        video_audio_span = audio_condition_latents[0].shape[0] / audio_channels
        delta = max(video_audio_span, video_span * ratio) - max(video_audio_span, video_span)
        after_video = int(condition_video_indices[-1]) + 1
        position_ids[after_video:, 0] += delta
        result[0] = position_ids
        return tuple(result)

    MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(corrected)
    try:
        yield
    finally:
        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(original)


def prepare_conditioning():
    observations = {}
    for fps in FPS_VALUES:
        configure(fps)
        with conditioning_fps_patch(fps):
            official.condition()
        for duration in DURATIONS:
            state, path, summary = official.load_conditioning(duration)
            ref = state.values["normalized_references"][0]
            latent = state.values["condition_latents"][0]
            # Match the setup step's round-half-up end slot (Python's round is banker's rounding).
            source_slots = int(np.floor(145 * fps / 24 + 0.5))
            expected_max = min(round(state.values["num_frames"] * fps / 24), source_slots)
            assert ref.frames.shape[0] == valid_frames(expected_max), (fps, duration, ref.frames.shape)
            assert latent.shape[2] == 5 * ((ref.frames.shape[0] - 5) // 17) + 2
            observations[f"{fps}fps_{duration}s"] = {
                "conditioning_cache": str(path),
                "cache_sha256": official.sha(path),
                "normalized_reference_shape": list(ref.frames.shape),
                "condition_latent_shape": list(latent.shape),
                "visual_llm_fps": fps,
                "reference_vae_fps": fps,
                "summary": summary,
            }
    write(ROOT / "conditioning_observations.json", observations)


def run_denoise(pipe, state, fps, mode, steps):
    import torch
    prepared = official.clone_state(state)
    prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
    with rope_patch(fps, mode), torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=prepared,
            num_frames=state.values["num_frames"],
            height=official.HEIGHT,
            width=official.WIDTH,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(official.SEED),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    return result, seconds


def inspect_video(path, expected_frames):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert int(video["width"]) == 1344 and int(video["height"]) == 768
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    if video.get("nb_frames") not in (None, "N/A"):
        assert int(video["nb_frames"]) == expected_frames
    return probe["format"]["duration"]


def main():
    import torch
    from diffusers import ComponentsManager
    from diffusers.utils.export_utils import encode_video

    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running",
        "source_video": str(official.VIDEO_REFERENCE),
        "conditioning_fps": FPS_VALUES,
        "target_durations": DURATIONS,
        "target_fps": 24,
        "target_resolution": [1344, 768],
        "steps": 20,
        "evaluations": 19,
        "seed": 42,
        "attention": "dense",
        "rope_modes": {
            "legacy": "H3 fixed 5/3 temporal units per source-frame schedule",
            "physical_time": "condition-video temporal offsets scaled by 24/input_fps; downstream clock shifted",
        },
        "runner_sha256": official.sha(Path(__file__).resolve()),
    }
    write(ROOT / "protocol.json", protocol)
    prepare_conditioning()
    official.release_cpu_arenas()

    states = {}
    for fps in FPS_VALUES:
        configure(fps)
        for duration in DURATIONS:
            states[(fps, duration)] = official.load_conditioning(duration)[0]

    workflow = official.get_workflow()
    for name in ("before_encode", "text_encoder", "vae_encoder", "decode.video", "decode.audio"):
        workflow.sub_blocks.pop(name)
    lock = official.acquire_weight_load_lock()
    manager = ComponentsManager()
    try:
        pipe = workflow.init_pipeline(str(official.MODEL), components_manager=manager)
        pipe.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path={"default": str(official.MODEL)},
            disable_mmap={"transformer_ref": True},
        )
    finally:
        official.release_weight_load_lock(lock)
    pipe.transformer_ref.to("cuda")
    torch.cuda.synchronize()
    official.release_cpu_arenas()

    records, payloads = {}, {}
    with official.compile_ref2va_transformer(pipe) as acceleration:
        wrapped = official._impl_bootstrap.verify_transformer_compile(acceleration, pipe.transformer_ref, requested=True)
        assert wrapped == 50
        for fps in FPS_VALUES:
            for duration in DURATIONS:
                state = states[(fps, duration)]
                warm, warm_seconds = run_denoise(pipe, state, fps, "legacy", 3)
                del warm
                for mode in ROPE_MODES:
                    latent_path = OUT / f"{fps}fps" / f"{duration}s" / mode / "latents.pt"
                    key = f"{fps}fps_{duration}s_{mode}"
                    if latent_path.is_file():
                        payload = torch.load(latent_path, map_location="cpu", weights_only=True)
                        log_text = (ROOT / "run.log").read_text(errors="replace")
                        matches = re.findall(rf"DENOISED {re.escape(key)} ([0-9.]+)s", log_text)
                        seconds = float(matches[-1]) if matches else None
                        payloads[key] = payload
                        records[key] = {
                            "conditioning_fps": fps, "duration": duration, "rope_mode": mode,
                            "denoise_seconds": seconds, "warmup_seconds": warm_seconds,
                            "latent_path": str(latent_path), "latent_sha256": official.sha(latent_path),
                            "condition_frames": state.values["normalized_references"][0].frames.shape[0],
                            "condition_latent_frames": state.values["condition_latents"][0].shape[2],
                            "compile_wrapped_blocks": wrapped, "resumed_existing": True,
                        }
                        print(f"RESUMED {key}", flush=True)
                        continue
                    result, seconds = run_denoise(pipe, state, fps, mode, 20)
                    payload = {k: result[k].detach().cpu().contiguous() for k in ("latents", "audio_latents")}
                    del result
                    official.atomic_torch_save(payload, latent_path)
                    payloads[key] = payload
                    records[key] = {
                        "conditioning_fps": fps,
                        "duration": duration,
                        "rope_mode": mode,
                        "denoise_seconds": seconds,
                        "warmup_seconds": warm_seconds,
                        "latent_path": str(latent_path),
                        "latent_sha256": official.sha(latent_path),
                        "condition_frames": states[(fps, duration)].values["normalized_references"][0].frames.shape[0],
                        "condition_latent_frames": states[(fps, duration)].values["condition_latents"][0].shape[2],
                        "compile_wrapped_blocks": wrapped,
                    }
                    print(f"DENOISED {key} {seconds:.3f}s", flush=True)
    del pipe, manager
    official.release_cpu_arenas()

    workflow = official.get_workflow()
    for name in list(workflow.sub_blocks):
        if not name.startswith("decode."):
            workflow.sub_blocks.pop(name)
    lock = official.acquire_weight_load_lock()
    manager = ComponentsManager()
    try:
        decoder = workflow.init_pipeline(str(official.MODEL), components_manager=manager)
        decoder.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path={"default": str(official.MODEL)},
            disable_mmap={"vae": True, "audio_vae": True},
        )
    finally:
        official.release_weight_load_lock(lock)
    decoder.vae.to("cuda")
    decoder.audio_vae.to("cuda")
    for key, payload in payloads.items():
        fps = records[key]["conditioning_fps"]
        duration = records[key]["duration"]
        requested = official.REQUESTED_FRAMES[duration]
        with torch.inference_mode():
            decoded = decoder(
                latents=payload["latents"].cuda(), audio_latents=payload["audio_latents"].cuda(),
                output_type="np", output=["videos", "audio", "sampling_rate"],
            )
        frames = decoded["videos"][0][:requested]
        rate = decoded["sampling_rate"]
        audio = decoded["audio"][0][..., :round(requested / 24 * rate)]
        video_path = Path(records[key]["latent_path"]).with_name("video.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = video_path.with_name(f".{video_path.stem}.partial-{os.getpid()}.mp4")
        encode_video(frames, fps=24, output_path=str(temporary), audio=audio, audio_sample_rate=rate)
        temporary.replace(video_path)
        records[key].update(
            video_path=str(video_path), video_sha256=official.sha(video_path),
            encoded_duration_seconds=float(inspect_video(video_path, requested)),
        )
        write(video_path.with_name("meta.json"), records[key])
        print(f"DECODED {key}", flush=True)

    result = {"status": "complete", "records": records}
    write(ROOT / "results.json", result)
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
