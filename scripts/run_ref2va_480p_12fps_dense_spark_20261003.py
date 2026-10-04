#!/usr/bin/env python3
"""Ref2VA 5s ablation: 480p/12fps reference, Dense vs target+condition Spark-H3 10%."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
sys.path.insert(0, str(BENCH / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import ref2va_official_case as official  # noqa: E402
from run_ref2va_condition_fps_rope_20261003 import rope_patch, valid_frames  # noqa: E402
from run_ref2va_video_scope_20261003 import spark_config  # noqa: E402


NAME = "ref2va_480p_12fps_dense_spark_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
FPS = 12
DURATION = 5
METHODS = ("dense", "spark_target_condition_video_10pct")


def write(path, value):
    official.write(path, value)


@contextlib.contextmanager
def conditioning_patch():
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    original_normalize = MiniMaxH3Ref2VASetupStep._normalize_video_condition
    original_gather = MiniMaxH3Ref2VATextEncoderStep._gather_vision_features

    def normalize(frames, source_fps, num_frames, canvas_multiple, canvas_short_edge,
                  canvas_max_pixels, target_fps):
        requested = round(num_frames * FPS / official.FPS)
        result = original_normalize(
            frames, source_fps, requested, canvas_multiple, 480, 480 * 832, FPS
        )
        return result[:valid_frames(len(result))]

    def gather(self, processor, references, ignored_fps):
        return original_gather(self, processor, references, FPS)

    MiniMaxH3Ref2VASetupStep._normalize_video_condition = staticmethod(normalize)
    MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = gather
    try:
        yield
    finally:
        MiniMaxH3Ref2VASetupStep._normalize_video_condition = staticmethod(original_normalize)
        MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = original_gather


def prepare_conditioning():
    official.CONDITIONING_CACHE_SCHEMA = 212
    official.CACHE_DIR = OUT / "conditioning_480p_12fps"
    official.CASE_DIR = ROOT / "conditioning"
    official.DURATIONS = (DURATION,)
    with conditioning_patch():
        official.condition()
    state, path, summary = official.load_conditioning(DURATION)
    ref = state.values["normalized_references"][0]
    latent = state.values["condition_latents"][0]
    assert tuple(ref.frames.shape) == (56, 480, 832, 3), ref.frames.shape
    assert tuple(latent.shape) == (1, 24, 17, 30, 52), latent.shape
    observation = {
        "conditioning_cache": str(path), "conditioning_sha256": official.sha(path),
        "normalized_reference_shape": list(ref.frames.shape), "condition_latent_shape": list(latent.shape),
        "visual_llm_fps": FPS, "reference_vae_fps": FPS, "summary": summary,
    }
    write(ROOT / "conditioning_observation.json", observation)
    return state, path, summary, observation


def run_denoise(pipe, state, method, steps):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    prepared = official.clone_state(state)
    prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
    sparse = method != "dense"
    plugin_context = install_h3_sparse_attention(pipe.transformer_ref, spark_config(steps)) if sparse else contextlib.nullcontext(None)
    with rope_patch(FPS, "physical_time"), plugin_context as plugin, torch.inference_mode():
        if plugin is not None:
            plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=prepared, num_frames=state.values["num_frames"], height=768, width=1344,
            num_inference_steps=steps, generator=torch.Generator(device="cpu").manual_seed(42),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary() if plugin is not None else None
    assert all(torch.isfinite(result[key]).all().item() for key in ("latents", "audio_latents"))
    if summary is not None:
        expected_dense = 4 if steps == 20 else 1
        assert summary["completed_evaluations"] == steps - 1
        assert summary["dense_evaluations"] == expected_dense
        assert summary["sol_sparse_video_scope"] == "target_and_condition"
        assert summary["condition_video_tokens"] > 0
        assert summary["sparse_video_tokens"] == summary["target_video_tokens"] + summary["condition_video_tokens"]
    return result, seconds, summary


def inspect_video(path):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (int(video["width"]), int(video["height"])) == (1344, 768)
    assert int(video["nb_frames"]) == 120
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    return {"width": 1344, "height": 768, "frames": 120, "duration": float(probe["format"]["duration"])}


def main():
    import torch
    from diffusers import ComponentsManager
    from diffusers.utils.export_utils import encode_video

    # A failed preflight/warmup may leave only protocol and conditioning-cache
    # artifacts.  Resume that exact run, but never overwrite completed results.
    if (ROOT / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running", "name": NAME, "duration": DURATION, "conditioning_resolution": [832, 480],
        "conditioning_fps": FPS, "rope": "physical_time (condition offsets x2)",
        "target_resolution": [1344, 768], "target_fps": 24, "steps": 20, "evaluations": 19, "seed": 42,
        "methods": METHODS,
        "spark": {"topk_ratio": 0.10, "sparse_video_scope": "target_and_condition", "dense_evaluations": 4, "dense_layers": 1},
        "runner_sha256": official.sha(Path(__file__).resolve()),
    }
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")
    state, cache_path, conditioning_summary, observation = prepare_conditioning()
    official.release_cpu_arenas()

    workflow = official.get_workflow()
    for name in ("before_encode", "text_encoder", "vae_encoder", "decode.video", "decode.audio"):
        workflow.sub_blocks.pop(name)
    lock = official.acquire_weight_load_lock()
    manager = ComponentsManager()
    try:
        pipe = workflow.init_pipeline(str(official.MODEL), components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path={"default": str(official.MODEL)},
                             disable_mmap={"transformer_ref": True})
    finally:
        official.release_weight_load_lock(lock)
    pipe.transformer_ref.to("cuda")
    torch.cuda.synchronize()
    official.release_cpu_arenas()

    records, payloads = {}, {}
    with official.compile_ref2va_transformer(pipe) as acceleration:
        wrapped = official._impl_bootstrap.verify_transformer_compile(acceleration, pipe.transformer_ref, requested=True)
        assert wrapped == 50
        warm, warm_seconds, warm_summary = run_denoise(pipe, state, "spark_target_condition_video_10pct", 3)
        del warm
        for method in METHODS:
            result, seconds, summary = run_denoise(pipe, state, method, 20)
            payload = {key: result[key].detach().cpu().contiguous() for key in ("latents", "audio_latents")}
            del result
            destination = OUT / method
            latent_path = destination / "latents.pt"
            official.atomic_torch_save(payload, latent_path)
            payloads[method] = payload
            records[method] = {
                "method": method, "denoise_seconds": seconds, "attention_summary": summary,
                "latent_path": str(latent_path), "latent_sha256": official.sha(latent_path),
                "conditioning_cache": str(cache_path), "conditioning_sha256": official.sha(cache_path),
                "conditioning_summary": conditioning_summary, "conditioning_observation": observation,
                "compile_wrapped_blocks": wrapped, "warmup_seconds": warm_seconds, "warmup_summary": warm_summary,
            }
            print("DENOISED", method, round(seconds, 3), flush=True)
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
        decoder.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path={"default": str(official.MODEL)},
                                disable_mmap={"vae": True, "audio_vae": True})
    finally:
        official.release_weight_load_lock(lock)
    decoder.vae.to("cuda")
    decoder.audio_vae.to("cuda")
    for method, payload in payloads.items():
        with torch.inference_mode():
            decoded = decoder(latents=payload["latents"].cuda(), audio_latents=payload["audio_latents"].cuda(),
                              output_type="np", output=["videos", "audio", "sampling_rate"])
        frames = decoded["videos"][0][:120]
        rate = decoded["sampling_rate"]
        audio = decoded["audio"][0][..., :round(5 * rate)]
        destination = OUT / method
        video_path = destination / "video.mp4"
        partial = video_path.with_name(f".{video_path.stem}.partial-{os.getpid()}.mp4")
        encode_video(frames, fps=24, output_path=str(partial), audio=audio, audio_sample_rate=rate)
        partial.replace(video_path)
        records[method].update(video_path=str(video_path), video_sha256=official.sha(video_path), video=inspect_video(video_path))
        write(destination / "meta.json", records[method])
        print("DECODED", method, flush=True)
    write(ROOT / "results.json", {"status": "complete", "records": records})
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
