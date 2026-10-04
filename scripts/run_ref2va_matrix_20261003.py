#!/usr/bin/env python3
"""Four-GPU Ref2VA 768/480-reference x 5/10-second Dense/Spark matrix."""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
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
import ref2va_official_case as official  # noqa: E402


NAME = "ref2va_768_480_dense_spark_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
CACHE_480 = OUT / "conditioning_480"
ORIGINAL_CACHE = official.CACHE_DIR
JOBS = (("original768", 5), ("original768", 10), ("reference480", 5), ("reference480", 10))
GPUS = (0, 1, 2, 3)


def write(path, value):
    official.write(path, value)


def sha(path):
    return official.sha(path)


def configure_variant(variant):
    if variant == "original768":
        official.CONDITIONING_CACHE_SCHEMA = 3
        official.CACHE_DIR = ORIGINAL_CACHE
    elif variant == "reference480":
        official.CONDITIONING_CACHE_SCHEMA = 4
        official.CACHE_DIR = CACHE_480
    else:
        raise ValueError(variant)


def assert_conditioning_geometry(state, variant, duration):
    values = state.values
    ref = values["normalized_references"][0]
    frames = tuple(ref.frames.shape)
    latents = [tuple(value.shape) for value in values["condition_latents"]]
    if variant == "reference480":
        assert frames[1:3] == (480, 832), frames
        assert latents[0][-2:] == (30, 52), latents
    else:
        assert frames[1:3] == (768, 1344), frames
        assert latents[0][-2:] == (48, 84), latents
    assert values["height"] == 768 and values["width"] == 1344
    assert values["num_frames"] == official.aligned_frames(official.REQUESTED_FRAMES[duration])
    return {"normalized_video_shape": list(frames), "condition_latent_shapes": [list(x) for x in latents]}


def condition_480():
    """Populate a separate cache while proving both visual consumers see 480p."""
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VATextEncoderStep,
    )

    configure_variant("reference480")
    original_normalize = MiniMaxH3Ref2VASetupStep._normalize_video_condition
    original_gather = MiniMaxH3Ref2VATextEncoderStep._gather_vision_features
    original_reference_call = MiniMaxH3Ref2VAReferenceEncoderStep.__call__
    observed = {"visual_llm": [], "reference_vae": []}

    def normalize(frames, fps, num_frames, canvas_multiple, canvas_short_edge,
                  canvas_max_pixels, target_fps):
        return original_normalize(
            frames, fps, num_frames, canvas_multiple, 480, 480 * 832, target_fps
        )

    def gather(self, processor, references, fps):
        shape = tuple(next(ref for ref in references if ref.kind == "video").frames.shape)
        assert shape[1:3] == (480, 832), shape
        observed["visual_llm"].append(list(shape))
        return original_gather(self, processor, references, fps)

    def reference_call(self, components, state):
        block_state = self.get_block_state(state)
        shape = tuple(next(ref for ref in block_state.normalized_references
                           if ref.kind == "video").frames.shape)
        assert shape[1:3] == (480, 832), shape
        observed["reference_vae"].append(list(shape))
        return original_reference_call(self, components, state)

    MiniMaxH3Ref2VASetupStep._normalize_video_condition = staticmethod(normalize)
    MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = gather
    MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = reference_call
    old_case_dir = official.CASE_DIR
    official.CASE_DIR = ROOT / "conditioning_480_case"
    try:
        official.condition()
    finally:
        official.CASE_DIR = old_case_dir
        MiniMaxH3Ref2VASetupStep._normalize_video_condition = staticmethod(original_normalize)
        MiniMaxH3Ref2VATextEncoderStep._gather_vision_features = original_gather
        MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = original_reference_call
    assert len(observed["visual_llm"]) >= 2, observed
    assert len(observed["reference_vae"]) >= 2, observed
    geometry = {}
    for duration in official.DURATIONS:
        state, path, summary = official.load_conditioning(duration)
        geometry[f"{duration}s"] = {
            "cache_path": str(path), "cache_sha256": sha(path),
            **assert_conditioning_geometry(state, "reference480", duration), **summary,
        }
    write(ROOT / "conditioning_480_observations.json", {
        "status": "passed", "observed_inputs": observed, "geometry": geometry,
    })


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for variant, duration in JOBS:
        for method in ("dense", "spark_10pct"):
            (OUT / variant / f"{duration}s" / method).mkdir(parents=True)
    configure_variant("original768")
    original = {}
    for duration in official.DURATIONS:
        state, path, summary = official.load_conditioning(duration)
        original[f"{duration}s"] = {
            "cache_path": str(path), "cache_sha256": sha(path),
            **assert_conditioning_geometry(state, "original768", duration), **summary,
        }
    write(ROOT / "conditioning_original_observations.json", {"status": "passed", "geometry": original})
    condition_480()
    # Conditioning is built in the launcher process.  Drop all CUDA allocator
    # cache before spawning GPU0's denoiser; otherwise the parent can retain
    # tens of GiB even though the conditioning pipeline has been destroyed.
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass
    protocol = {
        "status": "running", "name": NAME, "jobs": JOBS, "gpus": GPUS,
        "target_resolution": [1344, 768], "requested_steps": 20,
        "actual_evaluations": 19, "seed": 42,
        "spark": {"topk_ratio": 0.10, "schedule_dense_evaluations": 4, "dense_layers": 1},
        "runtime_warmup": "excluded 3-step Spark pass: one dense + one sparse evaluation",
        "reference_variants": {
            "original768": "normalized reference 768x1344",
            "reference480": "normalized reference 480x832; target remains 768x1344",
        },
        "runner_sha256": sha(Path(__file__).resolve()),
    }
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def spark_config(steps):
    from h3_sparse_attention import H3SparseAttentionConfig
    return H3SparseAttentionConfig.spark(
        steps, warmup_percent=20.0 if steps == 20 else 33.0,
        sol_dense_layers=1, sol_route_topk_ratio=0.10, sol_log_density=False,
    )


def run_denoise(pipe, state, *, steps, plugin_config=None):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention
    prepared = official.clone_state(state)
    prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
    context = (contextlib.nullcontext(None) if plugin_config is None else
               install_h3_sparse_attention(pipe.transformer_ref, plugin_config))
    with context as plugin, torch.inference_mode():
        if plugin is not None:
            plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=prepared, num_frames=state.values["num_frames"], height=768, width=1344,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = None if plugin is None else plugin.summary()
    assert all(torch.isfinite(result[key]).all().item() for key in ("latents", "audio_latents"))
    if summary is not None:
        assert summary["completed_evaluations"] == steps - 1, summary
        assert summary["dense_evaluations"] == (4 if steps == 20 else 1), summary
    del prepared
    return result, seconds, summary


def inspect_video(path, expected_frames):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True))
    videos = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    audios = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
    assert len(videos) == 1 and audios, probe
    video = videos[0]
    assert int(video["width"]) == 1344 and int(video["height"]) == 768, video
    if video.get("nb_frames") not in (None, "N/A"):
        assert int(video["nb_frames"]) == expected_frames, video
    duration = float(probe["format"]["duration"])
    assert abs(duration - expected_frames / 24) < 0.15, duration
    return {"width": 1344, "height": 768, "frames": expected_frames,
            "duration": duration, "audio_streams": len(audios)}


def worker(rank):
    import torch
    from diffusers import ComponentsManager
    from diffusers.utils.export_utils import encode_video

    rank = int(rank)
    variant, duration = JOBS[rank]
    configure_variant(variant)
    state, cache_path, conditioning_summary = official.load_conditioning(duration)
    geometry = assert_conditioning_geometry(state, variant, duration)
    workflow = official.get_workflow()
    for name in ("before_encode", "text_encoder", "vae_encoder", "decode.video", "decode.audio"):
        workflow.sub_blocks.pop(name)
    lock = official.acquire_weight_load_lock()
    manager = ComponentsManager()
    try:
        pipe = workflow.init_pipeline(str(official.MODEL), components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16,
                             pretrained_model_name_or_path={"default": str(official.MODEL)},
                             disable_mmap={"transformer_ref": True})
    finally:
        official.release_weight_load_lock(lock)
    pipe.transformer_ref.to("cuda")
    torch.cuda.synchronize()
    official.release_cpu_arenas()
    payloads = {}
    records = {}
    with official.compile_ref2va_transformer(pipe) as acceleration:
        wrapped = official._impl_bootstrap.verify_transformer_compile(
            acceleration, pipe.transformer_ref, requested=True)
        assert wrapped == 50, wrapped
        warm, warm_seconds, warm_summary = run_denoise(
            pipe, state, steps=3, plugin_config=spark_config(3))
        del warm
        order = ("dense", "spark_10pct") if rank % 2 == 0 else ("spark_10pct", "dense")
        for method in order:
            cfg = None if method == "dense" else spark_config(20)
            result, seconds, summary = run_denoise(pipe, state, steps=20, plugin_config=cfg)
            payload = {key: result[key].detach().cpu().contiguous()
                       for key in ("latents", "audio_latents")}
            del result
            latent = OUT / variant / f"{duration}s" / method / "latents.pt"
            official.atomic_torch_save(payload, latent)
            payloads[method] = payload
            records[method] = {
                "method": method, "variant": variant, "duration": duration,
                "denoise_seconds": seconds, "attention_summary": summary,
                "latent_path": str(latent), "latent_sha256": sha(latent),
                "conditioning_cache": str(cache_path),
                "conditioning_sha256": sha(cache_path), "conditioning_summary": conditioning_summary,
                "conditioning_geometry": geometry, "compile_wrapped_blocks": wrapped,
                "warmup_seconds": warm_seconds, "warmup_summary": warm_summary,
            }
            print("DENOISED", variant, duration, method, round(seconds, 3), flush=True)
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
        decoder.load_components(dtype=torch.bfloat16,
                                pretrained_model_name_or_path={"default": str(official.MODEL)},
                                disable_mmap={"vae": True, "audio_vae": True})
    finally:
        official.release_weight_load_lock(lock)
    decoder.vae.to("cuda")
    decoder.audio_vae.to("cuda")
    requested = official.REQUESTED_FRAMES[duration]
    for method, payload in payloads.items():
        with torch.inference_mode():
            decoded = decoder(latents=payload["latents"].cuda(),
                              audio_latents=payload["audio_latents"].cuda(),
                              output_type="np", output=["videos", "audio", "sampling_rate"])
        video_frames = decoded["videos"][0][:requested]
        assert video_frames.shape == (requested, 768, 1344, 3)
        assert np.isfinite(video_frames).all()
        rate = decoded["sampling_rate"]
        audio = decoded["audio"][0][..., :round(requested / 24 * rate)]
        video = OUT / variant / f"{duration}s" / method / "video.mp4"
        temporary = video.with_name(f".{video.stem}.partial-{os.getpid()}{video.suffix}")
        encode_video(video_frames, fps=24, output_path=str(temporary), audio=audio,
                     audio_sample_rate=rate)
        temporary.replace(video)
        records[method].update(video_path=str(video), video_sha256=sha(video),
                               video=inspect_video(video, requested))
        write(OUT / variant / f"{duration}s" / method / "meta.json", records[method])
        print("DECODED", variant, duration, method, flush=True)
    write(ROOT / f"worker_gpu{rank}.json", {"status": "complete", "job": [variant, duration]})


def summarize():
    records = {}
    for variant, duration in JOBS:
        for method in ("dense", "spark_10pct"):
            meta = OUT / variant / f"{duration}s" / method / "meta.json"
            records[f"{variant}_{duration}s_{method}"] = json.loads(meta.read_text())
    result = {"status": "complete", "outputs": len(records), "records": records}
    write(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    print(json.dumps({"status": "complete", "outputs": len(records)}, indent=2), flush=True)


def launch():
    if not ROOT.exists():
        prepare()
    else:
        # Resume only after a fully completed prepare stage.  Worker outputs
        # remain independently atomic, so a pre-denoise restart is safe.
        required = (ROOT / "protocol.json", ROOT / "conditioning_480_observations.json",
                    ROOT / "conditioning_original_observations.json")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"incomplete existing Ref2VA prepare stage: {missing}")
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **official.ENV, **official.SPARK_ENV,
                 "CUDA_VISIBLE_DEVICES": str(gpu), "H3_IMPL_REPO": str(REPO)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((proc, log))
    codes = [proc.wait() for proc, _ in jobs]
    for _, log in jobs:
        log.close()
    if any(codes):
        raise RuntimeError(f"Ref2VA workers failed: {codes}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(sys.argv[2])
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
