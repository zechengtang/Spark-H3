#!/usr/bin/env python3
"""Generate the requested giraffe T2VA prompt with dense, Sol-Attn, and Spark-H3.

GPU 0/1/2 run the three denoisers. GPU 3 first builds the shared text
conditioning and then decodes each completed latent with one resident decoder.
The implementation deliberately reuses MiniMax-H3-Benchmark's split T2VA
pipeline and weight-load lock.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
sys.path.insert(0, str(BENCH / "scripts"))
import minimax_h3_vbench_4gpu_pipeline as pipeline  # noqa: E402
import h3_sparse_attention as h3  # noqa: E402

MODEL = Path("/autodl-fs/data/models/MiniMax-H3")
NAME = "giraffe_768p10s_seed42_three_variants_20260923"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHODS = ("dense", "sol-attn", "spark-h3-10pct-warmup")
GPU_FOR_METHOD = dict(zip(METHODS, (0, 1, 2), strict=True))
HEIGHT, WIDTH, FRAMES, FPS, STEPS, SEED = 768, 1344, 240, 24, 20, 42
PROMPT = """integrated_multimodal_description: [Shot 1] A majestic giraffe, its long neck gracefully arching, bends down to drink from a serene river, surrounded by lush greenery and tall grasses. The sun casts a golden glow, highlighting the giraffe's patterned coat and the gentle ripples in the water. Nearby, a family of zebras grazes peacefully, adding to the tranquil scene. The giraffe's delicate movements create a sense of harmony with nature, as the river flows gently, reflecting the vibrant colors of the surrounding landscape.

overall_soundscape: Natural ambient sounds and physical action sounds match the depicted environment and visible motion. Audible events remain temporally synchronized with their visible sources and acoustically consistent across the full video.

non_diegetic_music: N/A"""
CASE = dict(index=1, sample_id="giraffe_river", slug="giraffe_river", prompt=PROMPT,
            prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(), seed=SEED,
            original_prompt=PROMPT, vbench_dimensions=[])
ENV = dict(HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="4",
           PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
           FLASHINFER_CUDA_ARCH_LIST="12.0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_args(command="denoise"):
    args = pipeline.build_parser().parse_args([
        command, "--model", str(MODEL), "--output", str(OUT), "--method", "dense",
        "--frames", str(FRAMES), "--height", str(HEIGHT), "--width", str(WIDTH),
        "--steps", str(STEPS), "--workers", "1", "--no-torch-compile",
    ])
    return args


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    write(ROOT / "prompt.json", {"prompt": PROMPT, "prompt_sha256": CASE["prompt_sha256"]})
    configs = {
        "dense": {"method": "dense"},
        "sol-attn": dataclasses.asdict(h3.H3SparseAttentionConfig.sol(STEPS)),
        "spark-h3-10pct-warmup": dataclasses.asdict(
            h3.H3SparseAttentionConfig.spark(STEPS, warmup_percent=10)),
    }
    write(ROOT / "protocol.json", dict(
        name=NAME, prompt=PROMPT, prompt_sha256=CASE["prompt_sha256"], methods=METHODS,
        configs=configs, gpus={**GPU_FOR_METHOD, "conditioning_and_decode": 3}, model=str(MODEL),
        seed=SEED, requested_steps=STEPS, actual_transformer_evaluations=STEPS - 1,
        height=HEIGHT, width=WIDTH, requested_frames=FRAMES, fps=FPS, duration_seconds=10,
        native_aligned_frames=243, packaging_crop_frames=240,
        pipeline_source=str(BENCH / "scripts/minimax_h3_vbench_4gpu_pipeline.py"),
        implementation_source=str(REPO), environment=ENV))
    print(f"prepared {ROOT}", flush=True)


def condition():
    import torch
    torch.set_num_threads(4)
    started = time.perf_counter()
    workflow = pipeline.get_workflow(MODEL)
    states = pipeline.prepare_conditioning(
        workflow, str(MODEL), [CASE], conditioning_cache_dir=OUT / "conditioning_cache")
    summary = pipeline._conditioning_values_summary(
        states[0].values, prompt=PROMPT, source="giraffe conditioning")
    write(OUT / "conditioning_manifest.json", dict(
        status="complete", prompt=PROMPT, prompt_sha256=CASE["prompt_sha256"],
        elapsed_seconds=time.perf_counter() - started, summary=summary))
    print("conditioning complete", summary, flush=True)


def config_for(method):
    if method == "dense":
        return None
    if method == "sol-attn":
        return h3.H3SparseAttentionConfig.sol(STEPS)
    if method == "spark-h3-10pct-warmup":
        return h3.H3SparseAttentionConfig.spark(STEPS, warmup_percent=10)
    raise ValueError(method)


def denoise(method):
    import torch
    torch.set_num_threads(4)
    args = base_args()
    workflow, states = pipeline.configure_denoise_workflow(args, [CASE])
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)
    cfg = config_for(method)
    plugin_context = (contextlib.nullcontext() if cfg is None
                      else h3.install_h3_sparse_attention(pipe.transformer, cfg))
    target = OUT / method
    target.mkdir(parents=True, exist_ok=True)
    try:
        state = pipeline.clone_state(states[0])
        state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), plugin_context as plugin:
            result = pipe(state=state, num_frames=FRAMES, height=HEIGHT, width=WIDTH,
                          num_inference_steps=STEPS,
                          generator=torch.Generator(device="cpu").manual_seed(SEED),
                          output=["latents", "audio_latents"])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            attention_summary = plugin.summary() if plugin is not None else {"method": "dense"}
        if cfg is not None:
            assert attention_summary["completed_evaluations"] == STEPS - 1, attention_summary
            assert attention_summary["dense_evaluations"] == cfg.dense_evaluations, attention_summary
        payload = {key: result[key].detach().cpu().contiguous()
                   for key in ("latents", "audio_latents")}
        assert all(torch.isfinite(value).all() for value in payload.values())
        latent_path = target / "latents.pt"
        pipeline.atomic_torch_save(payload, latent_path)
        write(target / "denoise.json", dict(
            status="complete", method=method, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            seed=SEED, steps=STEPS, evaluations=STEPS - 1, frames=FRAMES,
            height=HEIGHT, width=WIDTH, denoise_seconds=seconds,
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            transformer_placement=placement, attention_config=(dataclasses.asdict(cfg) if cfg else None),
            attention_summary=attention_summary, latent_path=str(latent_path),
            latent_sha256=sha(latent_path)))
        print(f"denoise complete method={method} seconds={seconds:.1f}", flush=True)
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


def decode_all():
    import numpy as np
    import torch
    from diffusers.utils.export_utils import encode_video

    torch.set_num_threads(4)
    deadline = time.monotonic() + 8 * 3600
    while not all((OUT / method / "denoise.json").is_file() for method in METHODS):
        if time.monotonic() >= deadline:
            raise TimeoutError("decoder timed out waiting for all denoisers")
        time.sleep(2)
    args = base_args("decode")
    pipe, manager, acceleration = pipeline.load_decoder(args)
    try:
        for method in METHODS:
            target = OUT / method
            payload = torch.load(target / "latents.pt", map_location="cpu", weights_only=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = pipe(latents=payload["latents"].to("cuda"),
                              audio_latents=payload["audio_latents"].to("cuda"),
                              output_type="np", output=["videos", "audio", "sampling_rate"])
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - started
            frames = result["videos"][0][:FRAMES]
            assert frames.shape == (FRAMES, HEIGHT, WIDTH, 3) and np.isfinite(frames).all()
            sampling_rate = int(result["sampling_rate"])
            audio = result["audio"][0][..., :round(FRAMES / FPS * sampling_rate)]
            video_path = target / "video.mp4"
            partial = target / f".video.partial-{os.getpid()}.mp4"
            encode_started = time.perf_counter()
            encode_video(frames, fps=FPS, output_path=str(partial), audio=audio,
                         audio_sample_rate=sampling_rate)
            encode_seconds = time.perf_counter() - encode_started
            metadata = pipeline.inspect_video(partial)
            assert metadata["frames"] == FRAMES, metadata
            assert metadata["width"] == WIDTH and metadata["height"] == HEIGHT, metadata
            partial.replace(video_path)
            write(target / "decode.json", dict(
                status="complete", method=method, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                decode_seconds=decode_seconds, encode_seconds=encode_seconds,
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                video_path=str(video_path), video_sha256=sha(video_path), video=metadata))
            print(f"decode complete method={method} seconds={decode_seconds:.1f}", flush=True)
            del payload, result, frames, audio
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


def summarize(exit_codes):
    results = {}
    for method, code in zip(METHODS, exit_codes[:3], strict=True):
        denoise_path, decode_path = OUT / method / "denoise.json", OUT / method / "decode.json"
        results[method] = dict(exit_code=code,
                               denoise=json.loads(denoise_path.read_text()) if denoise_path.is_file() else None,
                               decode=json.loads(decode_path.read_text()) if decode_path.is_file() else None)
    summary = dict(status="complete" if not any(exit_codes) else "failed",
                   exit_codes=exit_codes, prompt_sha256=CASE["prompt_sha256"], results=results)
    write(ROOT / "run_summary.json", summary)
    write(OUT / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def run():
    condition_log = (ROOT / "gpu3_condition.log").open("a")
    env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": "3"}
    condition_proc = subprocess.Popen([sys.executable, __file__, "condition"], cwd=REPO,
                                      env=env, stdout=condition_log, stderr=subprocess.STDOUT)
    condition_code = condition_proc.wait()
    condition_log.close()
    if condition_code:
        summarize([99, 99, 99, condition_code])
        raise RuntimeError(f"conditioning failed with exit code {condition_code}")
    jobs = []
    for method in METHODS:
        gpu = GPU_FOR_METHOD[method]
        log = (ROOT / f"gpu{gpu}_{method}.log").open("a")
        child_env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
        proc = subprocess.Popen([sys.executable, __file__, "denoise", "--method", method],
                                cwd=REPO, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    decode_log = (ROOT / "gpu3_decode.log").open("a")
    decode_proc = subprocess.Popen([sys.executable, __file__, "decode"], cwd=REPO,
                                   env=env, stdout=decode_log, stderr=subprocess.STDOUT)
    jobs.append((decode_proc, decode_log))
    write(ROOT / "pids.json", [proc.pid for proc, _ in jobs])
    codes = [proc.wait() for proc, _ in jobs]
    for _, log in jobs:
        log.close()
    summarize(codes)
    if any(codes):
        raise RuntimeError(f"workers failed: {codes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "condition", "denoise", "decode", "run"))
    parser.add_argument("--method", choices=METHODS)
    cli = parser.parse_args()
    os.environ.update(ENV)
    if cli.stage == "denoise":
        if cli.method is None:
            parser.error("denoise requires --method")
        denoise(cli.method)
    elif cli.stage == "decode":
        decode_all()
    else:
        globals()[cli.stage]()
