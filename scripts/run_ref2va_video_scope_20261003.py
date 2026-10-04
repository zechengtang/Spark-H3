#!/usr/bin/env python3
"""Ref2VA Spark test with target and conditioning video in the sparse domain."""
from __future__ import annotations

import contextlib
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


NAME = "ref2va_target_condition_video_spark_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
BASELINE = Path("/autodl-fs/data/h3_outputs/ref2va_768_480_dense_spark_20261003/original768")
DURATIONS = (5, 10)
GPUS = (0, 1)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spark_config(steps: int):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        steps,
        warmup_percent=20.0 if steps == 20 else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=0.10,
        sol_sparse_video_scope="target_and_condition",
        sol_log_density=False,
    )


def prepare() -> None:
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for duration in DURATIONS:
        (OUT / f"{duration}s" / "spark_target_condition_video_10pct").mkdir(parents=True)
    diff = subprocess.check_output(
        ["git", "diff", "--", "h3_sparse_attention/processor.py", "h3_sparse_attention/spark_integration.py"],
        cwd=REPO,
    )
    write(
        ROOT / "protocol.json",
        {
            "status": "running",
            "name": NAME,
            "gpus": GPUS,
            "durations": DURATIONS,
            "reference_variant": "original768",
            "target_resolution": [1344, 768],
            "requested_steps": 20,
            "actual_evaluations": 19,
            "seed": 42,
            "spark": {
                "topk_ratio": 0.10,
                "sparse_video_scope": "target_and_condition",
                "dense_evaluations": 4,
                "dense_layers": 1,
                "dense_context": ["text", "reference_audio", "target_audio"],
            },
            "runtime_warmup": "excluded 3-step pass: one dense + one sparse evaluation",
            "baseline_root": str(BASELINE),
            "runner_sha256": sha(Path(__file__).resolve()),
            "processor_sha256": sha(REPO / "h3_sparse_attention/processor.py"),
            "spark_integration_sha256": sha(REPO / "h3_sparse_attention/spark_integration.py"),
            "relevant_diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
    )
    shutil.copy2(__file__, ROOT / "runner_source.py")


def clone_state(state):
    return official.clone_state(state)


def run_denoise(pipe, state, *, steps: int):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    prepared = clone_state(state)
    prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
    with install_h3_sparse_attention(pipe.transformer_ref, spark_config(steps)) as plugin, torch.inference_mode():
        plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=prepared,
            num_frames=state.values["num_frames"],
            height=768,
            width=1344,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary()
    expected_dense = 4 if steps == 20 else 1
    assert summary["completed_evaluations"] == steps - 1, summary
    assert summary["dense_evaluations"] == expected_dense, summary
    assert summary["sol_sparse_video_scope"] == "target_and_condition", summary
    assert summary["condition_video_tokens"] > 0, summary
    assert summary["sparse_video_tokens"] == (
        summary["target_video_tokens"] + summary["condition_video_tokens"]
    ), summary
    assert summary["processor_calls"]["sparse:sol"] == (steps - 1 - expected_dense) * 49, summary
    assert all(torch.isfinite(result[key]).all().item() for key in ("latents", "audio_latents"))
    del prepared
    return result, seconds, summary


def inspect_video(path: Path, expected_frames: int):
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            text=True,
        )
    )
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
    assert (int(video["width"]), int(video["height"])) == (1344, 768)
    assert audio
    if video.get("nb_frames") not in (None, "N/A"):
        assert int(video["nb_frames"]) == expected_frames
    return {
        "width": 1344,
        "height": 768,
        "frames": expected_frames,
        "duration": float(probe["format"]["duration"]),
        "audio_streams": len(audio),
    }


def worker(rank: int) -> None:
    import torch
    from diffusers import ComponentsManager
    from diffusers.utils.export_utils import encode_video

    duration = DURATIONS[int(rank)]
    state, cache_path, conditioning_summary = official.load_conditioning(duration)
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

    with official.compile_ref2va_transformer(pipe) as acceleration:
        wrapped = official._impl_bootstrap.verify_transformer_compile(
            acceleration, pipe.transformer_ref, requested=True
        )
        assert wrapped == 50, wrapped
        warm, warm_seconds, warm_summary = run_denoise(pipe, state, steps=3)
        del warm
        result, seconds, summary = run_denoise(pipe, state, steps=20)
        payload = {
            key: result[key].detach().cpu().contiguous()
            for key in ("latents", "audio_latents")
        }
        del result

    destination = OUT / f"{duration}s" / "spark_target_condition_video_10pct"
    latent = destination / "latents.pt"
    official.atomic_torch_save(payload, latent)
    record = {
        "method": "spark_target_condition_video_10pct",
        "duration": duration,
        "denoise_seconds": seconds,
        "attention_summary": summary,
        "latent_path": str(latent),
        "latent_sha256": sha(latent),
        "conditioning_cache": str(cache_path),
        "conditioning_sha256": sha(cache_path),
        "conditioning_summary": conditioning_summary,
        "compile_wrapped_blocks": wrapped,
        "warmup_seconds": warm_seconds,
        "warmup_summary": warm_summary,
    }
    del pipe, manager
    official.release_cpu_arenas()
    gc.collect()
    torch.cuda.empty_cache()

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
    with torch.inference_mode():
        decoded = decoder(
            latents=payload["latents"].cuda(),
            audio_latents=payload["audio_latents"].cuda(),
            output_type="np",
            output=["videos", "audio", "sampling_rate"],
        )
    requested = official.REQUESTED_FRAMES[duration]
    frames = decoded["videos"][0][:requested]
    assert frames.shape == (requested, 768, 1344, 3)
    assert np.isfinite(frames).all()
    rate = decoded["sampling_rate"]
    audio = decoded["audio"][0][..., : round(requested / 24 * rate)]
    video = destination / "video.mp4"
    partial = video.with_name(f".{video.stem}.partial-{os.getpid()}{video.suffix}")
    encode_video(frames, fps=24, output_path=str(partial), audio=audio, audio_sample_rate=rate)
    partial.replace(video)
    record.update(video_path=str(video), video_sha256=sha(video), video=inspect_video(video, requested))
    write(destination / "meta.json", record)
    write(ROOT / f"worker_gpu{rank}.json", {"status": "complete", "duration": duration})
    print("COMPLETE", duration, round(seconds, 3), flush=True)


def tensor_metrics(candidate, reference):
    import torch

    candidate = candidate.float()
    reference = reference.float()
    error = candidate - reference
    mse = error.square().mean().item()
    signal = reference.square().mean().item()
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "relative_rmse": (mse / max(signal, 1e-30)) ** 0.5,
        "cosine": torch.nn.functional.cosine_similarity(
            candidate.flatten(), reference.flatten(), dim=0
        ).item(),
        "max_abs": error.abs().max().item(),
    }


def summarize() -> None:
    import torch

    records = {}
    for duration in DURATIONS:
        current_path = OUT / f"{duration}s" / "spark_target_condition_video_10pct" / "meta.json"
        current = json.loads(current_path.read_text())
        new_payload = torch.load(current["latent_path"], map_location="cpu", weights_only=True)
        comparisons = {}
        for method in ("dense", "spark_10pct"):
            baseline_meta = json.loads((BASELINE / f"{duration}s" / method / "meta.json").read_text())
            baseline_payload = torch.load(baseline_meta["latent_path"], map_location="cpu", weights_only=True)
            comparisons[method] = {
                "baseline_denoise_seconds": baseline_meta["denoise_seconds"],
                "speedup_over_new": baseline_meta["denoise_seconds"] / current["denoise_seconds"],
                "latents": tensor_metrics(new_payload["latents"], baseline_payload["latents"]),
                "audio_latents": tensor_metrics(
                    new_payload["audio_latents"], baseline_payload["audio_latents"]
                ),
            }
        records[f"{duration}s"] = {"current": current, "comparisons": comparisons}
    result = {"status": "complete", "records": records}
    write(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


def launch() -> None:
    prepare()
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(rank)],
            env={
                **os.environ,
                **official.ENV,
                **official.SPARK_ENV,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "H3_IMPL_REPO": str(REPO),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((process, log))
    codes = [process.wait() for process, _ in jobs]
    for _, log in jobs:
        log.close()
    if any(codes):
        raise RuntimeError(f"workers failed: {codes}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(int(sys.argv[2]))
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
