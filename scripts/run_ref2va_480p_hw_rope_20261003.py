#!/usr/bin/env python3
"""Three-arm Ref2VA 5s/480p H/W-RoPE ablation on independent GPUs."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
sys.path.insert(0, str(BENCH / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import ref2va_official_case as official  # noqa: E402
import run_ref2va_matrix_20261003 as matrix  # noqa: E402
import run_ref2va_480p_12fps_dense_spark_20261003 as fps12  # noqa: E402
from compute_ref2va_dense_reference_metrics_20261003 import compare, read_video  # noqa: E402
from run_ref2va_condition_fps_rope_20261003 import rope_patch  # noqa: E402


NAME = "ref2va_480p_hw_rope_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
REFERENCE = Path(
    "/autodl-fs/data/h3_outputs/ref2va_768_480_dense_spark_20261003/"
    "original768/5s/dense/video.mp4"
)
ARMS = {
    "480p_dense": {
        "fps": 24, "sparse": False,
        "cache_schema": 4,
        "cache_dir": matrix.CACHE_480,
        "case_dir": matrix.ROOT / "conditioning_480_case",
        "baseline": matrix.OUT / "reference480/5s/dense/video.mp4",
    },
    "480p_12fps_dense": {
        "fps": 12, "sparse": False,
        "cache_schema": 212,
        "cache_dir": fps12.OUT / "conditioning_480p_12fps",
        "case_dir": fps12.ROOT / "conditioning",
        "baseline": fps12.OUT / "dense/video.mp4",
    },
    "480p_12fps_spark": {
        "fps": 12, "sparse": True,
        "cache_schema": 212,
        "cache_dir": fps12.OUT / "conditioning_480p_12fps",
        "case_dir": fps12.ROOT / "conditioning",
        "baseline": fps12.OUT / "spark_target_condition_video_10pct/video.mp4",
    },
}
GPUS = tuple(int(x) for x in os.environ.get("H3_HW_ROPE_GPUS", "4,5,6").split(","))


def write(path, value):
    official.write(path, value)


def configure(arm):
    spec = ARMS[arm]
    official.CONDITIONING_CACHE_SCHEMA = spec["cache_schema"]
    official.CACHE_DIR = spec["cache_dir"]
    official.CASE_DIR = spec["case_dir"]
    official.DURATIONS = (5,)


@contextlib.contextmanager
def spatial_rope_patch(audit):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep

    original = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence

    def corrected(*args, **kwargs):
        result = list(original(*args, **kwargs))
        references, condition_latents = args[1], args[2]
        if len(references) != 2 or references[0].kind != "video" or references[1].kind != "audio":
            raise RuntimeError("H/W RoPE patch is scoped to the official [video, audio] Ref2VA case")
        position_ids, video_indices = result[0], result[2]
        condition_rows = result[5]
        condition_indices = video_indices[:condition_rows]
        _, patch_h, patch_w = args[8]
        reference_height, reference_width = condition_latents[0].shape[3:5]
        target_height, target_width = args[5], args[6]
        height_ratio = target_height / reference_height
        width_ratio = target_width / reference_width
        expected_rows = (
            condition_latents[0].shape[2]
            * (reference_height // patch_h)
            * (reference_width // patch_w)
        )
        assert condition_indices.numel() == expected_rows
        before = position_ids[condition_indices][:, 1:].clone()
        origin = before[0].clone()
        position_ids[condition_indices, 1] = origin[0] + (before[:, 0] - origin[0]) * height_ratio
        position_ids[condition_indices, 2] = origin[1] + (before[:, 1] - origin[1]) * width_ratio
        after = position_ids[condition_indices][:, 1:]
        audit.update(
            condition_rows=int(condition_rows),
            reference_latent_hw=[int(reference_height), int(reference_width)],
            target_latent_hw=[int(target_height), int(target_width)],
            scale_hw=[float(height_ratio), float(width_ratio)],
            origin_hw=[float(x) for x in origin],
            before_min_hw=[float(x) for x in before.amin(0)],
            before_max_hw=[float(x) for x in before.amax(0)],
            after_min_hw=[float(x) for x in after.amin(0)],
            after_max_hw=[float(x) for x in after.amax(0)],
        )
        result[0] = position_ids
        return tuple(result)

    MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(corrected)
    try:
        yield
    finally:
        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence = staticmethod(original)


def inspect_video(path):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True))
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    frames = video.get("nb_read_frames") or video.get("nb_frames")
    assert (int(video["width"]), int(video["height"]), int(frames)) == (1344, 768, 120)
    assert any(stream["codec_type"] == "audio" for stream in probe["streams"])
    return {"width": 1344, "height": 768, "frames": 120, "duration": float(probe["format"]["duration"])}


def worker(arm):
    import torch
    from diffusers import ComponentsManager
    from diffusers.utils.export_utils import encode_video
    from h3_sparse_attention import install_h3_sparse_attention

    spec = ARMS[arm]
    configure(arm)
    state, cache_path, conditioning_summary = official.load_conditioning(5)
    reference_latent = state.values["condition_latents"][0]
    assert tuple(reference_latent.shape[-2:]) == (30, 52)
    assert REFERENCE.is_file() and spec["baseline"].is_file()
    destination = OUT / arm
    destination.mkdir(parents=True, exist_ok=True)
    status = ROOT / f"{arm}.status.json"
    write(status, {"status": "loading", "arm": arm})

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

    audit = {}
    plugin_config = fps12.spark_config(20) if spec["sparse"] else None
    plugin_context = (
        install_h3_sparse_attention(pipe.transformer_ref, plugin_config)
        if plugin_config is not None else contextlib.nullcontext(None)
    )
    temporal_context = rope_patch(12, "physical_time") if spec["fps"] == 12 else contextlib.nullcontext()
    write(status, {"status": "denoising", "arm": arm})
    with official.compile_ref2va_transformer(pipe) as acceleration:
        wrapped = official._impl_bootstrap.verify_transformer_compile(acceleration, pipe.transformer_ref, requested=True)
        assert wrapped == 50
        prepared = official.clone_state(state)
        prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
        with temporal_context, spatial_rope_patch(audit), plugin_context as plugin, torch.inference_mode():
            if plugin is not None:
                plugin.reset()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipe(
                state=prepared, num_frames=state.values["num_frames"], height=768, width=1344,
                num_inference_steps=20, generator=torch.Generator(device="cpu").manual_seed(42),
                output=["latents", "audio_latents"],
            )
            torch.cuda.synchronize()
            denoise_seconds = time.perf_counter() - started
            attention_summary = None if plugin is None else plugin.summary()
        assert audit["scale_hw"] == [48 / 30, 84 / 52]
        if attention_summary is not None:
            assert attention_summary["completed_evaluations"] == 19
            assert attention_summary["dense_evaluations"] == 4
            assert attention_summary["sol_sparse_video_scope"] == "target_and_condition"
        payload = {key: result[key].detach().cpu().contiguous() for key in ("latents", "audio_latents")}
    del result, pipe, manager
    official.release_cpu_arenas()
    latent_path = destination / "latents.pt"
    official.atomic_torch_save(payload, latent_path)

    write(status, {"status": "decoding", "arm": arm})
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
    with torch.inference_mode():
        decoded = decoder(latents=payload["latents"].cuda(), audio_latents=payload["audio_latents"].cuda(),
                          output_type="np", output=["videos", "audio", "sampling_rate"])
    frames = decoded["videos"][0][:120]
    rate = decoded["sampling_rate"]
    audio = decoded["audio"][0][..., :round(5 * rate)]
    video_path = destination / "video.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    partial = destination / f".video.partial-{os.getpid()}.mp4"
    encode_video(frames, fps=24, output_path=str(partial), audio=audio, audio_sample_rate=rate)
    partial.replace(video_path)
    inspected = inspect_video(video_path)
    del decoder, manager, payload, decoded
    official.release_cpu_arenas()

    write(status, {"status": "quality", "arm": arm})
    reference_frames = read_video(REFERENCE)
    baseline_frames = read_video(spec["baseline"])
    candidate_frames = read_video(video_path)
    metrics = {
        "baseline_vs_original768_dense": compare(reference_frames, baseline_frames),
        "hw_rope_vs_original768_dense": compare(reference_frames, candidate_frames),
        "hw_rope_vs_arm_baseline": compare(baseline_frames, candidate_frames),
    }
    record = {
        "status": "complete", "arm": arm, "conditioning_fps": spec["fps"],
        "attention": "spark_target_condition_video_10pct" if spec["sparse"] else "dense",
        "baseline": str(spec["baseline"]), "dense_reference": str(REFERENCE),
        "denoise_seconds": denoise_seconds, "attention_summary": attention_summary,
        "conditioning_cache": str(cache_path), "conditioning_sha256": official.sha(cache_path),
        "conditioning_summary": conditioning_summary, "position_audit": audit,
        "latent_path": str(latent_path), "latent_sha256": official.sha(latent_path),
        "video_path": str(video_path), "video_sha256": official.sha(video_path),
        "video": inspected, "metrics": metrics,
    }
    write(ROOT / "records" / f"{arm}.json", record)
    write(status, {"status": "complete", "arm": arm})
    print(json.dumps(record, indent=2), flush=True)


def launch():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    if len(GPUS) < len(ARMS):
        raise RuntimeError(f"need at least {len(ARMS)} GPUs, got {GPUS}")
    serialized_arms = {
        arm: {key: str(value) if isinstance(value, Path) else value for key, value in spec.items()}
        for arm, spec in ARMS.items()
    }
    protocol = {
        "status": "running", "name": NAME, "arms": serialized_arms, "gpus": GPUS[:len(ARMS)],
        "duration": 5, "conditioning_resolution": [832, 480],
        "target_resolution": [1344, 768], "target_fps": 24,
        "steps": 20, "evaluations": 19, "seed": 42,
        "spatial_rope": "condition H/W offsets x target/reference latent H/W about first condition token",
        "timing_note": "compile-inclusive single-run diagnostic; quality metrics are the primary endpoint",
        "runner_sha256": official.sha(Path(__file__).resolve()),
    }
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")
    jobs = []
    for (arm, _), gpu in zip(ARMS.items(), GPUS):
        log = (ROOT / f"{arm}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", arm],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
                 "TORCHINDUCTOR_COMPILE_THREADS": "4", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((arm, process, log))
    codes = [(arm, process.wait()) for arm, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"workers failed: {codes}")
    records = {arm: json.loads((ROOT / "records" / f"{arm}.json").read_text()) for arm in ARMS}
    result = {"status": "complete", "records": records}
    write(ROOT / "results.json", result)
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(sys.argv[2])
    else:
        raise ValueError(command)
