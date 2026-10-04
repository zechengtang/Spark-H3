#!/usr/bin/env python3
"""Decode, losslessly archive, and score the 10s priority Reblock experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


import reblock_ablation_quality_5s768p as base


base.FRAMES = 240
ROOT_NAME = "reblock_priority_10s768p_25p_20261003"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_media(pipe, latent_info):
    import numpy as np
    import torch

    latent_path = Path(latent_info["latent_path"])
    if sha256(latent_path) != latent_info["latent_sha256"]:
        raise RuntimeError(f"latent changed after prepare: {latent_path}")
    payload = torch.load(latent_path, map_location="cpu", weights_only=True)
    for key in ("latents", "audio_latents"):
        if key not in payload or not torch.isfinite(payload[key]).all():
            raise RuntimeError(f"invalid {key}: {latent_path}")
    input_shapes = {key: list(payload[key].shape) for key in ("latents", "audio_latents")}
    torch.cuda.empty_cache()
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = pipe(
            latents=payload["latents"].cuda(),
            audio_latents=payload["audio_latents"].cuda(),
            output_type="np",
            output=["videos", "audio", "sampling_rate"],
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    video = np.asarray(decoded["videos"][0])
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < base.FRAMES:
        raise RuntimeError(f"unexpected decoded shape {video.shape}: {latent_path}")
    native_frames = int(video.shape[0])
    video = video[: base.FRAMES]
    if not np.isfinite(video).all():
        raise RuntimeError(f"non-finite decoded RGB: {latent_path}")
    minimum, maximum = float(video.min()), float(video.max())
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(f"decoded RGB outside [0,1]: min={minimum} max={maximum}")
    pixels = np.ascontiguousarray((video * 255).round().astype(np.uint8))
    rate = int(decoded["sampling_rate"])
    audio = decoded["audio"][0][..., : round(base.FRAMES / 24 * rate)]
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().float().numpy()
    else:
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None]
    if not np.isfinite(audio).all():
        raise RuntimeError(f"non-finite decoded audio: {latent_path}")
    del decoded, payload, video
    return pixels, np.ascontiguousarray(audio), rate, {
        "seconds": seconds,
        "native_frames": native_frames,
        "delivered_shape": list(pixels.shape),
        "decoded_float_min": minimum,
        "decoded_float_max": maximum,
        "input_shapes": input_shapes,
    }


def archive_media(experiment: Path, arm: str, case: dict, pixels, audio, rate: int):
    import numpy as np

    folder = experiment / "quality_videos" / arm
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{case['index']:02d}.mkv"
    evidence_path = output.with_suffix(".archive.json")
    if output.is_file() and evidence_path.is_file():
        evidence = base.read_json(evidence_path)
        if evidence.get("file_sha256") == sha256(output):
            return output, evidence
        raise RuntimeError(f"stale archive evidence: {output}")

    raw = folder / f".{case['index']:02d}.rgb-{os.getpid()}.npy"
    wav = folder / f".{case['index']:02d}.audio-{os.getpid()}.wav"
    np.save(raw, pixels)
    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate),
                "-ac", str(audio.shape[0]), "-i", "pipe:0", "-c:a", "pcm_f32le", str(wav),
            ],
            input=np.ascontiguousarray(audio.T).tobytes(),
            check=True,
        )
        if str(base.BENCH_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(base.BENCH_SCRIPTS))
        from h3_quality_batch_worker import export_rgb_video

        evidence = export_rgb_video(raw, wav, output, fps=24)
        evidence["file_sha256"] = sha256(output)
        evidence["arm"] = arm
        evidence["case"] = case["index"]
        base.atomic_json(evidence_path, evidence)
        return output, evidence
    finally:
        raw.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


def worker(experiment: Path, slot: int) -> None:
    import torch

    experiment = experiment.resolve()
    protocol = base.read_json(base.quality_root(experiment) / "protocol.json")
    cases = protocol["cases"][slot :: protocol["workers"]]
    pending_by_case = {
        case["index"]: [
            arm for arm in protocol["arms"]
            if not base.complete_result(base.result_path(experiment, arm, case), protocol, arm, case)
        ]
        for case in cases
    }
    pending_by_case = {case: arms for case, arms in pending_by_case.items() if arms}
    dense_missing = {
        case["index"] for case in cases
        if not (
            (experiment / "quality_videos" / "dense" / f"{case['index']:02d}.mkv").is_file()
            and (experiment / "quality_videos" / "dense" / f"{case['index']:02d}.archive.json").is_file()
        )
    }
    if not pending_by_case and not dense_missing:
        base.atomic_json(base.quality_root(experiment) / f"worker_{slot}.json", {"status": "complete", "slot": slot, "pairs": 0})
        return

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(base.LPIPS_PACKAGE))
    os.environ["TORCH_HOME"] = str(base.LPIPS_CACHE)
    import lpips
    torch.hub.set_dir(str(base.LPIPS_CACHE / "hub"))
    metric = lpips.LPIPS(net="alex", version="0.1", lpips=True, pnet_rand=False,
                         eval_mode=True, verbose=False).cuda().eval()
    coordinate = torch.arange(11, device="cuda", dtype=torch.float32) - 5
    kernel1 = torch.exp(-coordinate.square() / 4.5)
    kernel1 /= kernel1.sum()
    kernel = (kernel1[:, None] * kernel1[None, :]).expand(3, 1, 11, 11).contiguous()

    pipeline = base.import_pipeline()
    decode_args = pipeline.build_parser().parse_args(
        ["decode", "--model", str(base.MODEL), "--output", str(experiment),
         "--method", "dense", "--frames", str(base.FRAMES),
         "--height", str(base.HEIGHT), "--width", str(base.WIDTH)]
    )
    pipe, manager, acceleration = pipeline.load_decoder(decode_args)
    completed = 0
    try:
        for case in cases:
            arms = pending_by_case.get(case["index"], [])
            if not arms and case["index"] not in dense_missing:
                continue
            source = protocol["inputs"][str(case["index"])]
            reference, audio, rate, dense_decode = load_media(pipe, source["dense"])
            dense_video, dense_evidence = archive_media(
                experiment, "dense", case, reference, audio, rate
            )
            del audio
            for arm in arms:
                target = base.result_path(experiment, arm, case)
                try:
                    candidate, audio, rate, candidate_decode = load_media(pipe, source[arm])
                    video_path, video_evidence = archive_media(
                        experiment, arm, case, candidate, audio, rate
                    )
                    del audio
                    metrics = base.score_pair(reference, candidate, metric, kernel)
                    del candidate
                    base.atomic_json(target, {
                        "status": "complete", "schema": 1,
                        "protocol_digest": protocol["protocol_digest"],
                        "arm": arm, "case": case["index"],
                        "sample_id": case["sample_id"],
                        "prompt_sha256": case["prompt_sha256"],
                        "split": "tuning" if case["index"] in base.TUNING_CASES else "confirmation",
                        "inputs": {"dense": source["dense"], "candidate": source[arm]},
                        "pairing": {"dense_decoded_first": True, "same_worker": True,
                                    "dense_decode_reused_for_case": True},
                        "execution": {"slot": slot, "physical_gpu": slot,
                                      "gpu_uuid": protocol["gpu_inventory"][slot]["uuid"],
                                      "torch": torch.__version__, "cuda": torch.version.cuda,
                                      "dense_decode": dense_decode,
                                      "candidate_decode": candidate_decode,
                                      "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3},
                        "rgb": {"dtype": "uint8", "shape": [base.FRAMES, base.HEIGHT, base.WIDTH, 3],
                                "quantization": protocol["metrics"]["rgb_quantization"],
                                "decoded_video_saved": True},
                        "video_path": str(video_path),
                        "video_sha256": video_evidence["file_sha256"],
                        "dense_video_path": str(dense_video),
                        "dense_video_sha256": dense_evidence["file_sha256"],
                        "metrics": metrics,
                    })
                    completed += 1
                    print(f"COMPLETE slot={slot} arm={arm} case={case['index']} "
                          f"psnr={metrics['psnr']['db']:.6f} "
                          f"ssim={metrics['ssim']['mean']:.6f} "
                          f"lpips={metrics['lpips']['mean']:.6f}", flush=True)
                except Exception as error:
                    base.atomic_json(
                        base.quality_root(experiment) / "failures" /
                        f"{arm}_{case['index']:02d}_{time.time_ns()}.json",
                        {"status": "failed", "slot": slot, "arm": arm,
                         "case": case["index"], "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()},
                    )
                    raise
            del reference
        base.atomic_json(base.quality_root(experiment) / f"worker_{slot}.json",
                         {"status": "complete", "slot": slot, "pairs": completed})
    finally:
        acceleration.remove()
        del pipe, manager


_base_prepare = base.prepare


def prepare(experiment: Path, requested_arms):
    path = base.quality_root(experiment.resolve()) / "protocol.json"
    if path.exists():
        protocol = base.read_json(path)
        if protocol.get("implementation", {}).get("archive_adapter_sha256") != sha256(Path(__file__)):
            raise RuntimeError("10s quality/archive adapter differs from frozen protocol")
        return protocol
    protocol = _base_prepare(experiment, requested_arms)
    protocol["decoder"]["decoded_video_saved"] = True
    protocol["decoder"]["archive"] = "FFV1 level 3 Matroska with pixel/audio verification"
    protocol["implementation"]["archive_adapter"] = str(Path(__file__).resolve())
    protocol["implementation"]["archive_adapter_sha256"] = sha256(Path(__file__))
    core = {key: value for key, value in protocol.items()
            if key not in ("protocol_digest", "status", "created_utc")}
    protocol["protocol_digest"] = base.canonical_digest(core)
    base.atomic_json(path, protocol)
    return protocol


def manifests(experiment: Path, result: dict) -> None:
    protocol = base.read_json(base.quality_root(experiment) / "protocol.json")
    arms = ["dense", *protocol["arms"]]
    for arm in arms:
        records = []
        for case in protocol["cases"]:
            video = experiment / "quality_videos" / arm / f"{case['index']:02d}.mkv"
            evidence = base.read_json(video.with_suffix(".archive.json"))
            if evidence.get("file_sha256") != sha256(video):
                raise RuntimeError(f"invalid archive evidence: {video}")
            records.append({"index": case["index"], "sample_id": case["sample_id"],
                            "output_path": str(video.resolve()), "sha256": evidence["file_sha256"]})
        base.atomic_json(
            experiment / "quality_videos" / arm / "generation_manifest.json",
            {"schema_version": 1, "status": "passed", "method": arm,
             "sample_count": len(records), "records": records},
        )


_base_summarize = base.summarize


def summarize(experiment: Path):
    result = _base_summarize(experiment)
    manifests(experiment.resolve(), result)
    return result


def run(experiment: Path, requested_arms) -> None:
    protocol = prepare(experiment, requested_arms)
    root = base.quality_root(experiment.resolve())
    base.atomic_json(root / "status.json", {"status": "running", "arms": protocol["arms"]})
    jobs = []
    for slot in range(protocol["workers"]):
        log = (root / f"worker_{slot}.log").open("a")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(slot),
            "H3_IMPL_REPO": str(base.REPO),
            "H3_DIFFUSERS_DIR": str(base.MODEL),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": str(base.LPIPS_PACKAGE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "TORCH_HOME": str(base.LPIPS_CACHE),
        }
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker",
             "--experiment", str(experiment), "--slot", str(slot)],
            cwd=base.REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((slot, process, log))
    failures = []
    for slot, process, log in jobs:
        code = process.wait()
        log.close()
        if code:
            failures.append({"slot": slot, "exit_code": code})
    if failures:
        base.atomic_json(root / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"quality worker failures: {failures}")
    result = summarize(experiment)
    print(json.dumps(result["summaries"], indent=2))


base.prepare = prepare
base.worker = worker
base.summarize = summarize
base.run = run


if __name__ == "__main__":
    base.main()
