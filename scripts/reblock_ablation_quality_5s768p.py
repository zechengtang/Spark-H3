#!/usr/bin/env python3
"""Paired full-video quality evaluation for the 5s768p reblock ablation.

Each GPU keeps one official MiniMax-H3 decoder resident.  Cases, rather than
individual arm/case pairs, are sharded so the matched Dense latent is decoded
once and reused for every pending candidate in that case.  Metrics follow the
Benchmark quality engine exactly: pooled RGB PSNR, frame-mean Gaussian SSIM
(11x11, sigma 1.5, valid window), and LPIPS AlexNet v0.1 on RGB [-1, 1].
Decoded frames are evaluated in memory; the much larger video archives are not
written because the paired metric records freeze both input latent hashes and
the decoder/metric implementation fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
BENCH_SCRIPTS = BENCH / "scripts"
PIPELINE_SOURCE = BENCH_SCRIPTS / "minimax_h3_vbench_4gpu_pipeline.py"
MODEL = Path("/mnt/CFS/tangzecheng/models/MiniMax-H3")
LPIPS_PACKAGE = Path("/mnt/CFS/tangzecheng/experiments/reblock_quality_deps")
LPIPS_CACHE = Path("/mnt/CFS/tangzecheng/experiments/reblock_quality_torch_cache")
FRAMES, HEIGHT, WIDTH = 120, 768, 1344
WORKERS = 8
TUNING_CASES = {1, 2, 3, 8, 9, 10, 11, 17, 18, 19}
METRIC_BATCH = 4


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def quality_root(experiment: Path) -> Path:
    return experiment / "quality_latent_decode"


def result_path(experiment: Path, arm: str, case: dict[str, Any]) -> Path:
    return quality_root(experiment) / "records" / arm / (
        f"{case['index']:02d}_{case['sample_id']}.json"
    )


def load_generation_rows(experiment: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    protocol = read_json(experiment / "protocol.json")
    results = read_json(experiment / "results.json")
    if protocol.get("status") != "complete" or results.get("status") != "complete":
        raise RuntimeError(f"generation is not complete: {experiment}")
    rows = {(row["arm"], int(row["case"])): row for row in results["records"]}
    if len(rows) != len(results["records"]):
        raise RuntimeError("duplicate generation arm/case record")
    return protocol, rows


def prepare(experiment: Path, requested_arms: list[str] | None) -> dict[str, Any]:
    import torch

    experiment = experiment.resolve()
    generation, rows = load_generation_rows(experiment)
    selected = list(generation["selected_arms"])
    default_arms = [arm for arm in selected if arm != "dense"]
    arms = list(requested_arms or default_arms)
    if not arms or "dense" in arms or len(set(arms)) != len(arms):
        raise ValueError("quality arms must be unique sparse arms")
    unknown = sorted(set(arms) - set(default_arms))
    if unknown:
        raise ValueError(f"arms are not complete in generation root: {unknown}")

    cases = generation["cases"]
    inputs: dict[str, dict[str, Any]] = {}
    for case in cases:
        index = int(case["index"])
        inputs[str(index)] = {}
        for arm in ["dense", *arms]:
            row = rows.get((arm, index))
            if row is None:
                raise RuntimeError(f"missing generation row arm={arm} case={index}")
            latent = Path(row["latent_path"])
            if (
                row.get("status") != "complete"
                or row.get("sample_id") != case["sample_id"]
                or row.get("prompt_sha256") != case["prompt_sha256"]
                or not latent.is_file()
                or sha256(latent) != row.get("latent_sha256")
            ):
                raise RuntimeError(f"invalid generation input arm={arm} case={index}")
            inputs[str(index)][arm] = {
                "record_path": str(experiment / "records" / arm / latent.with_suffix(".json").name),
                "latent_path": str(latent),
                "latent_sha256": row["latent_sha256"],
                "denoise_seconds": row["denoise_seconds"],
                "steps": row["steps"],
                "seed": row["seed"],
                "frames": row["frames"],
                "height": row["height"],
                "width": row["width"],
            }
            if (row["steps"], row["seed"], row["frames"], row["height"], row["width"]) != (
                20, 42, FRAMES, HEIGHT, WIDTH
            ):
                raise RuntimeError(f"generation metadata mismatch arm={arm} case={index}")

    core = {
        "schema": 1,
        "experiment": str(experiment),
        "model": str(MODEL),
        "arms": arms,
        "cases": cases,
        "inputs": inputs,
        "workers": WORKERS,
        "gpu_inventory": generation["gpu_inventory"],
        "case_shards": {
            str(slot): [int(case["index"]) for case in cases[slot::WORKERS]]
            for slot in range(WORKERS)
        },
        "decoder": {
            "source": str(PIPELINE_SOURCE),
            "source_sha256": sha256(PIPELINE_SOURCE),
            "dtype": "bfloat16 load request; Benchmark keeps VAE numerics unchanged",
            "torch_compile": False,
            "native_video_frames_minimum": FRAMES,
            "delivered_frames": FRAMES,
            "decoded_video_saved": False,
        },
        "metrics": {
            "rgb_quantization": "assert finite and [0,1], then (x*255).round().astype(uint8)",
            "psnr": "-10*log10(sum((dense/255-candidate/255)^2)/(F*H*W*C))",
            "ssim": "frame mean; Gaussian 11x11 sigma=1.5 valid, C1=0.01^2 C2=0.03^2",
            "lpips": "AlexNet v0.1 learned, candidate first, uint8 RGB mapped to [-1,1]",
            "metric_batch": METRIC_BATCH,
            "tf32": False,
        },
        "lpips_package": str(LPIPS_PACKAGE),
        "lpips_cache": str(LPIPS_CACHE),
        "lpips_evidence": {
            "lpips_py_sha256": sha256(LPIPS_PACKAGE / "lpips" / "lpips.py"),
            "pretrained_networks_py_sha256": sha256(
                LPIPS_PACKAGE / "lpips" / "pretrained_networks.py"
            ),
            "learned_alex_v01_sha256": sha256(
                LPIPS_PACKAGE / "lpips" / "weights" / "v0.1" / "alex.pth"
            ),
            "torchvision_alexnet_sha256": sha256(
                LPIPS_CACHE / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
            ),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "implementation": {
            "quality_script": str(Path(__file__).resolve()),
            "quality_script_sha256": sha256(Path(__file__).resolve()),
            "generation_implementation": generation["implementation"],
        },
    }
    core["protocol_digest"] = canonical_digest(core)
    core["status"] = "prepared"
    core["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root = quality_root(experiment)
    existing_path = root / "protocol.json"
    if existing_path.exists():
        existing = read_json(existing_path)
        # Time/status are bookkeeping; every numerical input and implementation
        # field must otherwise remain frozen for resume.
        for key in ("created_utc", "status", "protocol_digest"):
            existing.pop(key, None)
            core.pop(key, None)
        if existing != core:
            raise RuntimeError("quality protocol differs from the existing frozen protocol")
        return read_json(existing_path)
    atomic_json(existing_path, core)
    atomic_json(root / "status.json", {"status": "prepared"})
    return core


def complete_result(path: Path, protocol: dict[str, Any], arm: str, case: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        row = read_json(path)
        source = protocol["inputs"][str(case["index"])]
        return (
            row.get("status") == "complete"
            and row.get("schema") == 1
            and row.get("protocol_digest") == protocol["protocol_digest"]
            and row.get("arm") == arm
            and row.get("case") == case["index"]
            and row.get("sample_id") == case["sample_id"]
            and row.get("prompt_sha256") == case["prompt_sha256"]
            and row["inputs"]["dense"]["latent_sha256"] == source["dense"]["latent_sha256"]
            and row["inputs"]["candidate"]["latent_sha256"] == source[arm]["latent_sha256"]
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def import_pipeline():
    os.environ.setdefault("H3_IMPL_REPO", str(REPO))
    os.environ.setdefault("H3_DIFFUSERS_DIR", str(MODEL))
    if str(BENCH_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BENCH_SCRIPTS))
    import _impl_bootstrap
    if Path(_impl_bootstrap.IMPL_REPO).resolve() != REPO.resolve():
        raise RuntimeError(f"implementation mismatch: {_impl_bootstrap.IMPL_REPO}")
    import minimax_h3_vbench_4gpu_pipeline as pipeline
    return pipeline


def load_pixels(pipe, latent_info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
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
    del decoded, payload
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < FRAMES:
        raise RuntimeError(f"unexpected decoded shape {video.shape}: {latent_path}")
    native_frames = int(video.shape[0])
    video = video[:FRAMES]
    if not np.isfinite(video).all():
        raise RuntimeError(f"non-finite decoded RGB: {latent_path}")
    minimum, maximum = float(video.min()), float(video.max())
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(f"decoded RGB outside [0,1]: min={minimum} max={maximum}")
    pixels = np.ascontiguousarray((video * 255).round().astype(np.uint8))
    del video
    if pixels.shape != (FRAMES, HEIGHT, WIDTH, 3):
        raise RuntimeError(f"unexpected delivered RGB shape {pixels.shape}: {latent_path}")
    return pixels, {
        "seconds": seconds,
        "native_frames": native_frames,
        "delivered_shape": list(pixels.shape),
        "decoded_float_min": minimum,
        "decoded_float_max": maximum,
        "input_shapes": input_shapes,
    }


def score_pair(reference, candidate, lpips_metric, kernel) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError("paired RGB shape/dtype mismatch")

    def ssim(a, b):
        ma = functional.conv2d(a, kernel, groups=3)
        mb = functional.conv2d(b, kernel, groups=3)
        va = functional.conv2d(a * a, kernel, groups=3) - ma * ma
        vb = functional.conv2d(b * b, kernel, groups=3) - mb * mb
        cov = functional.conv2d(a * b, kernel, groups=3) - ma * mb
        return (
            ((2 * ma * mb + 0.0001) * (2 * cov + 0.0009))
            / ((ma * ma + mb * mb + 0.0001) * (va + vb + 0.0009))
        ).flatten(1).mean(1)

    started = time.perf_counter()
    sse = 0.0
    count = 0
    frame_psnr: list[float | None] = []
    frame_ssim: list[float] = []
    frame_lpips: list[float] = []
    with torch.inference_mode():
        for start in range(0, reference.shape[0], METRIC_BATCH):
            a = (
                torch.from_numpy(reference[start : start + METRIC_BATCH])
                .cuda()
                .float()
                .permute(0, 3, 1, 2)
                / 255
            )
            b = (
                torch.from_numpy(candidate[start : start + METRIC_BATCH])
                .cuda()
                .float()
                .permute(0, 3, 1, 2)
                / 255
            )
            diff = (a - b).square().flatten(1).sum(1)
            per_frame_count = a[0].numel()
            sse += float(diff.sum())
            count += a.numel()
            frame_psnr.extend(
                None if float(value) == 0.0 else -10 * math.log10(float(value) / per_frame_count)
                for value in diff.cpu()
            )
            frame_ssim.extend(float(value) for value in ssim(a, b).cpu())
            frame_lpips.extend(
                float(value)
                for value in lpips_metric(b * 2 - 1, a * 2 - 1, normalize=False).flatten().cpu()
            )
    torch.cuda.synchronize()
    mse = sse / count
    return {
        "score_seconds": time.perf_counter() - started,
        "psnr": {
            "sse": sse,
            "count": count,
            "mse": mse,
            "db": None if sse == 0.0 else -10 * math.log10(mse),
            "exact_rgb_match": sse == 0.0,
            "frame_db": frame_psnr,
        },
        "ssim": {
            "mean": statistics.fmean(frame_ssim),
            "sum": sum(frame_ssim),
            "count": len(frame_ssim),
            "frame": frame_ssim,
        },
        "lpips": {
            "mean": statistics.fmean(frame_lpips),
            "sum": sum(frame_lpips),
            "count": len(frame_lpips),
            "frame": frame_lpips,
            "net": "alex",
            "version": "0.1",
        },
    }


def worker(experiment: Path, slot: int) -> None:
    import numpy as np
    import torch

    experiment = experiment.resolve()
    protocol = read_json(quality_root(experiment) / "protocol.json")
    cases = protocol["cases"][slot::protocol["workers"]]
    pending_by_case = {
        case["index"]: [
            arm
            for arm in protocol["arms"]
            if not complete_result(result_path(experiment, arm, case), protocol, arm, case)
        ]
        for case in cases
    }
    pending_by_case = {case: arms for case, arms in pending_by_case.items() if arms}
    if not pending_by_case:
        atomic_json(quality_root(experiment) / f"worker_{slot}.json", {"status": "complete", "slot": slot, "pairs": 0})
        return

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(LPIPS_PACKAGE))
    os.environ["TORCH_HOME"] = str(LPIPS_CACHE)
    import lpips
    torch.hub.set_dir(str(LPIPS_CACHE / "hub"))
    metric = lpips.LPIPS(
        net="alex", version="0.1", lpips=True, pnet_rand=False, eval_mode=True, verbose=False
    ).cuda().eval()
    coordinate = torch.arange(11, device="cuda", dtype=torch.float32) - 5
    kernel1 = torch.exp(-coordinate.square() / 4.5)
    kernel1 /= kernel1.sum()
    kernel = (kernel1[:, None] * kernel1[None, :]).expand(3, 1, 11, 11).contiguous()

    pipeline = import_pipeline()
    decode_args = pipeline.build_parser().parse_args(
        ["decode", "--model", str(MODEL), "--output", str(experiment), "--method", "dense",
         "--frames", str(FRAMES), "--height", str(HEIGHT), "--width", str(WIDTH)]
    )
    pipe, manager, acceleration = pipeline.load_decoder(decode_args)
    completed = 0
    try:
        for case in cases:
            arms = pending_by_case.get(case["index"], [])
            if not arms:
                continue
            torch.cuda.reset_peak_memory_stats()
            source = protocol["inputs"][str(case["index"])]
            reference, dense_decode = load_pixels(pipe, source["dense"])
            for arm in arms:
                target = result_path(experiment, arm, case)
                try:
                    candidate, candidate_decode = load_pixels(pipe, source[arm])
                    metrics = score_pair(reference, candidate, metric, kernel)
                    del candidate
                    atomic_json(
                        target,
                        {
                            "status": "complete",
                            "schema": 1,
                            "protocol_digest": protocol["protocol_digest"],
                            "arm": arm,
                            "case": case["index"],
                            "sample_id": case["sample_id"],
                            "prompt_sha256": case["prompt_sha256"],
                            "split": "tuning" if case["index"] in TUNING_CASES else "confirmation",
                            "inputs": {"dense": source["dense"], "candidate": source[arm]},
                            "pairing": {
                                "dense_decoded_first": True,
                                "same_worker": True,
                                "dense_decode_reused_for_case": True,
                            },
                            "execution": {
                                "slot": slot,
                                "physical_gpu": slot,
                                "gpu_uuid": protocol["gpu_inventory"][slot]["uuid"],
                                "torch": torch.__version__,
                                "cuda": torch.version.cuda,
                                "dense_decode": dense_decode,
                                "candidate_decode": candidate_decode,
                                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                            },
                            "rgb": {
                                "dtype": "uint8",
                                "shape": [FRAMES, HEIGHT, WIDTH, 3],
                                "quantization": protocol["metrics"]["rgb_quantization"],
                                "decoded_video_saved": False,
                            },
                            "metrics": metrics,
                        },
                    )
                    completed += 1
                    print(
                        f"COMPLETE slot={slot} arm={arm} case={case['index']} "
                        f"psnr={metrics['psnr']['db']:.6f} ssim={metrics['ssim']['mean']:.6f} "
                        f"lpips={metrics['lpips']['mean']:.6f}",
                        flush=True,
                    )
                except Exception as error:
                    atomic_json(
                        quality_root(experiment) / "failures" / f"{arm}_{case['index']:02d}_{time.time_ns()}.json",
                        {
                            "status": "failed",
                            "slot": slot,
                            "arm": arm,
                            "case": case["index"],
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise
            del reference
        atomic_json(
            quality_root(experiment) / f"worker_{slot}.json",
            {"status": "complete", "slot": slot, "pairs": completed},
        )
    finally:
        acceleration.remove()
        del pipe, manager


def summarize(experiment: Path) -> dict[str, Any]:
    experiment = experiment.resolve()
    root = quality_root(experiment)
    protocol = read_json(root / "protocol.json")
    rows = []
    for arm in protocol["arms"]:
        for case in protocol["cases"]:
            path = result_path(experiment, arm, case)
            if not complete_result(path, protocol, arm, case):
                raise RuntimeError(f"incomplete quality result arm={arm} case={case['index']}")
            rows.append(read_json(path))
    summaries = {}
    for split in ("tuning", "confirmation", "all"):
        summaries[split] = {}
        for arm in protocol["arms"]:
            selected = [
                row for row in rows
                if row["arm"] == arm and (split == "all" or row["split"] == split)
            ]
            summaries[split][arm] = {
                "samples": len(selected),
                "mean_psnr_db": statistics.fmean(row["metrics"]["psnr"]["db"] for row in selected),
                "mean_ssim": statistics.fmean(row["metrics"]["ssim"]["mean"] for row in selected),
                "mean_lpips": statistics.fmean(row["metrics"]["lpips"]["mean"] for row in selected),
                "mean_candidate_decode_seconds": statistics.fmean(
                    row["execution"]["candidate_decode"]["seconds"] for row in selected
                ),
                "mean_denoise_seconds": statistics.fmean(
                    row["inputs"]["candidate"]["denoise_seconds"] for row in selected
                ),
            }
    result = {
        "status": "complete",
        "schema": 1,
        "protocol_digest": protocol["protocol_digest"],
        "summaries": summaries,
        "records": rows,
    }
    atomic_json(root / "results.json", result)
    protocol["status"] = "complete"
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "status.json", {"status": "complete"})
    return result


def run(experiment: Path, requested_arms: list[str] | None) -> None:
    protocol = prepare(experiment, requested_arms)
    root = quality_root(experiment.resolve())
    atomic_json(root / "status.json", {"status": "running", "arms": protocol["arms"]})
    jobs = []
    for slot in range(protocol["workers"]):
        log = (root / f"worker_{slot}.log").open("a")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(slot),
            "H3_IMPL_REPO": str(REPO),
            "H3_DIFFUSERS_DIR": str(MODEL),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": str(LPIPS_PACKAGE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "TORCH_HOME": str(LPIPS_CACHE),
        }
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", "--experiment", str(experiment), "--slot", str(slot)],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((slot, process, log))
    failures = []
    for slot, process, log in jobs:
        code = process.wait()
        log.close()
        if code:
            failures.append({"slot": slot, "exit_code": code})
    if failures:
        atomic_json(root / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"quality worker failures: {failures}")
    result = summarize(experiment)
    print(json.dumps(result["summaries"], indent=2))


def status(experiment: Path) -> None:
    root = quality_root(experiment.resolve())
    payload = {
        "root": str(root),
        "status": read_json(root / "status.json") if (root / "status.json").exists() else None,
        "workers": [],
    }
    for slot in range(WORKERS):
        log = root / f"worker_{slot}.log"
        complete = []
        if log.exists():
            complete = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("COMPLETE ")]
        payload["workers"].append({"slot": slot, "completed": len(complete), "last": complete[-1] if complete else None})
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "summarize", "status"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--slot", type=int, choices=range(WORKERS))
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.experiment, args.arms), indent=2))
    elif args.command == "run":
        run(args.experiment, args.arms)
    elif args.command == "worker":
        if args.slot is None:
            parser.error("worker requires --slot")
        worker(args.experiment, args.slot)
    elif args.command == "summarize":
        print(json.dumps(summarize(args.experiment)["summaries"], indent=2))
    else:
        status(args.experiment)


if __name__ == "__main__":
    main()
