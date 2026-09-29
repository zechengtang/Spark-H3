#!/usr/bin/env python3
"""Paired ComfyUI Spark FP32/BF16-anchor benchmark on 5s and 10s 768p."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import subprocess

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source


NAME = "comfyui_spark_anchor_precision_10prompt_5s10s768p_20260925"
REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parent / "ComfyUI"
KITCHEN = REPO.parent / "comfy-kitchen"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
BASE5_ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_spark_sol_10prompt_5s768p_20260924")
BASE5_OUT = Path("/autodl-fs/data/h3_outputs/comfyui_spark_sol_10prompt_5s768p_20260924")
BASE10_ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_sol_tau1_vs_spark_10prompt_10s768p_20260924")
GPUS = (0, 1)
PORTS = {0: 8350, 1: 8351}
DECODE_PORTS = {0: 8360, 1: 8361}
SPARK_METHODS = ("spark_fp32", "spark_bf16")
GENERATED = {5: SPARK_METHODS, 10: ("dense",) + SPARK_METHODS}
ALL_METHODS = ("dense", "sol_tau1_extra0") + SPARK_METHODS
DURATION = {
    5: {"requested_frames": 120, "model_frames": 123},
    10: {"requested_frames": 240, "model_frames": 243},
}
WIDTH, HEIGHT, FPS, STEPS, SEED = 1344, 768, 24.0, 20, 42


def read(path):
    return json.loads(Path(path).read_text())


def cases():
    return source.cases()


def duration_root(seconds):
    return ROOT / f"{seconds}s"


def duration_out(seconds):
    return OUT / f"{seconds}s"


def latent_path(seconds, method, case):
    if seconds == 5 and method in ("dense", "sol_tau1_extra0"):
        return BASE5_ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"
    if seconds == 10 and method == "sol_tau1_extra0":
        return BASE10_ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"
    return duration_root(seconds) / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def graph(seconds, method, case, target, steps=STEPS):
    if method == "dense":
        value = source.denoise_graph("dense", case, target, steps)
    elif method in SPARK_METHODS:
        value = source.denoise_graph("spark_topk10", case, target, steps)
        value["2"]["inputs"].update(
            ablation_mode="full",
            video_tail_mode="dense",
            global_anchor_dtype="bfloat16" if method == "spark_bf16" else "float32",
        )
    else:
        raise ValueError(method)
    value["4"]["inputs"].update(
        width=WIDTH,
        height=HEIGHT,
        length=DURATION[seconds]["requested_frames"],
    )
    return value


def decode_graph(seconds, method, case):
    output = duration_out(seconds) / method / f"{case['index']:02}_{case['sample_id']}_ultrafast.mp4"
    return {
        "1": {"class_type": "LoadMiniMaxH3AVLatentCache", "inputs": {
            "cache_path": str(latent_path(seconds, method, case)),
        }},
        "2": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors",
        }},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["2", 0]}},
        "6": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["4", 0], "batch_index": 0,
            "length": DURATION[seconds]["requested_frames"],
        }},
        "8": {"class_type": "CreateVideo", "inputs": {
            "images": ["6", 0], "fps": FPS,
            "bit_depth": 8, "color_space": "sRGB", "codec": "none",
        }},
        "10": {"class_type": "SaveVideoLosslessUltrafast", "inputs": {
            "video": ["8", 0], "output_path": str(output),
        }},
    }


def _baseline_identity():
    expected = [(c["index"], c["sample_id"], c["prompt_sha256"]) for c in cases()]
    for path in (BASE5_ROOT / "protocol.json", BASE10_ROOT / "protocol.json"):
        actual = [(c["index"], c["sample_id"], c["prompt_sha256"]) for c in read(path)["cases"]]
        if actual != expected:
            raise RuntimeError(f"baseline prompt mismatch: {path}")
    for seconds in DURATION:
        for method in ("dense", "sol_tau1_extra0"):
            if seconds == 10 and method == "dense":
                continue
            for case in cases():
                if not latent_path(seconds, method, case).is_file():
                    raise FileNotFoundError(latent_path(seconds, method, case))


def prepare():
    _baseline_identity()
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for seconds, methods in GENERATED.items():
        dr, do = duration_root(seconds), duration_out(seconds)
        for method in methods:
            for section in ("latents", "records", "warmup"):
                (dr / section / method).mkdir(parents=True, exist_ok=True)
            (do / method).mkdir(parents=True, exist_ok=True)
        if seconds == 10:
            (do / "sol_tau1_extra0").mkdir(parents=True, exist_ok=True)
    for gpu in GPUS:
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
            (ROOT / section / f"decoder_gpu{gpu}").mkdir(parents=True, exist_ok=True)
    kitchen_diff = os.popen(f"git -C {KITCHEN} diff --no-ext-diff").read()
    base.write_json(ROOT / "protocol.json", {
        "name": NAME,
        "purpose": "Current comfy-kitchen Spark FP32 anchor vs BF16 anchor, with reused official Sol baselines",
        "pipeline": "ComfyUI-native MiniMax-H3 denoise-only; reused identical BF16 conditioning",
        "methods": {
            "dense": {"patch": None},
            "sol_tau1_extra0": {"node": "BlockSparseAttention", "tau": 1.0, "extra_tokens": 0},
            "spark_fp32": {"topk_ratio": 0.1, "global_anchor_dtype": "float32"},
            "spark_bf16": {"topk_ratio": 0.1, "global_anchor_dtype": "bfloat16"},
        },
        "shared_sparse": {
            "dense_evaluations": 4, "dense_layers": [0], "min_tokens": 4096,
            "sink_conditioning": "exact_kv_and_rows", "video_tail_mode": "dense",
            "forced_local_blocks": False, "extra_tokens": 0,
        },
        "settings": {
            "seed": SEED, "steps": STEPS, "sampler": "res_multistep", "scheduler": "simple",
            "width": WIDTH, "height": HEIGHT, "fps": FPS, "turbo_lora": False,
            "durations": DURATION, "gpus": list(GPUS),
            "timing": "SamplerCustomAdvanced only; conditioning and VAE excluded",
            "warmup": "one excluded 5-step full-resolution run per generated arm per GPU and duration",
            "pairing": "each prompt's FP32/BF16 arms stay on one GPU; order alternates",
        },
        "baselines": {"5s": str(BASE5_ROOT), "10s": str(BASE10_ROOT)},
        "cases": cases(),
        "conditioning_source": str(source.EMBEDDINGS),
        "revisions": {
            "comfyui": base.git_revision(COMFY), "spark_h3": base.git_revision(REPO),
            "comfy_kitchen": base.git_revision(KITCHEN),
            "comfy_kitchen_diff_sha256": __import__("hashlib").sha256(kitchen_diff.encode()).hexdigest(),
        },
    })


def assigned_cases(gpu):
    return [case for case in cases() if (case["index"] - 1) % len(GPUS) == gpu]


def _record_path(seconds, method, case):
    return duration_root(seconds) / "records" / method / f"case_{case['index']:02}.json"


def _run_one(port, seconds, method, case, target, steps=STEPS):
    result = base.queue_and_wait(port, graph(seconds, method, case, target, steps), source.SAMPLER_NODE)
    result.pop("history")
    return result


def run_gpu(gpu):
    port = PORTS[gpu]
    gpu_cases = assigned_cases(gpu)
    rows = []
    for seconds, methods in GENERATED.items():
        for method in methods:
            warm_json = duration_root(seconds) / "warmup" / method / f"gpu{gpu}.json"
            warm_target = duration_root(seconds) / "warmup" / method / f"gpu{gpu}.safetensors"
            if not warm_json.exists() or not warm_target.exists():
                warm = _run_one(port, seconds, method, gpu_cases[0], warm_target, 5)
                base.write_json(warm_json, {
                    **warm, "duration_seconds": seconds, "method": method, "gpu": gpu,
                    "steps": 5, "latent_path": str(warm_target),
                    "latent_sha256": base.sha256(warm_target),
                })
        for case in gpu_cases:
            shift = (case["index"] - 1) % len(methods)
            order = methods[shift:] + methods[:shift]
            for position, method in enumerate(order, 1):
                record_path = _record_path(seconds, method, case)
                target = latent_path(seconds, method, case)
                if record_path.exists() and target.exists():
                    row = read(record_path)
                    if row.get("latent_sha256") == base.sha256(target):
                        rows.append(row)
                        continue
                result = _run_one(port, seconds, method, case, target)
                row = {
                    **result, "duration_seconds": seconds, "method": method, "gpu": gpu,
                    "execution_order": position, "case": case["index"],
                    "sample_id": case["sample_id"], "prompt_sha256": case["prompt_sha256"],
                    "seed": SEED, "steps": STEPS, "width": WIDTH, "height": HEIGHT,
                    "requested_frames": DURATION[seconds]["requested_frames"],
                    "model_frames": DURATION[seconds]["model_frames"],
                    "latent_path": str(target), "latent_sha256": base.sha256(target),
                    "latent_bytes": target.stat().st_size,
                }
                rows.append(row)
                base.write_json(record_path, row)
    return rows


def _external_timing(seconds, method):
    path = BASE5_ROOT if seconds == 5 else BASE10_ROOT
    return read(path / "denoise_summary.json")["methods"][method]


def summarize(rows):
    result = {"status": "complete", "durations": {}}
    for seconds, methods in GENERATED.items():
        section = {"methods": {}, "paired_anchor": []}
        baseline_methods = ("dense", "sol_tau1_extra0")
        for method in baseline_methods:
            if seconds == 10 and method == "dense":
                records = sorted(
                    (row for row in rows if row["duration_seconds"] == seconds and row["method"] == method),
                    key=lambda row: row["case"],
                )
                values = [row["sampler_seconds"] for row in records]
                section["methods"][method] = {
                    "source": "current paired run", "mean_seconds": statistics.fmean(values),
                    "median_seconds": statistics.median(values), "sampler_seconds": values,
                    "records": records,
                }
            else:
                old = _external_timing(seconds, method)
                section["methods"][method] = {
                    "source": str(BASE5_ROOT if seconds == 5 else BASE10_ROOT),
                    "mean_seconds": old["mean_seconds"], "median_seconds": old["median_seconds"],
                    "sampler_seconds": old["sampler_seconds"], "records": old["records"],
                }
        for method in SPARK_METHODS:
            records = sorted(
                (row for row in rows if row["duration_seconds"] == seconds and row["method"] == method),
                key=lambda row: row["case"],
            )
            values = [row["sampler_seconds"] for row in records]
            section["methods"][method] = {
                "source": "current paired run", "mean_seconds": statistics.fmean(values),
                "median_seconds": statistics.median(values), "sampler_seconds": values,
                "records": records,
            }
        by_method = {m: {r["case"]: r["sampler_seconds"] for r in section["methods"][m]["records"]}
                     for m in SPARK_METHODS}
        for case in cases():
            fp32, bf16 = by_method["spark_fp32"][case["index"]], by_method["spark_bf16"][case["index"]]
            section["paired_anchor"].append({
                "case": case["index"], "bf16_minus_fp32_seconds": bf16 - fp32,
                "bf16_speedup_vs_fp32": fp32 / bf16,
            })
        dense = section["methods"]["dense"]["mean_seconds"]
        sol = section["methods"]["sol_tau1_extra0"]["mean_seconds"]
        for method, item in section["methods"].items():
            item["speedup_vs_dense"] = dense / item["mean_seconds"]
            item["speedup_vs_sol"] = sol / item["mean_seconds"]
        result["durations"][f"{seconds}s"] = section
    return result


def denoise():
    if not ROOT.exists():
        prepare()
    base.ROOT, base.OUT = ROOT, OUT
    servers = []
    try:
        for gpu in GPUS:
            process, log = base.start_server(f"gpu{gpu}", gpu, PORTS[gpu], OUT)
            servers.append((f"gpu{gpu}", process, log))
        for gpu in GPUS:
            base.write_json(ROOT / f"system_gpu{gpu}.json", base.wait_server(PORTS[gpu]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rows = [row for group in executor.map(run_gpu, GPUS) for row in group]
        base.write_json(ROOT / "denoise_summary.json", summarize(rows))
    finally:
        base.stop_servers(servers)


def _decode_jobs():
    jobs = []
    for seconds, methods in ((5, SPARK_METHODS), (10, ALL_METHODS)):
        for method in methods:
            for case in cases():
                jobs.append((seconds, method, case))
    return jobs


def _decode_gpu(gpu, jobs):
    rows = []
    for seconds, method, case in jobs:
        target_dir = duration_out(seconds) / method
        target_dir.mkdir(parents=True, exist_ok=True)
        pattern = f"{case['index']:02}_{case['sample_id']}_*.mp4"
        matches = sorted(target_dir.glob(pattern))
        result = {"reused_existing": True}
        if not matches:
            result = base.queue_and_wait(DECODE_PORTS[gpu], decode_graph(seconds, method, case))
            result.pop("history")
            matches = sorted(target_dir.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one decode for {seconds}s/{method}/{pattern}, got {matches}")
        path = matches[0]
        expected = {"frames": DURATION[seconds]["requested_frames"], "width": WIDTH,
                    "height": HEIGHT, "average_rate": FPS}
        info = base.inspect_video(path)
        # The first two resume-safe outputs may include audio from the original
        # graph.  All quality metrics in this experiment are visual-only, so
        # subsequent decodes intentionally skip the unused audio VAE.
        if {key: info[key] for key in expected} != expected or info["audio_streams"] not in (0, 1):
            raise RuntimeError(f"invalid video {path}: {info}")
        row = {
            **result, "duration_seconds": seconds, "method": method,
            "case": case["index"], "sample_id": case["sample_id"],
            "video_path": str(path), "video_sha256": base.sha256(path),
            "video_bytes": path.stat().st_size, "video": info,
        }
        rows.append(row)
        base.write_json(duration_root(seconds) / "decode" / method / f"case_{case['index']:02}.json", row)
    return rows


def _start_decode_server(label, gpu, port):
    """Keep the small decoder VAEs resident on these 97 GB GPUs.

    The generic benchmark launcher enables DynamicVRAM/async offload, which is
    useful for the 20 GB denoiser but needlessly streams the 5 GB video VAE on
    every decode and makes visual-only evaluation several times slower.
    """
    log = (ROOT / f"server_{label}.log").open("a")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        base.PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(port),
        "--disable-auto-launch", "--disable-cuda-malloc", "--preview-method", "none",
        "--highvram", "--disable-async-offload", "--disable-dynamic-vram",
        "--output-directory", str(OUT), "--temp-directory", str(ROOT / "temp" / label),
        "--input-directory", str(ROOT / "input" / label),
        "--user-directory", str(ROOT / "user" / label),
    ]
    process = subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log


def decode():
    base.ROOT, base.OUT = ROOT, OUT
    servers = []
    try:
        for gpu in GPUS:
            process, log = _start_decode_server(f"decoder_gpu{gpu}", gpu, DECODE_PORTS[gpu])
            servers.append((f"decoder_gpu{gpu}", process, log))
        for gpu in GPUS:
            base.wait_server(DECODE_PORTS[gpu])
        jobs = _decode_jobs()
        assigned = [jobs[gpu::len(GPUS)] for gpu in GPUS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            groups = list(executor.map(_decode_gpu, GPUS, assigned))
        base.write_json(ROOT / "decode_summary.json", {
            "status": "complete", "records": [row for group in groups for row in group],
        })
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "denoise", "decode", "all"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "denoise":
        denoise()
    elif args.command == "decode":
        decode()
    else:
        denoise(); decode()
