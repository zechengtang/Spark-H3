"""Background speed retest after one full sparse warmup for each shape/mode.

The 8-step warmup runs through the first sparse step and is excluded from all
reported sampler timings. Each measured seed is shared by all three methods.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import comfyui_vdn8_four_model_72_run_20260929 as original


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_lightx2v_sparse_warm_speed_20260929")
COMFY = original.COMFY
PYTHON = original.PYTHON
GPU = 1
PORT = 8615
MODEL = "lightx2v"
METHODS = ("official_sol", "spark_10pct", "spark_114blocks")
DURATIONS = ("10s", "14p4s")
MEASURED_SEEDS = (42, 43, 44)


def write(path: Path, value: object) -> None:
    original.write(path, value)


def run_case(port: int, case: dict, method: str, *, seed: int, phase: str) -> dict:
    target = ROOT / "latents" / f"{case['label']}_{method}_{phase}_seed{seed}.safetensors"
    graph = original.denoise_graph(MODEL, method, case, target, steps=8)
    graph["7"]["inputs"]["noise_seed"] = seed
    result = original.execute(port, graph, "11")
    if not target.is_file() or result.get("sampler_seconds") is None:
        raise RuntimeError(f"missing latent or sampler time: {case['label']} {method} {phase}")
    record = {
        "status": "complete", "duration": case["label"], "frames": case["frames"],
        "method": method, "phase": phase, "seed": seed, "steps": 8,
        "sampler_seconds": result["sampler_seconds"],
        "workflow_seconds": result["workflow_seconds"],
        "prompt_id": result["prompt_id"],
    }
    write(ROOT / "records" / f"{case['label']}_{method}_{phase}_seed{seed}.json", record)
    target.unlink()
    print(f"{phase.upper()} {case['label']} {method} seed={seed} sampler={record['sampler_seconds']:.3f}s", flush=True)
    return record


def start_server() -> tuple[subprocess.Popen, object]:
    for name in ("user", "input", "temp", "latents", "records"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    log = (ROOT / "comfyui.log").open("a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "HF_HUB_OFFLINE": "1",
           "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost",
           "no_proxy": "127.0.0.1,localhost"}
    command = [PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
               "--disable-auto-launch", "--disable-cuda-malloc", "--preview-method", "none",
               "--output-directory", str(ROOT / "output"),
               "--temp-directory", str(ROOT / "temp"),
               "--input-directory", str(ROOT / "input"),
               "--user-directory", str(ROOT / "user")]
    return subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT), log


def summarize(records: list[dict]) -> dict:
    summary = {}
    for duration in DURATIONS:
        if not any(r["duration"] == duration for r in records):
            continue
        summary[duration] = {}
        for method in METHODS:
            values = [r["sampler_seconds"] for r in records
                      if r["duration"] == duration and r["method"] == method]
            summary[duration][method] = {
                "seconds": values, "median_seconds": statistics.median(values),
                "mean_seconds": statistics.mean(values),
                "min_seconds": min(values), "max_seconds": max(values),
            }
        sol = summary[duration]["official_sol"]["median_seconds"]
        for method in METHODS:
            summary[duration][method]["speedup_vs_sol_median"] = (
                sol / summary[duration][method]["median_seconds"]
            )
    return summary


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = {case["label"]: case for case in original.cases()}
    write(ROOT / "protocol.json", {
        "status": "running", "started_unix": time.time(), "model": MODEL, "gpu": GPU,
        "port": PORT, "steps": 8, "methods": METHODS, "durations": DURATIONS,
        "measured_seeds": MEASURED_SEEDS,
        "warmup": "5s two-step dense weight load, then one uncounted full 8-step run per method and duration, including first sparse step",
        "cases": {d: {"frames": cases[d]["frames"], "workflow": cases[d]["workflow"],
                       "workflow_sha256": cases[d]["workflow_sha256"]} for d in DURATIONS},
        "settings": {"sol": {"tau": 1.3, "start_percent": .2, "min_tokens": 12288},
                     "spark": {"warmup_ratio": 0.2, "dense_layers": 1,
                               "min_tokens": 12288, "topk_ratio": .1,
                               "topk_blocks": 114}},
    })
    process, log = start_server()
    records = []
    try:
        original.api.wait_server(PORT)
        print(f"SERVER READY gpu={GPU} port={PORT}", flush=True)
        first_case = original.cases()[0]
        target = ROOT / "latents" / "weight_load_5s_dense_2step.safetensors"
        if not target.is_file():
            result = original.execute(PORT, original.denoise_graph(MODEL, "dense", first_case, target, steps=2), "11")
            write(ROOT / "weight_load.json", {"status": "complete", "seconds": result["sampler_seconds"]})
            target.unlink()
        print("WEIGHT LOAD COMPLETE", flush=True)

        for duration in DURATIONS:
            case = cases[duration]
            # Complete the first sparse step for every mode at this exact shape.
            for method in METHODS:
                run_case(PORT, case, method, seed=41, phase="warmup")
            print(f"SPARSE WARMUP COMPLETE {duration}", flush=True)
            for round_index, seed in enumerate(MEASURED_SEEDS):
                order = METHODS[round_index:] + METHODS[:round_index]
                for method in order:
                    records.append(run_case(PORT, case, method, seed=seed, phase="timed"))
            write(ROOT / "partial_results.json", {"status": "running", "summary": summarize(records)})
        summary = summarize(records)
        write(ROOT / "results.json", {"status": "complete", "summary": summary,
                                      "records": records, "completed_unix": time.time()})
        lines = ["# LightX2V 10s / 14.4s sparse warmup speed retest", "",
                 "Each method and duration received one uncounted 8-step warmup containing its first sparse step. "
                 "Three measured runs used seeds 42, 43, and 44; method order rotated. "
                 "Sampler timings exclude model load, prompt encoding, and latent save.", "",
                 "| Duration | Method | Runs (s) | Median (s) | Mean (s) | Speed vs Sol (median) |",
                 "|---|---|---|---:|---:|---:|"]
        for duration in DURATIONS:
            for method in METHODS:
                s = summary[duration][method]
                lines.append(f"| {duration.replace('14p4s','14.4s / 345f')} | {method} | "
                             f"{', '.join(f'{x:.2f}' for x in s['seconds'])} | "
                             f"{s['median_seconds']:.2f} | {s['mean_seconds']:.2f} | "
                             f"{s['speedup_vs_sol_median']:.3f}× |")
        (ROOT / "report.md").write_text("\n".join(lines) + "\n")
        print("ALL COMPLETE", flush=True)
    except BaseException as error:
        write(ROOT / "failure.json", {"status": "failed", "type": type(error).__name__,
                                      "error": str(error), "time_unix": time.time()})
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
