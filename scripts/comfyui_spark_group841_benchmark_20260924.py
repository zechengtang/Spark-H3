#!/usr/bin/env python3
"""Paired ComfyUI benchmark for baseline Spark and group-(8,4,1) reblocking."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import statistics

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source


NAME = "comfyui_spark_group841_10prompt_5s768p_20260924"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHODS = ("full_spark", "full_group841")
MODE = {"full_spark": "full", "full_group841": "full_group841"}
GPUS = (0, 1, 2, 3)
PORTS = tuple(8300 + gpu for gpu in GPUS)


def cases():
    return source.cases()


def latent_path(method, case):
    return ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def graph(method, case, target, steps=source.STEPS):
    value = source.denoise_graph("spark_topk10", case, target, steps)
    value["2"]["inputs"]["ablation_mode"] = MODE[method]
    return value


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for method in METHODS:
        for section in ("latents", "records", "warmup"):
            (ROOT / section / method).mkdir(parents=True)
        (OUT / method).mkdir(parents=True)
    for gpu in GPUS:
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True)
    base.write_json(ROOT / "protocol.json", {
        "name": NAME,
        "purpose": "paired validation of hierarchical (8,4,1) reblock grouping",
        "methods": {
            "full_spark": {"ablation_mode": "full", "group_size": 1},
            "full_group841": {"ablation_mode": "full_group841", "group_size": [8, 4, 1]},
        },
        "settings": {
            "seed": source.SEED, "steps": source.STEPS,
            "width": source.WIDTH, "height": source.HEIGHT,
            "requested_frames": source.REQUESTED_FRAMES,
            "duration_seconds": 5.0, "topk_ratio": 0.1,
            "warmup_percent": 20.0, "dense_layers": [0],
            "pairing": "both methods for each prompt run on the same GPU; order alternates",
            "timing": "SamplerCustomAdvanced only",
        },
        "cases": cases(),
        "revisions": {
            "comfyui": base.git_revision(source.COMFY),
            "spark_h3": base.git_revision(source.REPO),
        },
    })


def assigned_cases(gpu):
    return [case for case in cases() if (case["index"] - 1) % len(GPUS) == gpu]


def run_gpu(gpu):
    gpu_cases = assigned_cases(gpu)
    port = PORTS[gpu]
    for method in METHODS:
        target = ROOT / "warmup" / method / f"gpu{gpu}.safetensors"
        result = base.queue_and_wait(port, graph(method, gpu_cases[0], target, 5), source.SAMPLER_NODE)
        result.pop("history")
        base.write_json(ROOT / "warmup" / method / f"gpu{gpu}.json", {
            **result, "method": method, "gpu": gpu, "steps": 5,
            "latent_path": str(target), "latent_sha256": base.sha256(target),
        })
    records = []
    for case in gpu_cases:
        order = METHODS if case["index"] % 2 else tuple(reversed(METHODS))
        for position, method in enumerate(order, 1):
            target = latent_path(method, case)
            result = base.queue_and_wait(port, graph(method, case, target), source.SAMPLER_NODE)
            result.pop("history")
            row = {
                **result, "method": method, "gpu": gpu, "execution_order": position,
                "case": case["index"], "sample_id": case["sample_id"],
                "seed": source.SEED, "steps": source.STEPS,
                "latent_path": str(target), "latent_sha256": base.sha256(target),
                "latent_bytes": target.stat().st_size,
            }
            records.append(row)
            base.write_json(ROOT / "records" / method / f"case_{case['index']:02}.json", row)
    return records


def summarize(records):
    result = {"status": "complete", "methods": {}, "paired": []}
    for method in METHODS:
        rows = sorted((row for row in records if row["method"] == method), key=lambda row: row["case"])
        values = [row["sampler_seconds"] for row in rows]
        result["methods"][method] = {
            "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "sampler_seconds": values,
            "records": rows,
        }
    for case in cases():
        values = {row["method"]: row["sampler_seconds"] for row in records if row["case"] == case["index"]}
        result["paired"].append({
            "case": case["index"],
            "optimized_minus_baseline_seconds": values["full_group841"] - values["full_spark"],
            "speedup": values["full_spark"] / values["full_group841"],
        })
    return result


def main():
    prepare()
    base.ROOT, base.OUT = ROOT, OUT
    servers = []
    try:
        for gpu, port in zip(GPUS, PORTS, strict=True):
            process, log = base.start_server(f"gpu{gpu}", gpu, port, OUT)
            servers.append((f"gpu{gpu}", process, log))
        for gpu, port in zip(GPUS, PORTS, strict=True):
            base.write_json(ROOT / f"system_gpu{gpu}.json", base.wait_server(port))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            records = [row for group in executor.map(run_gpu, GPUS) for row in group]
        base.write_json(ROOT / "denoise_summary.json", summarize(records))
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    main()
