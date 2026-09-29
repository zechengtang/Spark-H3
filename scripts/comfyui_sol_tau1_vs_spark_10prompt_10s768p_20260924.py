#!/usr/bin/env python3
"""Paired ComfyUI speed benchmark: official Sol tau=1 vs Spark TopK10."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import statistics

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source


NAME = "comfyui_sol_tau1_vs_spark_global10_10prompt_10s768p_20260924"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHODS = ("sol_tau1_extra0", "spark_topk10")
GPUS = (0, 1, 2, 3)
PORTS = tuple(8320 + gpu for gpu in GPUS)
WIDTH, HEIGHT, REQUESTED_FRAMES, MODEL_FRAMES = 1344, 768, 240, 243
STEPS, SEED = 20, 42


def cases():
    return source.cases()


def latent_path(method, case):
    return ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def graph(method, case, target, steps=STEPS):
    value = source.denoise_graph(method, case, target, steps)
    value["4"]["inputs"].update(width=WIDTH, height=HEIGHT, length=REQUESTED_FRAMES)
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
        "purpose": "paired ComfyUI denoise speed: official Sol tau=1 extra0 vs published PyTorch-aligned Spark-H3-10pct",
        "methods": {
            "sol_tau1_extra0": {
                "node": "BlockSparseAttention", "tau": 1.0,
                "extra_tokens": 0, "sink_conditioning": "exact_kv_and_rows",
            },
            "spark_topk10": {
                "node": "MiniMaxH3SparkAttentionSM120", "topk_ratio": 0.1,
                "ablation_mode": "full", "video_tail_mode": "dense",
                "reweight": "global single video representative per head",
                "sol_virtual_query_levels_up": 99,
                "sol_virtual_query_target_blocks": None,
            },
        },
        "shared_sparse": {
            "dense_evaluations": 4, "dense_layers": [0], "min_tokens": 4096,
        },
        "settings": {
            "seed": SEED, "steps": STEPS, "sampler": "res_multistep",
            "scheduler": "simple", "width": WIDTH, "height": HEIGHT,
            "requested_frames": REQUESTED_FRAMES,
            "model_aligned_frames": MODEL_FRAMES, "fps": 24.0,
            "duration_seconds": 10.0, "turbo_lora": False,
            "timing": "SamplerCustomAdvanced only; conditioning/VAE excluded",
            "pairing": "both methods for each prompt run on the same GPU; order alternates",
            "warmup": "one excluded 5-step full-resolution run per method on every GPU",
        },
        "cases": cases(),
        "conditioning_source": str(source.EMBEDDINGS),
        "revisions": {
            "comfyui": base.git_revision(source.COMFY),
            "spark_h3": base.git_revision(source.REPO),
        },
    })


def assigned_cases(gpu):
    return [case for case in cases() if (case["index"] - 1) % len(GPUS) == gpu]


def run_gpu(gpu):
    port = PORTS[gpu]
    gpu_cases = assigned_cases(gpu)
    for method in METHODS:
        target = ROOT / "warmup" / method / f"gpu{gpu}.safetensors"
        result = base.queue_and_wait(port, graph(method, gpu_cases[0], target, 5), source.SAMPLER_NODE)
        result.pop("history")
        base.write_json(ROOT / "warmup" / method / f"gpu{gpu}.json", {
            **result, "method": method, "gpu": gpu, "steps": 5,
            "latent_path": str(target), "latent_sha256": base.sha256(target),
        })
    rows = []
    for case in gpu_cases:
        order = METHODS if case["index"] % 2 else tuple(reversed(METHODS))
        for position, method in enumerate(order, 1):
            target = latent_path(method, case)
            result = base.queue_and_wait(port, graph(method, case, target), source.SAMPLER_NODE)
            result.pop("history")
            row = {
                **result, "method": method, "gpu": gpu, "execution_order": position,
                "case": case["index"], "sample_id": case["sample_id"],
                "seed": SEED, "steps": STEPS, "width": WIDTH, "height": HEIGHT,
                "requested_frames": REQUESTED_FRAMES, "model_frames": MODEL_FRAMES,
                "latent_path": str(target), "latent_sha256": base.sha256(target),
                "latent_bytes": target.stat().st_size,
            }
            rows.append(row)
            base.write_json(ROOT / "records" / method / f"case_{case['index']:02}.json", row)
    return rows


def summarize(rows):
    result = {"status": "complete", "methods": {}, "paired": []}
    for method in METHODS:
        records = sorted((row for row in rows if row["method"] == method), key=lambda row: row["case"])
        values = [row["sampler_seconds"] for row in records]
        result["methods"][method] = {
            "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "sampler_seconds": values, "records": records,
        }
    for case in cases():
        values = {row["method"]: row["sampler_seconds"] for row in rows if row["case"] == case["index"]}
        result["paired"].append({
            "case": case["index"],
            "spark_minus_sol_seconds": values["spark_topk10"] - values["sol_tau1_extra0"],
            "spark_speedup_vs_sol": values["sol_tau1_extra0"] / values["spark_topk10"],
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
            rows = [row for group in executor.map(run_gpu, GPUS) for row in group]
        base.write_json(ROOT / "denoise_summary.json", summarize(rows))
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    main()
