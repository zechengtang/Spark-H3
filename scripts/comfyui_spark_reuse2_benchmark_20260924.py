#!/usr/bin/env python3
"""Generate and time the experimental two-layer reblock reuse mode."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import statistics

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_group841_benchmark_20260924 as paired


NAME = "comfyui_spark_reuse2_10prompt_5s768p_20260924"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
BASELINE_ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_spark_group841_10prompt_5s768p_20260924")
GPUS = paired.GPUS
PORTS = tuple(8310 + gpu for gpu in GPUS)


def target(case):
    return ROOT / "latents" / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def graph(case, path, steps=paired.source.STEPS):
    value = paired.graph("full_spark", case, path, steps)
    value["2"]["inputs"]["ablation_mode"] = "full_reuse2"
    return value


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    (ROOT / "latents").mkdir()
    (ROOT / "records").mkdir()
    (ROOT / "warmup").mkdir()
    for gpu in GPUS:
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True)
    base.write_json(ROOT / "protocol.json", {
        "name": NAME,
        "method": {"ablation_mode": "full_reuse2", "reblock_reuse_layers": 2},
        "baseline": str(BASELINE_ROOT),
        "settings": json.loads((BASELINE_ROOT / "protocol.json").read_text())["settings"],
        "cases": paired.cases(),
    })


def run_gpu(gpu):
    port = PORTS[gpu]
    gpu_cases = paired.assigned_cases(gpu)
    warm = ROOT / "warmup" / f"gpu{gpu}.safetensors"
    result = base.queue_and_wait(port, graph(gpu_cases[0], warm, 5), paired.source.SAMPLER_NODE)
    result.pop("history")
    base.write_json(ROOT / "warmup" / f"gpu{gpu}.json", {
        **result, "gpu": gpu, "latent_sha256": base.sha256(warm),
    })
    rows = []
    for case in gpu_cases:
        path = target(case)
        result = base.queue_and_wait(port, graph(case, path), paired.source.SAMPLER_NODE)
        result.pop("history")
        row = {
            **result, "method": "full_reuse2", "gpu": gpu,
            "case": case["index"], "sample_id": case["sample_id"],
            "latent_path": str(path), "latent_sha256": base.sha256(path),
            "latent_bytes": path.stat().st_size,
        }
        rows.append(row)
        base.write_json(ROOT / "records" / f"case_{case['index']:02}.json", row)
    return rows


def summarize(rows):
    baseline = json.loads((BASELINE_ROOT / "denoise_summary.json").read_text())["methods"]["full_spark"]["records"]
    base_by_case = {row["case"]: row["sampler_seconds"] for row in baseline}
    rows.sort(key=lambda row: row["case"])
    values = [row["sampler_seconds"] for row in rows]
    paired_rows = [{
        "case": row["case"],
        "reuse2_minus_baseline_seconds": row["sampler_seconds"] - base_by_case[row["case"]],
        "speedup": base_by_case[row["case"]] / row["sampler_seconds"],
    } for row in rows]
    return {
        "status": "complete", "records": rows,
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "paired_to_previous_baseline": paired_rows,
    }


def main():
    prepare()
    base.ROOT, base.OUT = ROOT, OUT
    servers = []
    try:
        for gpu, port in zip(GPUS, PORTS, strict=True):
            process, log = base.start_server(f"gpu{gpu}", gpu, port, OUT)
            servers.append((f"gpu{gpu}", process, log))
        for port in PORTS:
            base.wait_server(port)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            rows = [row for group in executor.map(run_gpu, GPUS) for row in group]
        base.write_json(ROOT / "denoise_summary.json", summarize(rows))
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    main()
