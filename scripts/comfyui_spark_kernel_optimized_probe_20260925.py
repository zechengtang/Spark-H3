#!/usr/bin/env python3
"""Two-prompt 10s768p probe for the direct-route/reweight Spark kernel."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source


NAME = "comfyui_spark_fanout16_final_probe_2prompt_10s768p_20260925"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
BASELINE = Path(
    "/autodl-fs/data/h3_experiments/"
    "comfyui_sol_tau1_vs_spark_10prompt_10s768p_20260924"
)
GPUS = (0, 1)
PORTS = {0: 8370, 1: 8371}


def graph(case, target, steps):
    value = source.denoise_graph("spark_topk10", case, target, steps)
    value["2"]["inputs"].update(
        ablation_mode="full",
        video_tail_mode="dense",
        global_anchor_dtype="float32",
    )
    value["4"]["inputs"].update(width=1344, height=768, length=240)
    return value


def run_gpu(gpu, case):
    port = PORTS[gpu]
    warm = ROOT / "warmup" / f"gpu{gpu}.safetensors"
    warm_result = base.queue_and_wait(port, graph(case, warm, 5), source.SAMPLER_NODE)
    target = ROOT / "latents" / f"case_{case['index']:02}_{case['sample_id']}.safetensors"
    result = base.queue_and_wait(port, graph(case, target, 20), source.SAMPLER_NODE)
    return {
        "case": case["index"],
        "sample_id": case["sample_id"],
        "gpu": gpu,
        "sampler_seconds": result["sampler_seconds"],
        "warmup_seconds": warm_result["sampler_seconds"],
        "latent_path": str(target),
        "latent_sha256": base.sha256(target),
    }


def main():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    for path in (ROOT / "warmup", ROOT / "latents", OUT):
        path.mkdir(parents=True)
    for gpu in GPUS:
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
    cases = source.cases()[:2]
    baseline = json.loads((BASELINE / "denoise_summary.json").read_text())
    sol_by_case = {
        row["case"]: row["sampler_seconds"]
        for row in baseline["methods"]["sol_tau1_extra0"]["records"]
    }
    base.write_json(ROOT / "protocol.json", {
        "name": NAME,
        "pipeline": "ComfyUI-native MiniMax-H3 denoise-only",
        "settings": {"duration": "10s", "resolution": "1344x768", "steps": 20,
                     "seed": 42, "warmup": "excluded 5-step denoise", "gpus": list(GPUS)},
        "spark": {"topk_ratio": 0.1, "reweight": "global", "reblock_reuse_layers": 1,
                  "route": "official INT8 route + in-kernel exact Top-K",
                  "reweight_transport": "direct complement bitmap",
                  "reblock_transport": "permutation fused into Sol preprocess/output scatter",
                  "reblock_route": "fanout16 exact fused compact route",
                  "landmark_count": 32, "group_size": 1},
        "baseline": str(BASELINE),
        "cases": cases,
    })
    servers = []
    try:
        base.ROOT = ROOT
        for gpu in GPUS:
            process, log = base.start_server(f"gpu{gpu}", gpu, PORTS[gpu], OUT)
            servers.append((f"gpu{gpu}", process, log))
        for gpu in GPUS:
            base.wait_server(PORTS[gpu])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(lambda pair: run_gpu(*pair), zip(GPUS, cases, strict=True)))
        for row in rows:
            row["sol_baseline_seconds"] = sol_by_case[row["case"]]
            row["speedup_vs_sol"] = row["sol_baseline_seconds"] / row["sampler_seconds"]
        base.write_json(ROOT / "results.json", {
            "status": "complete",
            "rows": rows,
            "mean_spark_seconds": sum(r["sampler_seconds"] for r in rows) / len(rows),
            "mean_sol_seconds": sum(r["sol_baseline_seconds"] for r in rows) / len(rows),
            "ratio_of_means": (
                sum(r["sol_baseline_seconds"] for r in rows)
                / sum(r["sampler_seconds"] for r in rows)
            ),
        })
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    main()
