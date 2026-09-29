#!/usr/bin/env python3
"""Four-prompt current-kernel timing probe at 5s/10s 768p on GPUs 0-3."""
from __future__ import annotations

import ast
import concurrent.futures
import json
import os
from pathlib import Path
import re
import statistics
import time

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source


NAME = "comfyui_spark_operation_profile_4prompt_5s10s768p_20260925"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
GPUS = (0, 1, 2, 3)
PORTS = {gpu: 8420 + gpu for gpu in GPUS}
DURATIONS = {5: (120, 123), 10: (240, 243)}
STEPS = 20
PROFILE_RE = re.compile(r"reblock CUDA profile: (\{.*\})")


def cases():
    return source.cases()[:4]


def graph(seconds, case, target, steps):
    value = source.denoise_graph("spark_topk10", case, target, steps)
    value["2"]["inputs"].update(
        ablation_mode="full",
        video_tail_mode="dense",
        global_anchor_dtype="float32",
    )
    value["4"]["inputs"].update(
        width=1344, height=768, length=DURATIONS[seconds][0]
    )
    return value


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    for path in (ROOT, OUT):
        path.mkdir(parents=True)
    for seconds in DURATIONS:
        for section in ("warmup", "latents", "records"):
            (ROOT / f"{seconds}s" / section).mkdir(parents=True)
    for gpu in GPUS:
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True)
    base.write_json(ROOT / "protocol.json", {
        "name": NAME,
        "purpose": "Current ComfyUI Spark operation timing at both H3 768p lengths",
        "pipeline": "ComfyUI-native denoise-only with reused BF16 conditioning",
        "settings": {
            "durations": DURATIONS, "width": 1344, "height": 768,
            "seed": 42, "steps": STEPS, "gpus": list(GPUS),
            "warmup": "excluded 5-step full-resolution run per duration/GPU",
            "profile": "CUDA events for reblock plan and fused sparse attention",
        },
        "spark": {
            "topk_ratio": 0.1, "fanout": 16, "reweight": "global",
            "anchor_dtype": "float32", "video_tail_mode": "dense",
            "reblock_reuse_layers": 1,
        },
        "cases": cases(),
    })


def wait_for_profile(log_path: Path, old_count: int, timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        matches = PROFILE_RE.findall(text)
        if len(matches) > old_count:
            return ast.literal_eval(matches[-1]), len(matches)
        time.sleep(0.2)
    raise TimeoutError(f"profile line did not appear in {log_path}")


def run_gpu(gpu, case):
    port = PORTS[gpu]
    log_path = ROOT / f"server_gpu{gpu}.log"
    profile_count = 0
    rows = []
    for seconds in DURATIONS:
        warm_target = ROOT / f"{seconds}s" / "warmup" / f"gpu{gpu}.safetensors"
        base.queue_and_wait(port, graph(seconds, case, warm_target, 5), source.SAMPLER_NODE)
        _, profile_count = wait_for_profile(log_path, profile_count)

        target = ROOT / f"{seconds}s" / "latents" / f"case_{case['index']:02}.safetensors"
        result = base.queue_and_wait(port, graph(seconds, case, target, STEPS), source.SAMPLER_NODE)
        profile, profile_count = wait_for_profile(log_path, profile_count)
        result.pop("history")
        row = {
            **result,
            "duration_seconds": seconds,
            "gpu": gpu,
            "case": case["index"],
            "sample_id": case["sample_id"],
            "prompt_sha256": case["prompt_sha256"],
            "requested_frames": DURATIONS[seconds][0],
            "model_frames": DURATIONS[seconds][1],
            "profile": profile,
            "latent_path": str(target),
            "latent_sha256": base.sha256(target),
        }
        rows.append(row)
        base.write_json(ROOT / f"{seconds}s" / "records" / f"case_{case['index']:02}.json", row)
    return rows


def summarize(rows):
    result = {"status": "complete", "durations": {}}
    for seconds in DURATIONS:
        selected = [row for row in rows if row["duration_seconds"] == seconds]
        operations = {}
        names = sorted({name for row in selected for name in row["profile"]})
        for name in names:
            per_call = [row["profile"][name]["mean_ms"] for row in selected]
            totals = [row["profile"][name]["total_ms"] for row in selected]
            operations[name] = {
                "mean_ms_per_call": statistics.fmean(per_call),
                "median_of_prompt_means_ms": statistics.median(per_call),
                "mean_total_ms_per_prompt": statistics.fmean(totals),
                "prompt_values_ms_per_call": per_call,
            }
        sampler = [row["sampler_seconds"] for row in selected]
        result["durations"][f"{seconds}s"] = {
            "mean_sampler_seconds": statistics.fmean(sampler),
            "sampler_seconds": sampler,
            "operations": operations,
            "records": selected,
        }
    return result


def main():
    prepare()
    base.ROOT, base.OUT = ROOT, OUT
    os.environ["SPARK_PROFILE_REBLOCK"] = "1"
    servers = []
    try:
        for gpu in GPUS:
            process, log = base.start_server(f"gpu{gpu}", gpu, PORTS[gpu], OUT)
            servers.append((f"gpu{gpu}", process, log))
        for gpu in GPUS:
            base.write_json(ROOT / f"system_gpu{gpu}.json", base.wait_server(PORTS[gpu]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            groups = list(pool.map(lambda x: run_gpu(*x), zip(GPUS, cases(), strict=True)))
        rows = [row for group in groups for row in group]
        base.write_json(ROOT / "results.json", summarize(rows))
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    main()
