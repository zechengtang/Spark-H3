#!/usr/bin/env python3
"""Four-case route regression check against archived 10s/768p Spark outputs.

Only the current BF16 threshold arm is generated. Historical threshold and
current packed-external videos already exist and are deliberately not rerun.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import pytorch_spark_route_execution_10prompt_5s768p_20260925 as route

NAME = "diagnose_spark_psnr_regression_4prompt_20260927"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
CASES = (2, 4, 6, 8)  # overlap the historical 20% and recent 10% subsets
METHOD = "current_bf16_threshold"
DENSE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json")
COND = DENSE.parent.parent


def config(_method):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        20, sol_log_density=False, sol_video_tail_mode="dense",
        sol_tail_granularity="query", sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore", sol_reweight_logmass_key="stored",
        sol_route_topk_execution="threshold",
    )


def parse_args_eager(command):
    return route.base.pipeline().build_parser().parse_args([
        command, "--samples", str(route.SAMPLES), "--output", str(OUT),
        "--method", "dense", "--steps", str(route.base.STEPS),
        "--frames", str(route.base.FRAMES), "--height", str(route.base.HEIGHT),
        "--width", str(route.base.WIDTH), "--workers", "1",
        "--no-torch-compile",
    ])


def configure():
    route.ROOT, route.OUT = ROOT, OUT
    route.SAMPLES = route.base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
    route.CASES, route.METHODS, route.GPUS = CASES, (METHOD,), (0, 1, 2, 3)
    route.EXECUTIONS = {METHOD: "threshold"}
    route.DENSE_OUT = ROOT / "dense_reference"
    route.spark_config = config
    route.cases = lambda: route.base.pipeline().load_cases(
        route.SAMPLES, list(CASES), expected_indices=tuple(range(1, 51)))
    route.parse_args = parse_args_eager
    route.base.FRAMES, route.base.INTERNAL_FRAMES = 240, 244


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("diagnostic output already exists")
    selected = route.cases()
    dense = json.loads(DENSE.read_text())
    by_id = {row["sample_id"]: row for row in dense["records"]}
    rows = []
    for case in selected:
        row = dict(by_id[case["sample_id"]])
        if row["prompt_sha256"] != case["prompt_sha256"]:
            raise ValueError(f"dense prompt mismatch: {case['index']}")
        row["index"] = case["index"]
        rows.append(row)
    for folder in (ROOT / "records" / METHOD, ROOT / "warmup" / METHOD,
                   ROOT / "decode_records" / METHOD,
                   OUT / METHOD / "latents", OUT / METHOD / "videos"):
        folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "dense_reference").mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(COND / name, target_is_directory=name == "conditioning_cache")
    route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=4,
        settings=dense.get("settings"), records=rows, source_manifest=str(DENSE)))
    route.base.write(ROOT / "protocol.json", dict(
        purpose="isolate BF16 threshold route versus BF16 packed-external PSNR regression",
        cases=selected, config=dataclasses.asdict(config(METHOD)),
        dense_reference_manifest=str(DENSE), conditioning_source=str(COND),
        seed=42, steps=20, evaluations=19, frames=240, height=768, width=1344,
        torch_compile=False, gpus=list(route.GPUS)))
    route.base.pipeline().configure_denoise_workflow(route.parse_args("denoise"), selected)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(route.GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(route.base.PYTHON), str(Path(__file__).resolve()), stage, str(rank)],
            env={**os.environ, **route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT)
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def quality():
    config_path = ROOT / "quality_config.json"
    route.base.write(config_path, dict(
        work_dir=str(ROOT / "quality"),
        reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
        candidate_manifest=str(OUT / METHOD / "generation_manifest.json"),
        method=METHOD, cases=list(CASES), workers=4))
    subprocess.run([str(route.base.PYTHON), str(route.QUALITY_SCRIPT), "run",
                    "--config", str(config_path)], check=True,
                   env={**os.environ, **route.base.ENV,
                        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "H3_NUM_GPUS": "4"})


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        prepare()
        spawn("generate_worker")
        spawn("decode_worker")
        route.manifests()
        quality()
    elif command in ("generate_worker", "decode_worker"):
        getattr(route, command)(int(sys.argv[2]))
    elif command == "quality":
        quality()
    else:
        raise ValueError(command)
