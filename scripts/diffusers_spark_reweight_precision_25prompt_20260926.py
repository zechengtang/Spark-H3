#!/usr/bin/env python3
"""Paired 25-prompt Diffusers Spark global-reweight precision ablation.

Reuses the historical 10s/768p dense videos and text conditioning. VBench is
deliberately omitted: this experiment isolates numeric precision.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytorch_spark_route_execution_10prompt_5s768p_20260925 as route


NAME = "diffusers_spark_reweight_precision_25prompt_10s768p_20260926"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = route.base.BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
SOURCE_DENSE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json")
SOURCE_COND = SOURCE_DENSE.parent.parent
METHODS = ("spark_bf16", "spark_anchor_fp32", "spark_comfy_fp32")
GPUS = tuple(range(6))
CASES = tuple(range(1, 26))


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig

    extra = {}
    if method in ("spark_anchor_fp32", "spark_comfy_fp32"):
        extra["sol_global_anchor_dtype"] = "float32"
    if method == "spark_comfy_fp32":
        extra.update(sol_reweight_summary_math="comfy_fp32",
                     sol_reweight_logmass_key="pre_round")
    return H3SparseAttentionConfig.spark(
        20, sol_log_density=False, sol_video_tail_mode="dense",
        sol_tail_granularity="query", **extra)


def configure():
    route.ROOT, route.OUT = ROOT, OUT
    route.SAMPLES, route.CASES = SAMPLES, CASES
    route.METHODS, route.GPUS = METHODS, GPUS
    route.EXECUTIONS = {
        method: "packed_external_no_route_qk" for method in METHODS
    }
    route.DENSE_OUT = ROOT / "dense_reference"
    route.spark_config = config
    route.cases = cases
    route.base.FRAMES = 240
    route.base.INTERNAL_FRAMES = 244


def cases():
    return route.base.pipeline().load_cases(
        SAMPLES, list(CASES), expected_indices=CASES)


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("experiment/output directory exists")
    selected = cases()
    dense = json.loads(SOURCE_DENSE.read_text())
    by_id = {item["sample_id"]: item for item in dense["records"]}
    matched = []
    for case in selected:
        item = dict(by_id[case["sample_id"]])
        assert item["prompt_sha256"] == case["prompt_sha256"]
        assert Path(item["output_path"]).is_file()
        item["index"] = case["index"]
        matched.append(item)
    for method in METHODS:
        for folder in (ROOT / "records" / method, ROOT / "warmup" / method,
                       ROOT / "decode_records" / method,
                       OUT / method / "latents", OUT / method / "videos"):
            folder.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(SOURCE_COND / name,
                               target_is_directory=name == "conditioning_cache")
    route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=25,
        settings=dense.get("settings"), records=matched,
        source_manifest=str(SOURCE_DENSE)))
    route.base.write(ROOT / "protocol.json", dict(
        name=NAME, status="prepared", pipeline="native PyTorch/Diffusers",
        purpose="25-prompt BF16 anchor vs FP32 anchor vs Comfy-style FP32 summary numeric ablation",
        dataset="VBench core-five 10pct subset, first 25 prompts",
        cases=selected, methods=METHODS,
        configs={m: dataclasses.asdict(config(m)) for m in METHODS},
        dense_reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
        conditioning_source=str(SOURCE_COND),
        seed=42, requested_steps=20, actual_evaluations=19,
        torch_compile=True,
        frames=240, resolution="1344x768", gpus=GPUS,
        warmup="one excluded full denoise per method/GPU",
        timing="synchronized denoise, excludes model load, conditioning, VAE, save",
        pairing="same prompt and GPU with rotating arm order",
        metrics="PSNR, SSIM, LPIPS against reused dense; no VBench",
        caveat="comfy_fp32 matches precision stages, not bitwise CUDA reduction order",
    ))
    route.base.pipeline().configure_denoise_workflow(route.parse_args("denoise"), selected)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(GPUS):
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


def aggregate():
    path = ROOT / "results.json"
    data = {"status": "runtime_complete", "comparisons": {
        "anchor_fp32_vs_bf16": "spark_anchor_fp32 vs spark_bf16",
        "comfy_fp32_vs_anchor_fp32": "spark_comfy_fp32 vs spark_anchor_fp32",
        "comfy_fp32_vs_bf16": "spark_comfy_fp32 vs spark_bf16",
    }}
    records = {method: [route.base.read(ROOT / "records" / method / f"case_{case:02}.json")
                        for case in CASES] for method in METHODS}
    data["summary"] = {}
    for method, rows in records.items():
        values = [row["denoise_seconds"] for row in rows]
        data["summary"][method] = dict(
            mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
            stdev_seconds=statistics.stdev(values), min_seconds=min(values),
            max_seconds=max(values),
            mean_peak_memory_gib=statistics.mean(row["peak_memory_bytes"] / 2**30
                                            for row in rows), values=values)
    data["paired_runtime"] = {}
    for name, reference, candidate in (
        ("anchor_fp32_vs_bf16", METHODS[0], METHODS[1]),
        ("comfy_fp32_vs_anchor_fp32", METHODS[1], METHODS[2]),
        ("comfy_fp32_vs_bf16", METHODS[0], METHODS[2]),
    ):
        deltas = [b["denoise_seconds"] - a["denoise_seconds"]
                  for a, b in zip(records[reference], records[candidate], strict=True)]
        data["paired_runtime"][name] = dict(
            versus=reference, candidate=candidate,
            mean_delta_seconds=statistics.mean(deltas),
            faster_prompts=sum(delta < 0 for delta in deltas), values=deltas)
    route.base.write(path, data)


def run():
    configure()
    if not ROOT.exists():
        prepare()
    spawn("generate_worker")
    aggregate()
    spawn("decode_worker")
    route.manifests()
    quality()


def quality():
    configure()
    summaries = {}
    for method in METHODS:
        work = ROOT / "quality" / method
        config_path = ROOT / "quality" / f"{method}_config.json"
        route.base.write(config_path, dict(
            work_dir=str(work),
            reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
            candidate_manifest=str(OUT / method / "generation_manifest.json"),
            method=method, cases=list(CASES), workers=6,
        ))
        subprocess.run([str(route.base.PYTHON), str(route.QUALITY_SCRIPT), "run",
                        "--config", str(config_path)], check=True,
                       env={**os.environ, **route.base.ENV,
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5", "H3_NUM_GPUS": "6"})
        summaries[method] = json.loads(
            (work / f"{method}_quality_results.json").read_text())["summary"]
    results = json.loads((ROOT / "results.json").read_text())
    results.update(quality=summaries, status="quality_complete")
    route.base.write(ROOT / "results.json", results)
    route.base.write(ROOT / "quality_results.json", dict(
        status="complete", reference="dense", cases=list(CASES), summaries=summaries))


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        getattr(route, command)(int(sys.argv[2]))
    elif command in ("run", "prepare", "aggregate", "quality"):
        globals()[command]()
    elif command == "manifests":
        route.manifests()
    else:
        raise ValueError(command)
