#!/usr/bin/env python3
"""Ten-prompt, four-arm Diffusers global-reweight component ablation.

The full-reweight arm is reused from the completed 25-prompt Comfy-precision
ablation. The other three arms are generated here with identical prompts,
conditioning, dense references, seed, and FP32 summary arithmetic.
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
import diffusers_spark_reweight_precision_25prompt_20260926 as source


NAME = "diffusers_spark_reweight_components_10prompt_10s768p_20260926"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SOURCE_ROOT = source.ROOT
SOURCE_OUT = source.OUT
METHODS = ("weights_only", "bias_only", "none")
ALL_METHODS = ("full",) + METHODS
GPUS = tuple(range(6))
DECODE_GPUS = tuple(range(4))
CASES = tuple(range(1, 11))


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig

    if method not in ALL_METHODS:
        raise ValueError(method)
    return H3SparseAttentionConfig.spark(
        20, sol_log_density=False, sol_video_tail_mode="dense",
        sol_tail_granularity="query", sol_global_anchor_dtype="float32",
        sol_reweight_summary_math="comfy_fp32",
        sol_reweight_logmass_key="pre_round",
        sol_reweight_components=method,
    )


def configure():
    route.ROOT, route.OUT = ROOT, OUT
    route.SAMPLES, route.CASES = source.SAMPLES, CASES
    route.METHODS, route.GPUS = METHODS, GPUS
    route.EXECUTIONS = {method: "packed_external" for method in METHODS}
    route.DENSE_OUT = ROOT / "dense_reference"
    route.spark_config = config
    route.cases = cases
    route.base.FRAMES = 240
    route.base.INTERNAL_FRAMES = 244


def cases():
    return route.base.pipeline().load_cases(
        source.SAMPLES, list(CASES), expected_indices=source.CASES)


def _subset_manifest(source_path, destination, selected):
    data = json.loads(Path(source_path).read_text())
    by_id = {row["sample_id"]: row for row in data["records"]}
    rows = []
    for case in selected:
        row = dict(by_id[case["sample_id"]])
        assert row["prompt_sha256"] == case["prompt_sha256"]
        assert Path(row["output_path"]).is_file()
        row["index"] = case["index"]
        rows.append(row)
    route.base.write(destination, dict(
        schema_version=1, status="passed", method=data.get("method"),
        sample_count=len(rows), settings=data.get("settings"), records=rows,
        source_manifest=str(source_path),
    ))


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    source_protocol = json.loads((SOURCE_ROOT / "protocol.json").read_text())
    if source_protocol.get("torch_compile") is not True:
        raise ValueError("cannot reuse full arm generated without torch.compile")
    source_results = json.loads((SOURCE_ROOT / "results.json").read_text())
    if source_results.get("status") != "quality_complete":
        raise RuntimeError("the 25-prompt precision experiment has not completed")
    selected = cases()
    full_source = SOURCE_OUT / "spark_comfy_fp32" / "generation_manifest.json"
    dense_source = SOURCE_ROOT / "dense_reference" / "generation_manifest.json"
    for method in METHODS:
        for folder in (ROOT / "records" / method, ROOT / "warmup" / method,
                       ROOT / "decode_records" / method,
                       OUT / method / "latents", OUT / method / "videos"):
            folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "full_reference").mkdir(parents=True, exist_ok=True)
    (ROOT / "dense_reference").mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(SOURCE_OUT / name,
                               target_is_directory=name == "conditioning_cache")
    _subset_manifest(full_source, ROOT / "full_reference" / "generation_manifest.json", selected)
    _subset_manifest(dense_source, ROOT / "dense_reference" / "generation_manifest.json", selected)
    route.base.write(ROOT / "protocol.json", dict(
        name=NAME, status="prepared", pipeline="native PyTorch/Diffusers",
        purpose="Global-reweight weighted-KV versus log-mass component ablation",
        cases=selected, methods=ALL_METHODS,
        configs={method: dataclasses.asdict(config(method)) for method in ALL_METHODS},
        reused_full_manifest=str(full_source), reused_dense_manifest=str(dense_source),
        conditioning_source=str(SOURCE_OUT), seed=42, requested_steps=20,
        torch_compile=True,
        actual_evaluations=19, frames=240, resolution="1344x768", gpus=GPUS,
        warmup="one excluded full denoise per method/GPU",
        timing="synchronized denoise; full-arm timing reused from prior run",
        metrics="PSNR, SSIM, LPIPS against the same dense reference; no VBench",
        component_definition={
            "weights_only": "anchor-softmax weighted K/V; log(block length) mass",
            "bias_only": "ordinary mean K/V; logsumexp(anchor.K)-anchor.meanK mass",
            "none": "ordinary mean K/V; log(block length) mass",
            "full": "anchor-softmax weighted K/V and corresponding log-mass",
        },
    ))
    route.base.pipeline().configure_denoise_workflow(route.parse_args("denoise"), selected)


def spawn(stage, *, gpus=GPUS):
    jobs = []
    for rank, gpu in enumerate(gpus):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(route.base.PYTHON), str(Path(__file__).resolve()), stage, str(rank)],
            env={**os.environ, **route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def aggregate():
    rows = {method: [route.base.read(ROOT / "records" / method / f"case_{case:02}.json")
                     for case in CASES] for method in METHODS}
    rows["full"] = [route.base.read(SOURCE_ROOT / "records" / "spark_comfy_fp32" /
                                    f"case_{case:02}.json") for case in CASES]
    summary = {}
    for method, records in rows.items():
        seconds = [record["denoise_seconds"] for record in records]
        summary[method] = dict(mean_seconds=statistics.mean(seconds),
                               median_seconds=statistics.median(seconds), values=seconds,
                               timing_reused=method == "full")
    route.base.write(ROOT / "results.json", dict(status="runtime_complete", summary=summary,
        caveat="full-arm timing is from the prior 25-prompt run; do not treat cross-run deltas as paired speedups"))


def quality():
    configure()
    summaries = {}
    for method in ALL_METHODS:
        work = ROOT / "quality" / method
        config_path = ROOT / "quality" / f"{method}_config.json"
        candidate = (ROOT / "full_reference" / "generation_manifest.json" if method == "full"
                     else OUT / method / "generation_manifest.json")
        route.base.write(config_path, dict(
            work_dir=str(work),
            reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
            candidate_manifest=str(candidate), method=method,
            cases=list(CASES), workers=len(DECODE_GPUS),
        ))
        subprocess.run([str(route.base.PYTHON), str(route.QUALITY_SCRIPT), "run",
                        "--config", str(config_path)], check=True,
                       env={**os.environ, **route.base.ENV,
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3", "H3_NUM_GPUS": "4"})
        summaries[method] = json.loads(
            (work / f"{method}_quality_results.json").read_text())["summary"]
    results = json.loads((ROOT / "results.json").read_text())
    results.update(quality=summaries, status="quality_complete")
    route.base.write(ROOT / "results.json", results)


def run():
    configure()
    if not ROOT.exists():
        prepare()
    spawn("generate_worker")
    aggregate()
    route.GPUS = DECODE_GPUS
    spawn("decode_worker", gpus=DECODE_GPUS)
    route.manifests()
    quality()


def resume_decode():
    """Resume completed generation on GPU0-3 without repeating denoising."""
    configure()
    missing = [ROOT / "records" / method / f"case_{case:02}.json"
               for method in METHODS for case in CASES
               if not (ROOT / "records" / method / f"case_{case:02}.json").is_file()]
    if missing:
        raise RuntimeError(f"generation is incomplete; missing {len(missing)} records")
    route.GPUS = DECODE_GPUS
    spawn("decode_worker", gpus=DECODE_GPUS)
    route.manifests()
    quality()


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "decode_worker":
        route.GPUS = DECODE_GPUS
        route.decode_worker(int(sys.argv[2]))
    elif command == "generate_worker":
        getattr(route, command)(int(sys.argv[2]))
    elif command in ("run", "prepare", "aggregate", "quality", "resume_decode"):
        globals()[command]()
    elif command == "manifests":
        route.manifests()
    else:
        raise ValueError(command)
