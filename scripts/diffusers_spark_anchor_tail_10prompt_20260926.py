#!/usr/bin/env python3
"""Current Diffusers Spark query/block-tail and BF16/FP32-anchor ablation."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytorch_spark_route_execution_10prompt_5s768p_20260925 as route

NAME = "diffusers_spark_anchor_tail_10prompt_10s768p_aligned_20260926"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = route.base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE_INDICES = (2, 13, 31, 33, 38, 41, 43, 44, 46, 47)
COMFY_PROTOCOL = Path("/autodl-fs/data/h3_experiments/comfyui_spark_sink_query_10prompt_20260926/protocol.json")
SOURCE_DENSE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json")
SOURCE_COND = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
METHODS = ("spark_threshold", "spark_block", "spark_fp32")
GPUS = (3, 4, 5)
CASES = tuple(range(1, 11))


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig
    return H3SparseAttentionConfig.spark(
        20, sol_log_density=False, sol_video_tail_mode="dense",
        sol_tail_granularity="block" if method == "spark_block" else "query",
        sol_global_anchor_dtype="float32" if method == "spark_fp32" else "bfloat16",
    )


def configure():
    route.ROOT, route.OUT = ROOT, OUT
    route.SAMPLES, route.CASES = SAMPLES, CASES
    route.METHODS, route.GPUS = METHODS, GPUS
    route.EXECUTIONS = {method: "packed_external" for method in METHODS}
    route.DENSE_OUT = ROOT / "dense_reference"
    route.spark_config = config
    route.cases = cases
    route.base.FRAMES = 240
    route.base.INTERNAL_FRAMES = 244


def cases():
    selected = route.base.pipeline().load_cases(
        SAMPLES, list(SOURCE_INDICES), expected_indices=tuple(range(1, 51)))
    aligned = [dict(case, index=i, source_index=case["index"])
               for i, case in enumerate(selected, 1)]
    comfy = json.loads(COMFY_PROTOCOL.read_text())["cases"]
    assert [(c["sample_id"], c["prompt_sha256"]) for c in aligned] == [
        (c["sample_id"], c["prompt_sha256"]) for c in comfy]
    return aligned


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("experiment/output directory exists")
    selected = cases()
    for method in METHODS:
        for folder in (ROOT / "records" / method, ROOT / "warmup" / method,
                       ROOT / "decode_records" / method, OUT / method / "latents",
                       OUT / method / "videos"):
            folder.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(SOURCE_COND / name, target_is_directory=name == "conditioning_cache")
    dense = json.loads(SOURCE_DENSE.read_text())
    source_by_id = {item["sample_id"]: item for item in dense["records"]}
    matched = []
    for case in selected:
        item = dict(source_by_id[case["sample_id"]])
        assert item["prompt_sha256"] == case["prompt_sha256"]
        item["index"] = case["index"]
        matched.append(item)
    (ROOT / "dense_reference").mkdir()
    route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=10,
        settings=dense.get("settings"), records=matched,
        source_manifest=str(SOURCE_DENSE)))
    route.base.write(ROOT / "protocol.json", dict(
        name=NAME, status="prepared", pipeline="native PyTorch/Diffusers",
        purpose="Query vs block approximate tail, and BF16 vs FP32 global anchor",
        cases=selected, methods=METHODS,
        configs={m: dataclasses.asdict(config(m)) for m in METHODS},
        dense_reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
        conditioning_source=str(SOURCE_COND),
        seed=42, requested_steps=20, actual_evaluations=19,
        frames=240, resolution="1344x768", gpus=GPUS,
        warmup="one excluded full denoise per method/GPU",
        timing="synchronized denoise; excludes model load, conditioning, VAE, save",
        pairing="same prompt and GPU with rotating arm order",
    ))
    route.base.pipeline().configure_denoise_workflow(route.parse_args("denoise"), selected)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        env = {**os.environ, **route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
        process = subprocess.Popen([str(route.base.PYTHON), str(Path(__file__).resolve()),
                                    stage, str(rank)], env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def aggregate():
    route.aggregate_runtime()
    path = ROOT / "results.json"
    data = json.loads(path.read_text())
    # The generic route runner uses spark_threshold as the paired baseline.
    data["comparisons"] = {
        "block_vs_query_bf16": "spark_block vs spark_threshold",
        "fp32_vs_bf16_query": "spark_fp32 vs spark_threshold",
    }
    route.base.write(path, data)


def run():
    configure()
    if not ROOT.exists():
        prepare()
    spawn("generate_worker")
    aggregate()
    spawn("decode_worker")
    route.manifests()


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
            method=method, cases=list(CASES), workers=2,
        ))
        subprocess.run([str(route.base.PYTHON), str(route.QUALITY_SCRIPT), "run",
                        "--config", str(config_path)], check=True,
                       env={**os.environ, **route.base.ENV,
                            "CUDA_VISIBLE_DEVICES": "0,1", "H3_NUM_GPUS": "2"})
        summaries[method] = json.loads((work / f"{method}_quality_results.json").read_text())["summary"]
    results = json.loads((ROOT / "results.json").read_text())
    results.update(quality=summaries, status="quality_complete")
    route.base.write(ROOT / "results.json", results)
    route.base.write(ROOT / "quality_results.json", dict(status="complete",
        reference="dense", cases=list(CASES), summaries=summaries))


def vbench():
    configure()
    adapter = ROOT / "vbench.py"
    if not adapter.exists():
        shutil.copy2(Path(__file__).with_name("diffusers_spark_anchor_tail_vbench_20260926.py"), adapter)
    spec = importlib.util.spec_from_file_location("experiment_vbench", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not (ROOT / "vbench" / "protocol.json").exists():
        module.prepare()
    subprocess.run([str(route.base.PYTHON), str(route.base.BENCH / "scripts" / "h3_vbench_queue.py"),
                    "run", "--experiment", str(ROOT), "--gpus", "0", "1", "2"],
                   check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        getattr(route, command)(int(sys.argv[2]))
    elif command == "run":
        run()
    elif command == "prepare":
        prepare()
    elif command == "aggregate":
        aggregate()
    elif command == "quality":
        quality()
    elif command == "vbench":
        vbench()
    elif command == "finalize":
        quality(); vbench()
    else:
        raise ValueError(command)
