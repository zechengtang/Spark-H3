#!/usr/bin/env python3
"""Resume-safe BF16 Spark query/block-tail extension to 50 prompts on GPU0-3.

Run only after the 50-prompt reweight-component ablation releases the GPUs.
The aligned ten-prompt query/block artifacts are reused by sample ID.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import diffusers_spark_reweight_components_50prompt_20260926 as shared
import diffusers_spark_anchor_tail_10prompt_20260926 as aligned

route = shared.route
NAME = "diffusers_spark_tail_granularity_50prompt_10s768p_20260927"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = route.base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
DENSE = shared.DENSE
COND = DENSE.parent.parent
METHODS = ("query", "block")
SOURCE = {"query": "spark_threshold", "block": "spark_block"}
GPUS = (0, 1, 2, 3)
CASES = tuple(range(1, 51))


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig

    if method not in METHODS:
        raise ValueError(method)
    return H3SparseAttentionConfig.spark(
        20, sol_log_density=False, sol_video_tail_mode="dense",
        sol_tail_granularity=method, sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore", sol_reweight_logmass_key="stored",
        sol_reweight_components="full",
    )


def cases():
    return route.base.pipeline().load_cases(SAMPLES, list(CASES), expected_indices=CASES)


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
    # Reuse the shared resume-safe worker/decoder, not its experiment constants.
    shared.ROOT, shared.OUT = ROOT, OUT
    shared.METHODS, shared.GPUS, shared.CASES = METHODS, GPUS, CASES
    shared.cases, shared.config = cases, config


def _reusable(method):
    source_method = SOURCE[method]
    records = aligned.ROOT / "records" / source_method
    manifest_path = aligned.OUT / source_method / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = json.loads(json.dumps(dataclasses.asdict(config(method))))
    by_id = {}
    for row in manifest["records"]:
        index = int(row["index"])
        record = json.loads((records / f"case_{index:02}.json").read_text())
        source_config = dict(record["config"])
        source_config.setdefault("sol_reweight_summary_math", "tensorcore")
        source_config.setdefault("sol_reweight_logmass_key", "stored")
        source_config.setdefault("sol_reweight_components", "full")
        if source_config != expected:
            raise ValueError(f"aligned config mismatch: {method}/{index}")
        if row["sample_id"] != record["sample_id"] or row["prompt_sha256"] != record["prompt_sha256"]:
            raise ValueError(f"aligned record/manifest mismatch: {method}/{index}")
        if not Path(row["output_path"]).is_file() or not Path(record["latent_path"]).is_file():
            raise FileNotFoundError(f"aligned artifact missing: {method}/{index}")
        by_id[row["sample_id"]] = (record, row, manifest_path)
    return by_id


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    selected = cases()
    dense = json.loads(DENSE.read_text())
    dense_by_id = {row["sample_id"]: row for row in dense["records"]}
    if len(selected) != 50 or len(dense_by_id) != 50:
        raise ValueError("50-prompt dataset or dense reference incomplete")
    reuse = {method: _reusable(method) for method in METHODS}
    for method in METHODS:
        for folder in (ROOT / "records" / method, ROOT / "warmup" / method,
                       ROOT / "decode_records" / method, OUT / method / "latents",
                       OUT / method / "videos"):
            folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "dense_reference").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(COND / name, target_is_directory=name == "conditioning_cache")
    dense_rows = []
    reused = {method: 0 for method in METHODS}
    for case in selected:
        dense_row = dict(dense_by_id[case["sample_id"]])
        if dense_row["prompt_sha256"] != case["prompt_sha256"] or not Path(dense_row["output_path"]).is_file():
            raise ValueError(f"dense mismatch: {case['sample_id']}")
        dense_row["index"] = case["index"]
        dense_rows.append(dense_row)
        for method in METHODS:
            found = reuse[method].get(case["sample_id"])
            if found is None:
                continue
            old_record, old_video, source_manifest = found
            if old_record["prompt_sha256"] != case["prompt_sha256"]:
                raise ValueError(f"reused prompt mismatch: {method}/{case['sample_id']}")
            route.base.write(ROOT / "records" / method / f"case_{case['index']:02}.json",
                             dict(old_record, method=method, case=case["index"],
                                  reused_from=str(source_manifest)))
            route.base.write(ROOT / "decode_records" / method / f"case_{case['index']:02}.json",
                             dict(status="complete", method=method, case=case["index"],
                                  sample_id=case["sample_id"], output_path=old_video["output_path"],
                                  sha256=old_video["sha256"], reused_from=str(source_manifest)))
            reused[method] += 1
    route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=50,
        settings=dense.get("settings"), records=dense_rows, source_manifest=str(DENSE)))
    route.base.write(ROOT / "protocol.json", dict(
        name=NAME, status="prepared", pipeline="native PyTorch/Diffusers",
        purpose="Paired BF16 Spark query versus block approximate tail",
        dataset="VBench core-five 20pct first 50", cases=selected, methods=METHODS,
        configs={method: dataclasses.asdict(config(method)) for method in METHODS},
        dense_reference_manifest=str(DENSE), conditioning_source=str(COND),
        seed=42, requested_steps=20, actual_evaluations=19,
        frames=240, resolution="1344x768", gpus=GPUS,
        reused_counts=reused, newly_required=100-sum(reused.values()),
        reuse_source=str(aligned.ROOT),
        excluded_canonical_source=(
            "/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920; "
            "same 10 prompt query latent SHA differs 10/10 from aligned run"),
        timing="synchronized denoise; mixed reused/new runtimes are not paired",
        warmup="one excluded full denoise per method/GPU",
        metrics="paired PSNR/SSIM/LPIPS versus dense plus source-assigned VBench core-five",
        caveat="block consumer path is functionally aligned but not performance-optimized",
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


def paired():
    records = {method: {case: json.loads((ROOT / "quality" / method /
                         f"{method}_{case:02}.json").read_text()) for case in CASES}
               for method in METHODS}
    result = {}
    for metric in ("psnr_db", "ssim", "lpips"):
        deltas = [records["block"][case][metric] - records["query"][case][metric]
                  for case in CASES]
        favorable = deltas if metric != "lpips" else [-value for value in deltas]
        mean = statistics.mean(deltas)
        half_width = 2.0096 * statistics.stdev(deltas) / math.sqrt(len(deltas))
        result[metric] = dict(mean_block_minus_query=mean,
                              ci95=[mean-half_width, mean+half_width],
                              block_wins=sum(value > 0 for value in favorable),
                              query_wins=sum(value < 0 for value in favorable),
                              ties=sum(value == 0 for value in favorable))
    timings = {
        method: {case: json.loads((ROOT / "records" / method /
                            f"case_{case:02}.json").read_text()) for case in CASES}
        for method in METHODS
    }
    runtime = {}
    for label, subset in (
        ("reused_aligned10", [case for case in CASES
                              if "reused_from" in timings["query"][case]]),
        ("newly_generated40", [case for case in CASES
                               if "reused_from" not in timings["query"][case]]),
    ):
        if any(("reused_from" in timings["query"][case]) !=
               ("reused_from" in timings["block"][case]) for case in subset):
            raise ValueError(f"runtime pairing mismatch: {label}")
        deltas = [timings["block"][case]["denoise_seconds"] -
                  timings["query"][case]["denoise_seconds"] for case in subset]
        runtime[label] = dict(count=len(subset), mean_block_minus_query_seconds=statistics.mean(deltas),
                              median_block_minus_query_seconds=statistics.median(deltas),
                              block_faster=sum(value < 0 for value in deltas))
    route.base.write(ROOT / "paired_results.json", dict(status="complete", cases=list(CASES),
        quality=result, paired_runtime_by_source=runtime,
        caveat="Do not pool cross-run absolute times; block consumer is not optimized"))


def vbench():
    adapter = ROOT / "vbench.py"
    source = Path(__file__).with_name("diffusers_spark_tail_granularity_50prompt_vbench_20260927.py")
    if not adapter.exists():
        shutil.copy2(source, adapter)
    spec = importlib.util.spec_from_file_location("experiment_vbench", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not (ROOT / "vbench" / "protocol.json").exists():
        module.prepare()
    subprocess.run([
        str(route.base.PYTHON), str(route.base.BENCH / "scripts" / "h3_vbench_queue.py"),
        "run", "--experiment", str(ROOT), "--gpus", "0", "1", "2", "3",
    ], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    result = json.loads((ROOT / "results.json").read_text())
    result["vbench"] = json.loads((ROOT / "vbench" / "results.json").read_text())[
        "scores_percent_assigned_prompts"]
    result["status"] = "vbench_complete"
    route.base.write(ROOT / "results.json", result)


def run():
    configure()
    if not ROOT.exists() and not OUT.exists():
        prepare()
    elif not (ROOT / "protocol.json").is_file():
        raise RuntimeError("experiment directory exists without protocol")
    spawn("generate_worker")
    shared.aggregate()
    spawn("decode_worker")
    shared.manifests()
    shared.quality()
    paired()
    vbench()


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        getattr(shared, command)(int(sys.argv[2]))
    elif command in ("run", "prepare", "paired", "vbench"):
        globals()[command]()
    elif command == "aggregate":
        shared.aggregate()
    elif command == "manifests":
        shared.manifests()
    elif command == "quality":
        shared.quality()
    else:
        raise ValueError(command)
