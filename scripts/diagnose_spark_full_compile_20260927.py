#!/usr/bin/env python3
"""Compile-on recovery check for the current full Spark preset (4/10/25 cases)."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import diagnose_spark_psnr_regression_4prompt_20260927 as prior

COUNT = int(os.environ.get("H3_DIAG_CASE_COUNT", "4"))
if COUNT not in (4, 10, 25):
    raise ValueError("H3_DIAG_CASE_COUNT must be 4, 10, or 25")
CASES = (2, 4, 6, 8) if COUNT == 4 else tuple(range(1, COUNT + 1))
NAME = f"diagnose_spark_full_compile_{COUNT}prompt_20260927"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHOD = "spark_full_compile"


def config(_method):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        20, sol_log_density=False, sol_video_tail_mode="dense",
        sol_tail_granularity="query", sol_global_anchor_dtype="float32",
        sol_reweight_summary_math="comfy_fp32",
        sol_reweight_logmass_key="pre_round",
        sol_reweight_components="full",
        sol_route_topk_execution="packed_external_no_route_qk",
    )


def parse_args(command):
    return prior.route.base.pipeline().build_parser().parse_args([
        command,
        "--samples", str(prior.route.SAMPLES),
        "--output", str(OUT),
        "--method", "dense",
        "--steps", str(prior.route.base.STEPS),
        "--frames", str(prior.route.base.FRAMES),
        "--height", str(prior.route.base.HEIGHT),
        "--width", str(prior.route.base.WIDTH),
        "--workers", "1",
    ])


def configure():
    prior.ROOT, prior.OUT, prior.METHOD, prior.CASES = ROOT, OUT, METHOD, CASES
    prior.config = config
    prior.configure()
    prior.route.spark_config = config
    prior.route.EXECUTIONS = {METHOD: "packed_external_no_route_qk"}
    prior.route.parse_args = parse_args
    original_run_one = prior.route.run_one

    def run_one(pipe, p, state, case, method, placement, *, warmup=False):
        record_path = ROOT / "records" / METHOD / f"case_{case['index']:02}.json"
        if not warmup and record_path.is_file():
            record = json.loads(record_path.read_text())
            if record.get("torch_compile") is True and Path(record["latent_path"]).is_file():
                return record
        return original_run_one(pipe, p, state, case, method, placement, warmup=warmup)

    prior.route.run_one = run_one


def reuse_earlier_cases():
    if COUNT == 4:
        return 0
    source_count = 4 if COUNT == 10 else 10
    source_root = ROOT.parent / f"diagnose_spark_full_compile_{source_count}prompt_20260927"
    source_protocol = json.loads((source_root / "protocol.json").read_text())
    if source_protocol.get("torch_compile") is not True or source_protocol.get("config") != json.loads(
        json.dumps(dataclasses.asdict(config(METHOD)))):
        raise ValueError("previous compile-on diagnostic is not config-compatible")
    by_id = {case["sample_id"]: case for case in source_protocol["cases"]}
    count = 0
    for case in prior.route.cases():
        source_case = by_id.get(case["sample_id"])
        if source_case is None:
            continue
        old = source_root / "records" / METHOD / f"case_{source_case['index']:02}.json"
        record = json.loads(old.read_text())
        if record.get("torch_compile") is not True or record["config"] != source_protocol["config"]:
            raise ValueError(f"source provenance mismatch: {case['sample_id']}")
        target = OUT / METHOD / "latents" / f"{case['index']:02}_{case['sample_id']}.pt"
        target.symlink_to(Path(record["latent_path"]))
        record.update(case=case["index"], latent_path=str(target), reused_from=str(old))
        prior.route.base.write(ROOT / "records" / METHOD / f"case_{case['index']:02}.json", record)
        count += 1
    return count


def prepare():
    prior.prepare()
    reused = reuse_earlier_cases()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol.update(torch_compile=True, case_count=COUNT,
                    reused_cases=reused,
                    purpose="test whether compile-on restores full Spark PSNR")
    prior.route.base.write(ROOT / "protocol.json", protocol)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(prior.route.GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(prior.route.base.PYTHON), str(Path(__file__).resolve()), stage, str(rank)],
            env={**os.environ, **prior.route.base.ENV,
                 "H3_DIAG_CASE_COUNT": str(COUNT), "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT)
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        prepare()
        spawn("generate_worker")
        spawn("decode_worker")
        prior.route.manifests()
        prior.quality()
    elif command in ("generate_worker", "decode_worker"):
        getattr(prior.route, command)(int(sys.argv[2]))
    else:
        raise ValueError(command)
