#!/usr/bin/env python3
"""Decode archived Spark latents with the current FP32 VAE on four cases."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import diagnose_spark_psnr_regression_4prompt_20260927 as prior

NAME = "diagnose_old_spark_fp32_decode_4prompt_20260927"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHOD = "old_spark_fp32_decode"
SOURCE = Path("/autodl-fs/data/h3_outputs/topk_reblock_reweight_50prompt_20260920/latents")


def configure():
    prior.ROOT, prior.OUT, prior.METHOD = ROOT, OUT, METHOD
    prior.configure()


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("diagnostic output already exists")
    cases = prior.route.cases()
    manifest = json.loads(prior.DENSE.read_text())
    by_id = {row["sample_id"]: row for row in manifest["records"]}
    for folder in (ROOT / "decode_records" / METHOD, ROOT / "dense_reference",
                   OUT / METHOD / "latents", OUT / METHOD / "videos"):
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        row = dict(by_id[case["sample_id"]])
        if row["prompt_sha256"] != case["prompt_sha256"]:
            raise ValueError(f"reference prompt mismatch: {case['index']}")
        row["index"] = case["index"]
        rows.append(row)
        source = SOURCE / f"topk10_reblock_global_reweight_{case['index']:02}.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = OUT / METHOD / "latents" / f"{case['index']:02}_{case['sample_id']}.pt"
        target.symlink_to(source)
    prior.route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=4,
        settings=manifest.get("settings"), records=rows,
        source_manifest=str(prior.DENSE)))
    prior.route.base.write(ROOT / "protocol.json", dict(
        purpose="isolate decoder dtype / decoder implementation from sparse denoising",
        cases=cases, source_latents=str(SOURCE), decoder="current FP32 VAE",
        dense_reference_manifest=str(prior.DENSE)))


def spawn():
    jobs = []
    for rank, gpu in enumerate(prior.route.GPUS):
        log = (ROOT / f"decode_worker_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(prior.route.base.PYTHON), str(Path(__file__).resolve()), "decode_worker", str(rank)],
            env={**os.environ, **prior.route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT)
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"decoder workers failed: {codes}")


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        prepare()
        spawn()
        prior.route.manifests()
        prior.quality()
    elif command == "decode_worker":
        prior.route.decode_worker(int(sys.argv[2]))
    else:
        raise ValueError(command)
