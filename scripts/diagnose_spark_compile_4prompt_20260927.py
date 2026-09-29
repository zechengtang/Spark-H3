#!/usr/bin/env python3
"""Four-case current-BF16-threshold run with transformer torch.compile enabled."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import diagnose_spark_psnr_regression_4prompt_20260927 as prior

NAME = "diagnose_spark_compile_4prompt_20260927"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHOD = "current_bf16_threshold_compile"


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
    prior.ROOT, prior.OUT, prior.METHOD = ROOT, OUT, METHOD
    prior.configure()
    prior.route.parse_args = parse_args


def prepare():
    prior.prepare()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["torch_compile"] = True
    prior.route.base.write(ROOT / "protocol.json", protocol)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(prior.route.GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(prior.route.base.PYTHON), str(Path(__file__).resolve()), stage, str(rank)],
            env={**os.environ, **prior.route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
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
