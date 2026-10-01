#!/usr/bin/env python3
"""Run paired SM80 duration benchmarks concurrently on GPUs 0-7."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "benchmark_sm80_dense_vs_spark10_10s768p.py"
LOG_DIR = ROOT / "reports" / "sm80_multilength_logs"
JOBS = {
    0: (5, "gpu0"),
    1: (5, "gpu1"),
    2: (5, "gpu2"),
    3: (10, "gpu3"),
    4: (10, "gpu4"),
    5: (15, "gpu5"),
    6: (15, "gpu6"),
    7: (10, "gpu7"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--route-execution", choices=("threshold", "fused"), default="threshold"
    )
    parser.add_argument("--reuse-dense", action="store_true")
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="prefix output tags while retaining the GPU suffix",
    )
    args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",") if item]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    running = {}
    for gpu in gpus:
        seconds, tag = JOBS[gpu]
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            HF_HUB_OFFLINE="1",
            OMP_NUM_THREADS="4",
            TORCHINDUCTOR_COMPILE_THREADS="4",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            PYTHONPATH=str(ROOT),
            CUTE_DSL_ARCH="sm_80",
        )
        output_tag = f"{args.tag_prefix}_{tag}" if args.tag_prefix else tag
        log_path = LOG_DIR / f"{seconds}s_{output_tag}_{args.route_execution}.log"
        log = log_path.open("w")
        command = [
                sys.executable,
                str(BENCHMARK),
                "--seconds",
                str(seconds),
                "--tag",
                output_tag,
                "--route-execution",
                args.route_execution,
            ]
        if args.reuse_dense:
            route_suffix = "_cta_topk" if args.route_execution == "fused" else ""
            dense_report = (
                ROOT / "reports"
                / f"sm80_fused{route_suffix}_dense_vs_spark10_{seconds}s768p_seed42_{tag}.json"
            )
            command.extend(("--reuse-dense-report", str(dense_report)))
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        running[gpu] = (process, log, log_path, seconds, tag)
        print(f"started gpu={gpu} duration={seconds}s pid={process.pid} log={log_path}", flush=True)

    failed = False
    while running:
        time.sleep(5)
        for gpu, item in list(running.items()):
            process, log, log_path, seconds, tag = item
            code = process.poll()
            if code is None:
                continue
            log.close()
            failed |= code != 0
            print(
                f"finished gpu={gpu} duration={seconds}s tag={tag} exit={code} log={log_path}",
                flush=True,
            )
            del running[gpu]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
