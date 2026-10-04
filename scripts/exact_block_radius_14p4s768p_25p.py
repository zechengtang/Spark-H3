#!/usr/bin/env python3
"""Run the four requested Spark/Dense variants on the 14.4s 10%-25p protocol."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


os.environ.setdefault("REB_ABLATION_NAME", "spark_radius_q_from_k_14p4s768p_25p_20261004")
os.environ.setdefault(
    "REB_ABLATION_ROOT",
    "/mnt/CFS/tangzecheng/experiments/spark_radius_q_from_k_14p4s768p_25p_20261004",
)
os.environ.setdefault(
    "REB_CONDITIONING_SOURCE_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_10s768p_25p_20261003",
)

import reblock_ablation_5s768p as base  # noqa: E402


ADAPTER = Path(__file__).resolve()
base.FRAMES = 360
base.WARMUP_STEPS = 6
METHODS = ("dense", "baseline", "q_reuses_k_layout", "q_reuses_k_exact_radius0")
base.ARMS = {name: base.ARMS[name] for name in METHODS}
base.DEFAULT_ARMS = METHODS

_base_fingerprint = base.implementation_fingerprint


def implementation_fingerprint():
    result = _base_fingerprint()
    result["adapter"] = {
        "path": str(ADAPTER),
        "sha256": hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
    }
    return result


base.implementation_fingerprint = implementation_fingerprint


def orchestrate(names: tuple[str, ...]) -> None:
    base.prepare(names)
    base.validate_arm_configs(names)
    base.atomic_json(base.ROOT / "status.json", {"status": "running", "arms": list(names)})
    env_base = {
        **os.environ,
        "H3_IMPL_REPO": str(base.REPO),
        "H3_DIFFUSERS_DIR": str(base.MODEL),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "H3_VERBOSE_EXACT_BLOCKS": "1",
    }
    processes = []
    for gpu in base.GPUS:
        log = (base.ROOT / f"worker_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(ADAPTER), "worker", "--gpu", str(gpu), "--arms", *names],
            cwd=base.REPO,
            env={**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log))
    failures = []
    for gpu, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failures.append({"gpu": gpu, "exit_code": code})
    if failures:
        base.atomic_json(base.ROOT / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"worker failures: {failures}")
    base.summarize(names)


base.orchestrate = orchestrate


if __name__ == "__main__":
    base.main()
