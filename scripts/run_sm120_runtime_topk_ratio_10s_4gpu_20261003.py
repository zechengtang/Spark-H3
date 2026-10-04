#!/usr/bin/env python3
"""Four-GPU 10s confirmation of SM120 cross-Top-K-ratio callable reuse."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import run_sm120_runtime_topk_ratio_4gpu_20261003 as base


NAME = "sm120_runtime_topk_ratio_10s_4gpu_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
GPUS = (0, 1, 2, 3)

base.NAME = NAME
base.ROOT = ROOT
base.OUT = OUT
base.FRAMES = 240


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(
            base.SOURCE / name, target_is_directory=name.endswith("cache")
        )
    selected = base.cases()
    base.base.pipeline().configure_denoise_workflow(base.args(), selected)
    diff = subprocess.check_output(
        ["git", "-C", str(base.base.IMPL), "diff", "--binary"]
    )
    protocol = {
        "status": "running",
        "name": NAME,
        "hypothesis": (
            "at 10s/768p a callable compiled for positive Top-K 10% safely "
            "accepts 20%, with no measurable steady-state regression"
        ),
        "cases": selected,
        "gpus": list(GPUS),
        "frames": 240,
        "resolution": [base.WIDTH, base.HEIGHT],
        "steps": base.STEPS,
        "evaluations": base.STEPS - 1,
        "seed": 42,
        "design": (
            "per GPU and mode: discarded 3-step 10% warmup, one formal 20% "
            "cross-ratio trial, then one compile-free 20% repeat; mode order balanced"
        ),
        "expected_total_compile_calls": {"static": 2, "runtime": 1},
        "git_revision": base.git("rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": base.base.sha(Path(__file__).resolve()),
        "timing": "CUDA-synchronized denoise only; signatures excluded",
    }
    base.base.write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def worker(rank):
    import torch
    import h3_sparse_attention.sol_numerator_virtual_q as fused

    rank = int(rank)
    gpu = GPUS[rank]
    torch.set_num_threads(4)
    pipeline_module = base.base.pipeline()
    selected = base.cases()
    workflow, states = pipeline_module.configure_denoise_workflow(base.args(), selected)
    pipe, manager, acceleration, placement = pipeline_module.load_denoiser(
        base.args(), workflow
    )
    mode_order = base.MODES if rank < 2 else tuple(reversed(base.MODES))
    try:
        fused._FUSED_COMPILED.clear()
        fused._FUSED_COMPILE_CALLS = 0
        fused._FUSED_COMPILE_SECONDS = 0.0
        for order, mode in enumerate(mode_order):
            os.environ["H3_SM120_RUNTIME_TOPK_RATIO"] = (
                "1" if mode == "runtime" else "0"
            )
            warm = base.run_once(
                pipe, pipeline_module, states[rank], steps=3, ratio=0.10,
                seed=40 + order,
            )
            base.base.write(ROOT / "records" / f"gpu{gpu}_{mode}_warmup10.json", {
                "phase": "discarded_warmup_10pct", "mode": mode, "gpu": gpu,
                "case": selected[rank]["index"], "placement": placement, **warm,
            })
            print("WARMUP", mode, round(warm["seconds"], 3),
                  "new_compiles", warm["new_compile_calls"], flush=True)
            for phase in ("cross_ratio", "steady_repeat"):
                row = base.run_once(
                    pipe, pipeline_module, states[rank], steps=base.STEPS,
                    ratio=0.20, seed=42,
                )
                base.base.write(ROOT / "records" / f"gpu{gpu}_{mode}_{phase}.json", {
                    "phase": phase, "mode": mode, "gpu": gpu,
                    "mode_order": order, "case": selected[rank]["index"],
                    "placement": placement, **row,
                })
                print(phase.upper(), mode, round(row["seconds"], 3),
                      "new_compiles", row["new_compile_calls"], flush=True)
        base.base.write(ROOT / f"worker_gpu{gpu}.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline_module.release_cpu_arenas()


def summarize():
    rows = [json.loads(path.read_text()) for path in (ROOT / "records").glob("*.json")]
    result = {"status": "complete", "phases": {}, "compile": {}}
    for phase in ("cross_ratio", "steady_repeat"):
        result["phases"][phase] = {}
        for mode in base.MODES:
            chosen = [r for r in rows if r.get("phase") == phase and r["mode"] == mode]
            if len(chosen) != 4:
                raise RuntimeError(f"missing {phase}/{mode}: {len(chosen)}")
            chosen.sort(key=lambda row: row["gpu"])
            result["phases"][phase][mode] = {
                "seconds": [r["seconds"] for r in chosen],
                "mean_seconds": statistics.mean(r["seconds"] for r in chosen),
                "new_compile_calls": [r["new_compile_calls"] for r in chosen],
                "new_compile_seconds": [r["new_compile_seconds"] for r in chosen],
            }
        static = result["phases"][phase]["static"]
        runtime = result["phases"][phase]["runtime"]
        paired = [b - a for a, b in zip(static["seconds"], runtime["seconds"])]
        result["phases"][phase]["paired_runtime_minus_static_seconds"] = paired
        result["phases"][phase]["mean_runtime_minus_static_seconds"] = statistics.mean(paired)
    for mode in base.MODES:
        by_gpu = {}
        for gpu in GPUS:
            selected = [r for r in rows if r.get("mode") == mode and r.get("gpu") == gpu]
            by_gpu[str(gpu)] = {
                "calls": sum(r["new_compile_calls"] for r in selected),
                "seconds": sum(r["new_compile_seconds"] for r in selected),
            }
        result["compile"][mode] = by_gpu
    signatures_match = True
    mismatches = []
    for gpu in GPUS:
        for phase in ("cross_ratio", "steady_repeat"):
            pair = [r for r in rows if r.get("gpu") == gpu and r.get("phase") == phase]
            if len(pair) != 2 or pair[0]["output_signature"] != pair[1]["output_signature"]:
                signatures_match = False
                mismatches.append({"gpu": gpu, "phase": phase})
    compile_ok = all(
        result["compile"]["static"][str(gpu)]["calls"] == 2
        and result["compile"]["runtime"][str(gpu)]["calls"] == 1
        for gpu in GPUS
    )
    steady_delta = result["phases"]["steady_repeat"]["mean_runtime_minus_static_seconds"]
    steady_static = result["phases"]["steady_repeat"]["static"]["mean_seconds"]
    result.update(
        signatures_match=signatures_match,
        signature_mismatches=mismatches,
        expected_compile_reuse=compile_ok,
        steady_overhead_percent=steady_delta / steady_static * 100.0,
        accepted=signatures_match and compile_ok and abs(steady_delta) <= 3.0,
        acceptance_rule=(
            "matching signatures, static 2 vs runtime 1 compile per GPU, "
            "and <=3s paired mean steady delta"
        ),
    )
    base.base.write(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete"
    base.base.write(ROOT / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


def launch():
    prepare()
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen(
            [str(base.base.PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **base.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, proc, log))
    codes = [(gpu, proc.wait()) for gpu, proc, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"workers failed: {codes}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(sys.argv[2])
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
