#!/usr/bin/env python3
"""Paired-card correction for the dynamic-text Spark warmup study."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import run_spark_dynamic_text_warmup_study_20261002 as study


study.NAME = "spark_dynamic_text_warmup_paired_20261003"
study.ROOT = Path("/autodl-fs/data/h3_experiments") / study.NAME
study.OUT = Path("/autodl-fs/data/h3_outputs") / study.NAME
ROOT, OUT = study.ROOT, study.OUT


def worker(rank):
    import torch
    import h3_sparse_attention.sol_numerator_virtual_q as fused

    rank = int(rank)
    torch.set_num_threads(4)
    p = study.base.pipeline()
    selected = study.cases()
    workflow, states = p.configure_denoise_workflow(study.args(), selected)
    pipe, manager, acceleration, placement = p.load_denoiser(study.args(), workflow)
    relation = "same" if rank % 2 == 0 else "cross"
    conditions = [f"full_{relation}", f"short_{relation}"]
    if rank >= 2:
        conditions.reverse()
    try:
        for order, condition in enumerate(conditions):
            fused._FUSED_COMPILED.clear()
            fused._FUSED_COMPILE_CALLS = 0
            fused._FUSED_COMPILE_SECONDS = 0.0
            warm_steps = (study.SHORT_STEPS if condition.startswith("short")
                          else study.FORMAL_STEPS)
            warm = study.run_once(pipe, p, states[0], steps=warm_steps,
                                  seed=40 + order)
            study.base.write(ROOT / "records" / f"gpu{study.GPUS[rank]}_{condition}_warmup.json", {
                "phase": "discarded_warmup", "condition": condition,
                "gpu": study.GPUS[rank], "order": order,
                "case": selected[0]["index"], "placement": placement, **warm,
            })
            target = 0 if relation == "same" else 1
            row = study.run_once(pipe, p, states[target], steps=study.FORMAL_STEPS,
                                 seed=42)
            study.base.write(ROOT / "records" / f"gpu{study.GPUS[rank]}_{condition}_measured.json", {
                "phase": "measured", "condition": condition,
                "gpu": study.GPUS[rank], "order": order,
                "case": selected[target]["index"],
                "prompt_sha256": selected[target]["prompt_sha256"],
                "placement": placement, **row,
            })
            print("MEASURED", condition, round(row["seconds"], 3),
                  "new_compiles", row["new_compile_calls"], flush=True)
        study.base.write(ROOT / f"worker_gpu{study.GPUS[rank]}.json",
                         {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def summarize():
    rows = [json.loads(path.read_text()) for path in
            (ROOT / "records").glob("*measured.json")]
    by_condition = {}
    for condition in study.CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != 2:
            raise RuntimeError(f"missing {condition}: {len(selected)}")
        by_condition[condition] = {
            "values": [row["seconds"] for row in selected],
            "mean_seconds": statistics.mean(row["seconds"] for row in selected),
            "new_compile_calls": [row["new_compile_calls"] for row in selected],
        }
    pairs = {}
    for relation, gpus in (("same", (0, 2)), ("cross", (1, 3))):
        deltas = []
        for gpu in gpus:
            full = next(row["seconds"] for row in rows
                        if row["gpu"] == gpu and row["condition"] == f"full_{relation}")
            short = next(row["seconds"] for row in rows
                         if row["gpu"] == gpu and row["condition"] == f"short_{relation}")
            deltas.append(short - full)
        pairs[relation] = {"short_minus_full_seconds": deltas,
                           "mean_delta_seconds": statistics.mean(deltas)}
    passed = all(abs(value["mean_delta_seconds"]) <= 2.0 for value in pairs.values())
    passed = passed and all(
        calls == [0, 0] for calls in
        (by_condition[name]["new_compile_calls"] for name in study.CONDITIONS)
    )
    result = {"status": "complete", "paired_by_gpu": pairs,
              "conditions": by_condition, "short_warmup_accepted": passed,
              "selected_protocol": ("one_dense_one_sparse" if passed
                                     else "kernel_performance_matrix")}
    study.base.write(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol.update(status="complete", paired_card_correction=True,
                    selection_source="results.json")
    study.base.write(ROOT / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


def launch():
    study.prepare()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["paired_card_correction"] = True
    protocol["formal_repeats_per_condition"] = 2
    protocol["pairing"] = "same GPU; condition order reversed on replicate GPU"
    protocol["runner_sha256"] = study.base.sha(Path(__file__).resolve())
    study.base.write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")
    jobs = []
    for rank, gpu in enumerate(study.GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen(
            [str(study.base.PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **study.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT)
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
