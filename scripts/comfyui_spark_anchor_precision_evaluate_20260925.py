#!/usr/bin/env python3
"""Decode-independent manifests, paired quality, VBench, and final report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import comfyui_spark_anchor_precision_benchmark_20260925 as exp


ROOT, OUT = exp.ROOT, exp.OUT
BENCH = exp.REPO.parent / "MiniMax-H3-Benchmark"
PYTHON = "/root/miniconda3/bin/python"
CANDIDATES = ("sol_tau1_extra0", "spark_fp32", "spark_bf16")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _rows_from_decode(seconds, method):
    decoded = read(ROOT / "decode_summary.json")["records"]
    rows = [row for row in decoded
            if row["duration_seconds"] == seconds and row["method"] == method]
    rows.sort(key=lambda row: row["case"])
    return rows


def _source5_manifest(method):
    return read(exp.BASE5_OUT / method / "generation_manifest.json")


def manifests():
    for seconds in exp.DURATION:
        for method in exp.ALL_METHODS:
            if seconds == 5 and method in ("dense", "sol_tau1_extra0"):
                source = _source5_manifest(method)
                rows = [{
                    "index": row["index"], "sample_id": row["sample_id"],
                    "output_path": row["output_path"], "sha256": row["sha256"],
                    "video": row["video"], "source_manifest": str(exp.BASE5_OUT / method / "generation_manifest.json"),
                } for row in source["records"]]
            else:
                decoded = _rows_from_decode(seconds, method)
                rows = [{
                    "index": row["case"], "sample_id": row["sample_id"],
                    "output_path": row["video_path"], "sha256": row["video_sha256"],
                    "video": row["video"],
                } for row in decoded]
            if [row["index"] for row in rows] != list(range(1, 11)):
                raise RuntimeError(f"incomplete manifest for {seconds}s/{method}")
            write(ROOT / "manifests" / f"{seconds}s" / f"{method}.json", {
                "status": "passed", "duration_seconds": seconds, "method": method,
                "sample_count": len(rows), "records": rows,
            })


def _quality_result_path(seconds, method):
    label = f"{seconds}s_{method}"
    return ROOT / "quality" / label / f"{label}_quality_results.json"


def _run_quality(seconds, method, reference="dense", label=None):
    label = label or f"{seconds}s_{method}"
    work = ROOT / "quality" / label
    config = {
        "work_dir": str(work),
        "reference_manifest": str(ROOT / "manifests" / f"{seconds}s" / f"{reference}.json"),
        "candidate_manifest": str(ROOT / "manifests" / f"{seconds}s" / f"{method}.json"),
        "method": label, "cases": list(range(1, 11)), "workers": 2,
    }
    config_path = ROOT / "quality" / f"{label}_config.json"
    write(config_path, config)
    subprocess.run(
        [PYTHON, str(BENCH / "scripts" / "h3_quality_video_pair.py"),
         "run", "--config", str(config_path)],
        check=True,
        env={**os.environ, "H3_NUM_GPUS": "2", "HF_HUB_OFFLINE": "1"},
    )
    return read(work / f"{label}_quality_results.json")["summary"]


def _old_5s_sol_summary():
    return read(exp.BASE5_ROOT / "quality_results.json")["summary"]["sol_tau1_extra0"]


def quality():
    summaries = {"5s": {}, "10s": {}}
    summaries["5s"]["sol_tau1_extra0"] = _old_5s_sol_summary()
    for seconds in exp.DURATION:
        for method in CANDIDATES:
            if seconds == 5 and method == "sol_tau1_extra0":
                continue
            summaries[f"{seconds}s"][method] = _run_quality(seconds, method)
        summaries[f"{seconds}s"]["spark_bf16_vs_fp32"] = _run_quality(
            seconds, "spark_bf16", reference="spark_fp32",
            label=f"{seconds}s_spark_bf16_vs_fp32",
        )
    write(ROOT / "quality_results.json", {"status": "complete", "summary": summaries})


def vbench():
    shutil.copy2(
        exp.REPO / "scripts" / "comfyui_spark_anchor_precision_vbench_20260925.py",
        ROOT / "vbench.py",
    )
    subprocess.run([PYTHON, "-c", "import vbench; vbench.prepare()"], cwd=ROOT, check=True)
    subprocess.run(
        [PYTHON, str(BENCH / "scripts" / "h3_vbench_queue.py"), "run",
         "--experiment", str(ROOT), "--gpus", "0", "1"],
        check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )


def _case_quality(seconds, method, case):
    if seconds == 5 and method == "sol_tau1_extra0":
        return read(exp.BASE5_ROOT / "quality" / method / f"{method}_{case:02}.json")
    label = f"{seconds}s_{method}"
    return read(ROOT / "quality" / label / f"{label}_{case:02}.json")


def _win_loss(seconds, lhs, rhs):
    result = {metric: {"win": 0, "tie": 0, "loss": 0, "deltas": []}
              for metric in ("psnr_db", "ssim", "lpips")}
    for case in range(1, 11):
        a, b = _case_quality(seconds, lhs, case), _case_quality(seconds, rhs, case)
        values = {
            "psnr_db": (a["psnr_db"], b["psnr_db"], 1),
            "ssim": (a["ssim"], b["ssim"], 1),
            "lpips": (a["lpips"], b["lpips"], -1),
        }
        for metric, (av, bv, direction) in values.items():
            delta = av - bv
            result[metric]["deltas"].append(delta)
            signed = direction * delta
            result[metric]["win" if signed > 1e-12 else "loss" if signed < -1e-12 else "tie"] += 1
    return result


def report():
    timing = read(ROOT / "denoise_summary.json")
    quality_result = read(ROOT / "quality_results.json")
    vbench_result = read(ROOT / "vbench" / "results.json")
    comparisons = {}
    for seconds in exp.DURATION:
        comparisons[f"{seconds}s"] = {
            "spark_fp32_vs_sol": _win_loss(seconds, "spark_fp32", "sol_tau1_extra0"),
            "spark_bf16_vs_sol": _win_loss(seconds, "spark_bf16", "sol_tau1_extra0"),
            "spark_bf16_vs_fp32": _win_loss(seconds, "spark_bf16", "spark_fp32"),
        }
    write(ROOT / "results.json", {
        "status": "complete", "timing": timing["durations"],
        "quality": quality_result["summary"],
        "vbench": vbench_result["scores_by_duration"],
        "quality_win_loss": comparisons,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifests", "quality", "vbench", "report", "all"))
    args = parser.parse_args()
    if args.command == "manifests":
        manifests()
    elif args.command == "quality":
        quality()
    elif args.command == "vbench":
        vbench()
    elif args.command == "report":
        report()
    else:
        manifests(); quality(); vbench(); report()
