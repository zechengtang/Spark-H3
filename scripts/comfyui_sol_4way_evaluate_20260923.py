#!/usr/bin/env python3
"""Evaluate paired fidelity and assigned VBench metrics for the ComfyUI run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess


NAME = "comfyui_sol_4way_blog_ablation_4prompts_20260923"
REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHODS = ("dense", "xmarre_sol", "comfyui_sol", "spark_h3_sol")
CASES = (1, 2, 3, 4)
PYTHON = "/root/miniconda3/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def manifests():
    decoded = read(ROOT / "decode_summary.json")["records"]
    for method in METHODS:
        rows = []
        for row in decoded:
            if row["method"] != method:
                continue
            rows.append({
                "index": row["case"], "sample_id": row["sample_id"],
                "output_path": row["video_path"], "sha256": row["video_sha256"],
                "video": row["video"],
            })
        assert [r["index"] for r in rows] == list(CASES)
        write(OUT / method / "generation_manifest.json", {
            "status": "passed", "method": method, "sample_count": len(rows), "records": rows,
        })


def quality():
    summaries = {}
    script = BENCH / "scripts" / "h3_quality_video_pair.py"
    for method in METHODS[1:]:
        work = ROOT / "quality" / method
        config = {
            "work_dir": str(work),
            "reference_manifest": str(OUT / "dense" / "generation_manifest.json"),
            "candidate_manifest": str(OUT / method / "generation_manifest.json"),
            "method": method, "cases": list(CASES), "workers": 4,
        }
        config_path = ROOT / "quality" / f"{method}_config.json"; write(config_path, config)
        subprocess.run([PYTHON, str(script), "run", "--config", str(config_path)], check=True,
                       env={**os.environ, "H3_NUM_GPUS": "4", "HF_HUB_OFFLINE": "1"})
        summaries[method] = read(work / f"{method}_quality_results.json")["summary"]
    write(ROOT / "quality_results.json", {"status": "complete", "summary": summaries})


def vbench():
    shutil.copy2(REPO / "scripts" / "comfyui_sol_4way_vbench_module_20260923.py", ROOT / "vbench.py")
    subprocess.run([PYTHON, "-c", "import vbench; vbench.prepare()"], cwd=ROOT, check=True)
    subprocess.run([PYTHON, str(BENCH / "scripts" / "h3_vbench_queue.py"), "run", "--experiment", str(ROOT),
                    "--gpus", "0", "1", "2", "3"], check=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "1"})


def report():
    denoise = read(ROOT / "denoise_summary.json")["methods"]
    timing = {}
    dense_mean = statistics.mean(r["sampler_seconds"] for r in denoise["dense"]["records"])
    for method in METHODS:
        values = [r["sampler_seconds"] for r in denoise[method]["records"]]
        mean = statistics.mean(values)
        timing[method] = {"sampler_seconds_mean": mean, "sampler_seconds": values,
                          "speedup_vs_dense": dense_mean / mean}
    result = {"status": "complete", "timing": timing,
              "quality": read(ROOT / "quality_results.json")["summary"],
              "vbench": read(ROOT / "vbench" / "results.json")["scores_percent_assigned_prompts"]}
    write(ROOT / "results.json", result)


def all_stages():
    manifests(); quality(); vbench(); report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifests", "quality", "vbench", "report", "all")); args = parser.parse_args()
    if args.command == "manifests": manifests()
    elif args.command == "quality": quality()
    elif args.command == "vbench": vbench()
    elif args.command == "report": report()
    else: all_stages()
