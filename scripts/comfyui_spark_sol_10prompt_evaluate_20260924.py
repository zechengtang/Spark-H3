#!/usr/bin/env python3
"""PSNR/SSIM/LPIPS and VBench evaluation for the 10-prompt ComfyUI run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


NAME = "comfyui_spark_sol_10prompt_5s768p_20260924"
REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
METHODS = (
    "dense",
    "sol_tau1_extra0",
    "sol_tau1_extra256",
    "sol_tau13_extra256",
    "spark_topk10",
)
SPARSE = METHODS[1:]
CASES = tuple(range(1, 11))
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
        rows = [{"index": row["case"], "sample_id": row["sample_id"],
                 "output_path": row["video_path"], "sha256": row["video_sha256"],
                 "video": row["video"]}
                for row in decoded if row["method"] == method]
        rows.sort(key=lambda row: row["index"])
        assert [row["index"] for row in rows] == list(CASES)
        write(OUT / method / "generation_manifest.json", {
            "status": "passed", "method": method, "sample_count": len(rows), "records": rows,
        })


def quality():
    summaries = {}
    script = BENCH / "scripts" / "h3_quality_video_pair.py"
    for method in SPARSE:
        work = ROOT / "quality" / method
        config = {"work_dir": str(work),
                  "reference_manifest": str(OUT / "dense" / "generation_manifest.json"),
                  "candidate_manifest": str(OUT / method / "generation_manifest.json"),
                  "method": method, "cases": list(CASES), "workers": 4}
        config_path = ROOT / "quality" / f"{method}_config.json"
        write(config_path, config)
        subprocess.run([PYTHON, str(script), "run", "--config", str(config_path)], check=True,
                       env={**os.environ, "H3_NUM_GPUS": "4", "HF_HUB_OFFLINE": "1"})
        summaries[method] = read(work / f"{method}_quality_results.json")["summary"]
    write(ROOT / "quality_results.json", {"status": "complete", "summary": summaries})


def vbench():
    shutil.copy2(REPO / "scripts" / "comfyui_spark_sol_10prompt_vbench_20260924.py", ROOT / "vbench.py")
    subprocess.run([PYTHON, "-c", "import vbench; vbench.prepare()"], cwd=ROOT, check=True)
    subprocess.run([PYTHON, str(BENCH / "scripts" / "h3_vbench_queue.py"), "run",
                    "--experiment", str(ROOT), "--gpus", "0", "1", "2", "3"],
                   check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})


def report():
    write(ROOT / "results.json", {
        "status": "complete",
        "timing": read(ROOT / "denoise_summary.json")["methods"],
        "quality": read(ROOT / "quality_results.json")["summary"],
        "vbench": read(ROOT / "vbench" / "results.json")["scores_percent_assigned_prompts"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("manifests", "quality", "vbench", "report", "all"))
    args = parser.parse_args()
    if args.command == "manifests": manifests()
    elif args.command == "quality": quality()
    elif args.command == "vbench": vbench()
    elif args.command == "report": report()
    else: manifests(); quality(); vbench(); report()
