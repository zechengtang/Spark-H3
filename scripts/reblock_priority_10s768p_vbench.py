#!/usr/bin/env python3
"""Prepare and run VBench core-five scoring for the 10s priority experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
EXPERIMENTS = REPO.parent / "MiniMax-H3-Experiments"
VBENCH = REPO.parent / "VBench"
ROOT = Path(os.environ.get(
    "REB_PRIORITY_10S_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_10s768p_25p_20261003",
)).resolve()
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
WEIGHT_ROOT = Path("/mnt/CFS/tangzecheng/models/vbench")
METHODS = (
    "dense", "sol", "baseline", "landmarks64",
    "q_reuses_k_layout", "k_reuses_q_layout",
)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}{path.suffix}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def setup() -> None:
    for required in (BENCH, EXPERIMENTS, VBENCH, SAMPLES, WEIGHT_ROOT / "manifest.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    code = ROOT / "vbench_code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPERIMENTS / "scripts/vbench_core5_cached_module.py", ROOT / "vbench.py")
    queue_text = (EXPERIMENTS / "scripts/h3_vbench_queue.py").read_text()
    queue_text = queue_text.replace(
        "os.environ['VBENCH_CACHE_DIR']='/autodl-fs/data/models/vbench'",
        f"os.environ['VBENCH_CACHE_DIR']={str(WEIGHT_ROOT)!r}",
    )
    (ROOT / "h3_vbench_queue.py").write_text(queue_text)
    shutil.copy2(BENCH / "scripts/_impl_bootstrap.py", ROOT / "_impl_bootstrap.py")
    for name in ("score_vbench20pct_768p10s.py", "score_vbench20pct_aesthetic.py"):
        text = (EXPERIMENTS / "scripts" / name).read_text()
        text = text.replace(
            "REPO = Path(__file__).resolve().parents[1]",
            f"REPO = Path({str(BENCH)!r})",
            1,
        ).replace("/autodl-fs/data/models/vbench", str(WEIGHT_ROOT))
        (code / name).write_text(text)

    manifest = json.loads((WEIGHT_ROOT / "manifest.json").read_text())
    weights = {
        str((WEIGHT_ROOT / row["path"]).resolve()): {
            "sha256": row["sha256"], "bytes": row["bytes"]
        }
        for row in manifest["checkpoints"]
    }
    weight_list = ROOT / "vbench_model_weights.json"
    write(weight_list, weights)
    labels = {
        "dense": "Dense", "sol": "Sol-Attn",
        "baseline": "Spark-H3-10pct baseline",
        "landmarks64": "Spark-H3-10pct landmarks64",
        "q_reuses_k_layout": "Spark-H3-10pct Q reuses K",
        "k_reuses_q_layout": "Spark-H3-10pct K reuses Q",
    }
    sources = []
    for method in METHODS:
        manifest_path = ROOT / "quality_videos" / method / "generation_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        sources.append({"method": method, "from": "manifest", "path": str(manifest_path)})
    write(ROOT / "vbench/module_config.json", {
        "cases": list(range(1, 26)), "labels": labels,
        "video_sources": sources, "historical_scores": [],
        "model_weights": str(weight_list), "samples": str(SAMPLES),
    })
    env = {**os.environ, "H3_IMPL_REPO": str(REPO),
           "HF_HUB_OFFLINE": "1", "VBENCH_CACHE_DIR": str(WEIGHT_ROOT)}
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0,'.'); import vbench; vbench.prepare()"],
        cwd=ROOT, env=env, check=True,
    )


def run(gpus: list[int]) -> None:
    env = {**os.environ, "H3_IMPL_REPO": str(REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(WEIGHT_ROOT), "PYTHONUNBUFFERED": "1"}
    subprocess.run(
        [sys.executable, str(ROOT / "h3_vbench_queue.py"), "run",
         "--experiment", str(ROOT), "--gpus", *map(str, gpus)],
        cwd=ROOT, env=env, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "run", "all", "status"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    if args.command in ("setup", "all"):
        setup()
    if args.command in ("run", "all"):
        run(args.gpus)
    if args.command == "status":
        result = ROOT / "vbench/results.json"
        print(result.read_text() if result.exists() else json.dumps({"status": "not complete"}))


if __name__ == "__main__":
    main()
