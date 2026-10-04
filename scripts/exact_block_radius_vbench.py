#!/usr/bin/env python3
"""Run the Benchmark-compatible cached VBench core-five protocol for selected arms."""
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
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
WEIGHT_ROOT = Path("/mnt/CFS/tangzecheng/models/vbench")
LABELS = {
    "dense": "Dense",
    "baseline": "Spark-H3-10pct",
    "q_reuses_k_layout": "Spark-H3-10pct Q reuses K, radius=None",
    "k_reuses_q_layout": "Spark-H3-10pct K reuses Q, radius=None",
    "q_reuses_k_exact_radius0": "Spark-H3-10pct Q reuses K, radius=0",
    "k_reuses_q_exact_radius0": "Spark-H3-10pct K reuses Q, radius=0",
}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}{path.suffix}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def setup(root: Path, methods: list[str]) -> None:
    for required in (BENCH, EXPERIMENTS, VBENCH, SAMPLES, WEIGHT_ROOT / "manifest.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    unknown = sorted(set(methods) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    code = root / "vbench_code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPERIMENTS / "scripts/vbench_core5_cached_module.py", root / "vbench.py")
    queue_text = (EXPERIMENTS / "scripts/h3_vbench_queue.py").read_text().replace(
        "os.environ['VBENCH_CACHE_DIR']='/autodl-fs/data/models/vbench'",
        f"os.environ['VBENCH_CACHE_DIR']={str(WEIGHT_ROOT)!r}",
    )
    (root / "h3_vbench_queue.py").write_text(queue_text)
    shutil.copy2(BENCH / "scripts/_impl_bootstrap.py", root / "_impl_bootstrap.py")
    for name in ("score_vbench20pct_768p10s.py", "score_vbench20pct_aesthetic.py"):
        text = (EXPERIMENTS / "scripts" / name).read_text()
        text = text.replace(
            "REPO = Path(__file__).resolve().parents[1]", f"REPO = Path({str(BENCH)!r})", 1
        ).replace("/autodl-fs/data/models/vbench", str(WEIGHT_ROOT))
        (code / name).write_text(text)
    manifest = json.loads((WEIGHT_ROOT / "manifest.json").read_text())
    weight_list = root / "vbench_model_weights.json"
    write(weight_list, {
        str((WEIGHT_ROOT / row["path"]).resolve()): {
            "sha256": row["sha256"], "bytes": row["bytes"]
        }
        for row in manifest["checkpoints"]
    })
    sources = []
    for method in methods:
        path = root / "quality_videos" / method / "generation_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append({"method": method, "from": "manifest", "path": str(path)})
    write(root / "vbench/module_config.json", {
        "cases": list(range(1, 26)),
        "labels": {method: LABELS[method] for method in methods},
        "video_sources": sources,
        "historical_scores": [],
        "model_weights": str(weight_list),
        "samples": str(SAMPLES),
    })
    env = {**os.environ, "H3_IMPL_REPO": str(REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(WEIGHT_ROOT)}
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0,'.'); import vbench; vbench.prepare()"],
        cwd=root, env=env, check=True,
    )


def run(root: Path, gpus: list[int]) -> None:
    env = {**os.environ, "H3_IMPL_REPO": str(REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(WEIGHT_ROOT), "PYTHONUNBUFFERED": "1"}
    subprocess.run(
        [sys.executable, str(root / "h3_vbench_queue.py"), "run",
         "--experiment", str(root), "--gpus", *map(str, gpus)],
        cwd=root, env=env, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "run", "all", "status"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    root = args.experiment.resolve()
    if args.command in ("setup", "all"):
        setup(root, args.methods)
    if args.command in ("run", "all"):
        run(root, args.gpus)
    if args.command == "status":
        result = root / "vbench/results.json"
        print(result.read_text() if result.exists() else json.dumps({"status": "not complete"}))


if __name__ == "__main__":
    main()
