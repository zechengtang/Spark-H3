#!/usr/bin/env python3
"""Score Dense plus the three priority 5s Reblock variants with VBench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


import reblock_priority_5s768p_vbench as decoded


METHODS = ("dense", "landmarks64", "q_reuses_k_layout", "k_reuses_q_layout")
ROOT = decoded.ROOT
TORCH_HOME = ROOT / "torch_cache"


def prepare_runtime_cache() -> None:
    """Bind loaders that use Torch Hub implicitly to the audited local weights."""
    checkpoint_dir = TORCH_HOME / "hub/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source = decoded.WEIGHT_ROOT / "dino_model/dino_vitbase16_pretrain.pth"
    target = checkpoint_dir / source.name
    if target.is_symlink() and target.resolve() != source.resolve():
        target.unlink()
    if not target.exists():
        target.symlink_to(source)
    if target.resolve() != source.resolve():
        raise RuntimeError(f"unexpected DINO cache target: {target}")


def setup() -> None:
    prepare_runtime_cache()
    decoded.build_manifests()
    code = ROOT / "vbench_code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copy2(decoded.REPO / "scripts/vbench_core5_official_subset_module.py", ROOT / "vbench.py")
    queue_text = (decoded.EXPERIMENTS / "scripts/h3_vbench_queue.py").read_text().replace(
        "os.environ['VBENCH_CACHE_DIR']='/autodl-fs/data/models/vbench'",
        f"os.environ['VBENCH_CACHE_DIR']={str(decoded.WEIGHT_ROOT)!r}",
    )
    (ROOT / "h3_vbench_queue.py").write_text(queue_text)
    shutil.copy2(decoded.BENCH / "scripts/_impl_bootstrap.py", ROOT / "_impl_bootstrap.py")
    for name in ("score_vbench20pct_768p10s.py", "score_vbench20pct_aesthetic.py"):
        text = (decoded.EXPERIMENTS / "scripts" / name).read_text()
        text = text.replace(
            "REPO = Path(__file__).resolve().parents[1]",
            f"REPO = Path({str(decoded.BENCH)!r})",
            1,
        ).replace("/autodl-fs/data/models/vbench", str(decoded.WEIGHT_ROOT))
        (code / name).write_text(text)
    manifest = decoded.read(decoded.WEIGHT_ROOT / "manifest.json")
    weights = {
        str((decoded.WEIGHT_ROOT / row["path"]).resolve()): {
            "sha256": row["sha256"], "bytes": row["bytes"]
        }
        for row in manifest["checkpoints"]
    }
    weight_list = ROOT / "vbench_model_weights.json"
    decoded.write(weight_list, weights)
    labels = {
        "dense": "Dense",
        "landmarks64": "Spark-H3-10pct landmarks64",
        "q_reuses_k_layout": "Spark-H3-10pct Q reuses K",
        "k_reuses_q_layout": "Spark-H3-10pct K reuses Q",
    }
    decoded.write(ROOT / "vbench/module_config.json", {
        "cases": list(range(1, 26)), "labels": labels,
        "video_sources": [
            {"method": method, "from": "manifest",
             "path": str(ROOT / "quality_videos" / method / "generation_manifest.json")}
            for method in METHODS
        ],
        "historical_scores": [], "model_weights": str(weight_list),
        "samples": str(decoded.SAMPLES),
        "setup_driver": str(Path(__file__).resolve()),
        "setup_driver_sha256": decoded.sha256(Path(__file__).resolve()),
    })
    env = {**os.environ, "H3_IMPL_REPO": str(decoded.REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(decoded.WEIGHT_ROOT), "TORCH_HOME": str(TORCH_HOME)}
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0,'.'); import vbench; vbench.prepare()"],
        cwd=ROOT, env=env, check=True,
    )
    protocol = decoded.read(ROOT / "vbench/protocol.json")
    expected = {"subject_consistency": 28, "background_consistency": 36,
                "motion_smoothness": 28, "imaging_quality": 36,
                "aesthetic_quality": 36}
    if protocol["dimension_job_counts"] != expected:
        raise RuntimeError((protocol["dimension_job_counts"], expected))
    decoded.write(ROOT / "status.json", {"status": "vbench_prepared", "jobs": 164})


def run() -> None:
    prepare_runtime_cache()
    env = {**os.environ, "H3_IMPL_REPO": str(decoded.REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(decoded.WEIGHT_ROOT), "TORCH_HOME": str(TORCH_HOME),
           "PYTHONUNBUFFERED": "1"}
    decoded.write(ROOT / "status.json", {"status": "vbench_running", "jobs": 164})
    subprocess.run(
        [sys.executable, str(ROOT / "h3_vbench_queue.py"), "run",
         "--experiment", str(ROOT), "--gpus", *map(str, range(8))],
        cwd=ROOT, env=env, check=True,
    )
    result = decoded.read(ROOT / "vbench/results.json")
    if tuple(result["scores_percent_official_subsets"]) != METHODS:
        raise RuntimeError("unexpected scored methods")
    decoded.write(ROOT / "status.json", {"status": "complete", "jobs": 164})


def status() -> None:
    payload = {"status": decoded.read(ROOT / "status.json") if (ROOT / "status.json").exists() else None}
    db = ROOT / "vbench/queue.sqlite"
    if db.exists():
        import sqlite3
        connection = sqlite3.connect(db)
        payload["jobs"] = dict(connection.execute(
            "SELECT state,count(*) FROM jobs GROUP BY state").fetchall())
        connection.close()
    result = ROOT / "vbench/results.json"
    if result.exists():
        payload["results"] = decoded.read(result)["scores_percent_official_subsets"]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "run", "all", "status"))
    args = parser.parse_args()
    if args.command in ("setup", "all"): setup()
    if args.command in ("run", "all"): run()
    if args.command == "status": status()


if __name__ == "__main__":
    main()
