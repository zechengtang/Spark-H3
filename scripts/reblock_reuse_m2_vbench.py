#!/usr/bin/env python3
"""Official core-five VBench evaluation for the shared-layout M2 matrix."""
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
DURATION = os.environ.get("REB_MATRIX_DURATION", "5s")
if DURATION not in ("5s", "10s"):
    raise ValueError("REB_MATRIX_DURATION must be 5s or 10s")
ROOT = Path(
    os.environ.get(
        "REB_ABLATION_ROOT",
        f"/mnt/CFS/tangzecheng/experiments/reblock_reuse_m2_{DURATION}768p_25p_20261003",
    )
).resolve()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}{path.suffix}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def methods() -> tuple[str, ...]:
    protocol = read(ROOT / "protocol.json")
    if protocol.get("status") != "complete" or len(protocol.get("cases", [])) != 25:
        raise RuntimeError("generation protocol is not complete for 25 cases")
    if protocol["settings"]["frames"] != (120 if DURATION == "5s" else 240):
        raise RuntimeError("generation duration mismatch")
    return tuple(protocol["selected_arms"])


def setup() -> None:
    for required in (BENCH, EXPERIMENTS, VBENCH, SAMPLES, WEIGHT_ROOT / "manifest.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    selected = methods()
    for method in selected:
        manifest = ROOT / "quality_videos" / method / "generation_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        payload = read(manifest)
        if payload.get("status") != "passed" or payload.get("sample_count") != 25:
            raise RuntimeError(f"invalid video manifest: {manifest}")

    code = ROOT / "vbench_code"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "scripts/vbench_core5_official_subset_module.py", ROOT / "vbench.py")
    queue_text = (EXPERIMENTS / "scripts/h3_vbench_queue.py").read_text().replace(
        "os.environ['VBENCH_CACHE_DIR']='/autodl-fs/data/models/vbench'",
        f"os.environ['VBENCH_CACHE_DIR']={str(WEIGHT_ROOT)!r}",
    )
    (ROOT / "h3_vbench_queue.py").write_text(queue_text)
    shutil.copy2(BENCH / "scripts/_impl_bootstrap.py", ROOT / "_impl_bootstrap.py")
    for name in ("score_vbench20pct_768p10s.py", "score_vbench20pct_aesthetic.py"):
        content = (EXPERIMENTS / "scripts" / name).read_text()
        content = content.replace(
            "REPO = Path(__file__).resolve().parents[1]",
            f"REPO = Path({str(BENCH)!r})",
            1,
        ).replace("/autodl-fs/data/models/vbench", str(WEIGHT_ROOT))
        (code / name).write_text(content)
    manifest = read(WEIGHT_ROOT / "manifest.json")
    weights = {
        str((WEIGHT_ROOT / row["path"]).resolve()): {
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in manifest["checkpoints"]
    }
    write(ROOT / "vbench_model_weights.json", weights)
    write(
        ROOT / "vbench/module_config.json",
        {
            "cases": list(range(1, 26)),
            "labels": {method: method.replace("_", " ") for method in selected},
            "video_sources": [
                {
                    "method": method,
                    "from": "manifest",
                    "path": str(
                        ROOT / "quality_videos" / method / "generation_manifest.json"
                    ),
                }
                for method in selected
            ],
            "historical_scores": [],
            "model_weights": str(ROOT / "vbench_model_weights.json"),
            "samples": str(SAMPLES),
        },
    )
    env = {
        **os.environ,
        "H3_IMPL_REPO": str(REPO),
        "HF_HUB_OFFLINE": "1",
        "VBENCH_CACHE_DIR": str(WEIGHT_ROOT),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0,'.'); import vbench; vbench.prepare()",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def run(gpus: list[int]) -> None:
    env = {
        **os.environ,
        "H3_IMPL_REPO": str(REPO),
        "HF_HUB_OFFLINE": "1",
        "VBENCH_CACHE_DIR": str(WEIGHT_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "h3_vbench_queue.py"),
            "run",
            "--experiment",
            str(ROOT),
            "--gpus",
            *map(str, gpus),
        ],
        cwd=ROOT,
        env=env,
        check=True,
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
        print(result.read_text() if result.is_file() else json.dumps({"status": "not complete"}))


if __name__ == "__main__":
    main()
