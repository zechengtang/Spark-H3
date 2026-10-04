#!/usr/bin/env python3
"""Decode/archive the five priority 5s arms and run official-subset VBench."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
EXPERIMENTS = REPO.parent / "MiniMax-H3-Experiments"
VBENCH_REPO = REPO.parent / "VBench"
MODEL = Path("/mnt/CFS/tangzecheng/models/MiniMax-H3")
WEIGHT_ROOT = Path("/mnt/CFS/tangzecheng/models/vbench")
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
ROOT = Path(os.environ.get(
    "REB_PRIORITY_5S_VBENCH_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_vbench_25p_20261003",
)).resolve()
METHODS = (
    "dense", "baseline", "landmarks64",
    "q_reuses_k_layout", "k_reuses_q_layout",
)
SOURCE_ROOTS = {
    "dense": Path("/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002"),
    "baseline": Path("/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002"),
    "landmarks64": Path("/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002"),
    "q_reuses_k_layout": Path("/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll11_q_reuses_k_layout_20261002"),
    "k_reuses_q_layout": Path("/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll15_priority4_20261003"),
}
WORKERS = 8


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    def command(*args):
        return subprocess.check_output(args, cwd=path, text=True).strip()
    return {
        "path": str(path.resolve()), "revision": command("git", "rev-parse", "HEAD"),
        "status": command("git", "status", "--porcelain"),
    }


def prepare() -> None:
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        if protocol["implementation"]["runner_sha256"] != sha256(Path(__file__)):
            raise RuntimeError("runner differs from frozen protocol")
        print("REUSE PREPARED", ROOT, flush=True)
        return
    primary = read(SOURCE_ROOTS["dense"] / "protocol.json")
    cases = primary["cases"]
    if len(cases) != 25:
        raise RuntimeError("expected 25 cases")
    inputs = {str(case["index"]): {} for case in cases}
    for method in METHODS:
        source = SOURCE_ROOTS[method]
        result = read(source / "results.json")
        rows = [row for row in result["records"] if row["arm"] == method]
        by_case = {int(row["case"]): row for row in rows}
        if result.get("status") != "complete" or len(by_case) != 25:
            raise RuntimeError(f"incomplete source method={method}")
        for case in cases:
            row = by_case[int(case["index"])]
            latent = Path(row["latent_path"])
            if (
                row["sample_id"] != case["sample_id"]
                or row["prompt_sha256"] != case["prompt_sha256"]
                or (row["frames"], row["height"], row["width"], row["steps"], row["seed"])
                != (120, 768, 1344, 20, 42)
                or not latent.is_file()
                or sha256(latent) != row["latent_sha256"]
            ):
                raise RuntimeError(f"source gate failed method={method} case={case['index']}")
            inputs[str(case["index"])][method] = {
                "source_root": str(source), "latent_path": str(latent.resolve()),
                "latent_sha256": row["latent_sha256"], "denoise_seconds": row["denoise_seconds"],
            }
    protocol = {
        "status": "prepared", "purpose": "5s768p priority Reblock VBench official subsets",
        "methods": METHODS, "cases": cases, "inputs": inputs,
        "settings": {"frames": 120, "height": 768, "width": 1344,
                     "fps": 24, "steps": 20, "seed": 42},
        "evaluation": "Only each prompt's declared samples.json vbench_dimensions",
        "source_roots": {method: str(path) for method, path in SOURCE_ROOTS.items()},
        "implementation": {
            "runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__)),
            "archive_helper": str(REPO / "scripts/reblock_priority_10s768p_quality.py"),
            "archive_helper_sha256": sha256(REPO / "scripts/reblock_priority_10s768p_quality.py"),
            "official_module": str(REPO / "scripts/vbench_core5_official_subset_module.py"),
            "official_module_sha256": sha256(REPO / "scripts/vbench_core5_official_subset_module.py"),
            "repos": {"spark_h3": git_state(REPO), "benchmark": git_state(BENCH),
                      "experiments": git_state(EXPERIMENTS), "vbench": git_state(VBENCH_REPO)},
        },
    }
    write(protocol_path, protocol)
    write(ROOT / "status.json", {"status": "prepared"})


def complete(method: str, case: dict, protocol: dict) -> bool:
    record = ROOT / "decode_records" / method / f"{case['index']:02d}.json"
    video = ROOT / "quality_videos" / method / f"{case['index']:02d}.mkv"
    evidence = video.with_suffix(".archive.json")
    try:
        row = read(record)
        source = protocol["inputs"][str(case["index"])][method]
        return (
            row.get("status") == "complete" and row.get("method") == method
            and row.get("case") == case["index"]
            and row.get("latent_sha256") == source["latent_sha256"]
            and video.is_file() and evidence.is_file()
            and row.get("video_sha256") == sha256(video)
            and read(evidence).get("file_sha256") == row.get("video_sha256")
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def worker(slot: int) -> None:
    import torch
    sys.path.insert(0, str(REPO / "scripts"))
    import reblock_priority_10s768p_quality as archive
    archive.base.FRAMES = 120

    protocol = read(ROOT / "protocol.json")
    cases = protocol["cases"][slot::WORKERS]
    pending = [(method, case) for case in cases for method in METHODS
               if not complete(method, case, protocol)]
    if not pending:
        write(ROOT / f"decode_worker_{slot}.json", {"status": "complete", "decoded": 0})
        return
    pipeline = archive.base.import_pipeline()
    args = pipeline.build_parser().parse_args(
        ["decode", "--model", str(MODEL), "--output", str(ROOT), "--method", "dense",
         "--frames", "120", "--height", "768", "--width", "1344"]
    )
    pipe, manager, acceleration = pipeline.load_decoder(args)
    decoded = 0
    try:
        for case in cases:
            for method in METHODS:
                if complete(method, case, protocol):
                    continue
                source = protocol["inputs"][str(case["index"])][method]
                pixels, audio, rate, decode = archive.load_media(pipe, source)
                video, evidence = archive.archive_media(
                    ROOT, method, case, pixels, audio, rate
                )
                del pixels, audio
                write(ROOT / "decode_records" / method / f"{case['index']:02d}.json", {
                    "status": "complete", "method": method, "case": case["index"],
                    "sample_id": case["sample_id"], "prompt_sha256": case["prompt_sha256"],
                    "latent_path": source["latent_path"], "latent_sha256": source["latent_sha256"],
                    "video_path": str(video.resolve()), "video_sha256": evidence["file_sha256"],
                    "decode": decode, "physical_gpu": slot,
                })
                decoded += 1
                print(f"ARCHIVED slot={slot} method={method} case={case['index']}", flush=True)
        write(ROOT / f"decode_worker_{slot}.json", {"status": "complete", "decoded": decoded})
    except Exception as error:
        write(ROOT / "failures" / f"decode_slot{slot}_{time.time_ns()}.json", {
            "status": "failed", "slot": slot, "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        acceleration.remove()
        del pipe, manager
        torch.cuda.empty_cache()


def build_manifests() -> None:
    protocol = read(ROOT / "protocol.json")
    for method in METHODS:
        records = []
        for case in protocol["cases"]:
            if not complete(method, case, protocol):
                raise RuntimeError(f"incomplete archive method={method} case={case['index']}")
            row = read(ROOT / "decode_records" / method / f"{case['index']:02d}.json")
            records.append({"index": case["index"], "sample_id": case["sample_id"],
                            "output_path": row["video_path"], "sha256": row["video_sha256"]})
        write(ROOT / "quality_videos" / method / "generation_manifest.json", {
            "schema_version": 1, "status": "passed", "method": method,
            "sample_count": 25, "records": records,
        })


def decode() -> None:
    prepare()
    write(ROOT / "status.json", {"status": "decoding"})
    env_base = {**os.environ, "H3_IMPL_REPO": str(REPO), "H3_DIFFUSERS_DIR": str(MODEL),
                "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"}
    jobs = []
    for slot in range(WORKERS):
        log = (ROOT / f"decode_worker_{slot}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", "--slot", str(slot)],
            cwd=REPO, env={**env_base, "CUDA_VISIBLE_DEVICES": str(slot)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((slot, process, log))
    failures = []
    for slot, process, log in jobs:
        code = process.wait(); log.close()
        if code:
            failures.append({"slot": slot, "exit_code": code})
    if failures:
        write(ROOT / "status.json", {"status": "decode_failed", "workers": failures})
        raise RuntimeError(f"decode failures: {failures}")
    build_manifests()
    write(ROOT / "status.json", {"status": "decoded"})


def vbench_setup() -> None:
    build_manifests()
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
        text = (EXPERIMENTS / "scripts" / name).read_text()
        text = text.replace("REPO = Path(__file__).resolve().parents[1]",
                            f"REPO = Path({str(BENCH)!r})", 1)
        text = text.replace("/autodl-fs/data/models/vbench", str(WEIGHT_ROOT))
        (code / name).write_text(text)
    manifest = read(WEIGHT_ROOT / "manifest.json")
    weights = {str((WEIGHT_ROOT / row["path"]).resolve()): {
        "sha256": row["sha256"], "bytes": row["bytes"]
    } for row in manifest["checkpoints"]}
    weight_list = ROOT / "vbench_model_weights.json"
    write(weight_list, weights)
    labels = {
        "dense": "Dense", "baseline": "Spark-H3-10pct baseline",
        "landmarks64": "Spark-H3-10pct landmarks64",
        "q_reuses_k_layout": "Spark-H3-10pct Q reuses K",
        "k_reuses_q_layout": "Spark-H3-10pct K reuses Q",
    }
    write(ROOT / "vbench/module_config.json", {
        "cases": list(range(1, 26)), "labels": labels,
        "video_sources": [{"method": method, "from": "manifest",
                           "path": str(ROOT / "quality_videos" / method / "generation_manifest.json")}
                          for method in METHODS],
        "historical_scores": [], "model_weights": str(weight_list), "samples": str(SAMPLES),
    })
    env = {**os.environ, "H3_IMPL_REPO": str(REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(WEIGHT_ROOT)}
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0,'.'); import vbench; vbench.prepare()"],
                   cwd=ROOT, env=env, check=True)
    write(ROOT / "status.json", {"status": "vbench_prepared"})


def vbench_run() -> None:
    env = {**os.environ, "H3_IMPL_REPO": str(REPO), "HF_HUB_OFFLINE": "1",
           "VBENCH_CACHE_DIR": str(WEIGHT_ROOT), "PYTHONUNBUFFERED": "1"}
    write(ROOT / "status.json", {"status": "vbench_running"})
    subprocess.run([sys.executable, str(ROOT / "h3_vbench_queue.py"), "run",
                    "--experiment", str(ROOT), "--gpus", *map(str, range(8))],
                   cwd=ROOT, env=env, check=True)
    write(ROOT / "status.json", {"status": "complete"})


def status() -> None:
    payload = {"root": str(ROOT), "status": read(ROOT / "status.json") if (ROOT / "status.json").exists() else None}
    payload["archived"] = sum(1 for method in METHODS for case in range(1, 26)
                              if (ROOT / "decode_records" / method / f"{case:02d}.json").is_file())
    db = ROOT / "vbench/queue.sqlite"
    if db.exists():
        import sqlite3
        connection = sqlite3.connect(db)
        payload["vbench_jobs"] = dict(connection.execute(
            "SELECT state,count(*) FROM jobs GROUP BY state").fetchall())
        connection.close()
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "decode", "worker", "vbench-setup",
                                             "vbench-run", "all", "status"))
    parser.add_argument("--slot", type=int, choices=range(WORKERS))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "decode": decode()
    elif args.command == "worker":
        if args.slot is None: parser.error("worker requires --slot")
        worker(args.slot)
    elif args.command == "vbench-setup": vbench_setup()
    elif args.command == "vbench-run": vbench_run()
    elif args.command == "all":
        decode(); vbench_setup(); vbench_run()
    else: status()


if __name__ == "__main__":
    main()
