#!/usr/bin/env python3
"""Resume-safe 5s/10s, 25-prompt M2 x distance ablation for shared Q/K layouts.

Select the duration with REB_MATRIX_DURATION=5s or 10s.  REB_MATRIX_CASES=1
is reserved for the one-case historical Q-layout reproducibility check.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


DURATION = os.environ.get("REB_MATRIX_DURATION", "5s")
if DURATION not in ("5s", "10s"):
    raise ValueError("REB_MATRIX_DURATION must be 5s or 10s")
CASE_SPEC = os.environ.get("REB_MATRIX_CASES", "1-25")
if CASE_SPEC not in ("1", "1-25"):
    raise ValueError("REB_MATRIX_CASES must be 1 or 1-25")
REUSE = os.environ.get("REB_MATRIX_REUSE", "1") == "1"
DEFAULT_NAME = f"reblock_reuse_m2_{DURATION}768p_25p_20261003"
os.environ.setdefault("REB_ABLATION_NAME", DEFAULT_NAME)
os.environ.setdefault(
    "REB_ABLATION_ROOT",
    f"/mnt/CFS/tangzecheng/experiments/{DEFAULT_NAME}",
)
os.environ.setdefault(
    "REB_CONDITIONING_SOURCE_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002",
)
os.environ.setdefault(
    "REB_DENSE_SOURCE_ROOT",
    "/mnt/CFS/tangzecheng/experiments/"
    + (
        "reblock_ablation_5s768p_25p_20261002"
        if DURATION == "5s"
        else "reblock_priority_10s768p_25p_20261003"
    ),
)

import reblock_ablation_5s768p as base  # noqa: E402


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = Path("/mnt/CFS/tangzecheng/experiments")
SOURCE_ROOTS = {
    "5s": {
        "q_reuses_k_layout": EXPERIMENTS
        / "reblock_ablation_5s768p_25p_roll11_q_reuses_k_layout_20261002",
        "k_reuses_q_layout": EXPERIMENTS
        / "reblock_ablation_5s768p_25p_roll15_priority4_20261003",
    },
    "10s": {
        "q_reuses_k_layout": EXPERIMENTS / "reblock_priority_10s768p_25p_20261003",
        "k_reuses_q_layout": EXPERIMENTS / "reblock_priority_10s768p_25p_20261003",
    },
}
BASELINES = ("q_reuses_k_layout", "k_reuses_q_layout")
base.FRAMES = 120 if DURATION == "5s" else 240
base.CASES = (1,) if CASE_SPEC == "1" else tuple(range(1, 26))
base.TUNING_CASES = tuple(index for index in base.TUNING_CASES if index in base.CASES)
base.CONFIRMATION_CASES = tuple(
    index for index in base.CASES if index not in base.TUNING_CASES
)
base.GPUS = (0,) if CASE_SPEC == "1" else tuple(range(8))


def selected_cases():
    return base.import_pipeline().load_cases(
        base.SAMPLES, list(base.CASES), expected_indices=tuple(range(1, 26))
    )


base.selected_cases = selected_cases
arms = {name: base.ARMS[name] for name in ("dense", "baseline")}
for name, reuse in (
    ("q_reuses_k_layout", "q_from_k"),
    ("k_reuses_q_layout", "k_from_q"),
):
    arms[name] = base.ARMS[name]
    for m2, distance in (
        ("both", "euclidean"),
        ("none", "cosine"),
        ("none", "euclidean"),
    ):
        arm = f"{name}__m2_{m2}_{distance}"
        arms[arm] = base.ArmSpec(
            "layout_reuse_x_m2_x_distance",
            f"{reuse}; M2 {m2}; {distance}",
            {
                "landmark_tree_v2_layout_reuse": reuse,
                "landmark_tree_v2_m2_side": m2,
                "landmark_tree_v2_distance": distance,
            },
        )
base.ARMS = arms
base.DEFAULT_ARMS = tuple(name for name in arms if name != "baseline")

_fingerprint = base.implementation_fingerprint


def implementation_fingerprint():
    value = _fingerprint()
    value["adapter"] = {
        "path": str(SCRIPT),
        "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "duration": DURATION,
        "cases": CASE_SPEC,
    }
    return value


base.implementation_fingerprint = implementation_fingerprint
_prepare = base.prepare


def seed_existing_results(names: tuple[str, ...]) -> None:
    configs = json.loads(json.dumps(base.validate_arm_configs(names)))
    reused = []
    for arm in BASELINES:
        if arm not in names:
            continue
        source_root = SOURCE_ROOTS[DURATION][arm]
        source_protocol = base.read_json(source_root / "protocol.json")
        if source_protocol["settings"]["frames"] != base.FRAMES:
            raise RuntimeError(f"source frame mismatch: {source_root}")
        source_cases = {
            row["sample_id"]: row for row in source_protocol["cases"]
        }
        for case in selected_cases():
            sample_id = case["sample_id"]
            source_case = source_cases.get(sample_id)
            if source_case is None or source_case["prompt_sha256"] != case["prompt_sha256"]:
                raise RuntimeError(f"source prompt mismatch: {arm}/{sample_id}")
            source_stem = f"{source_case['index']:02d}_{sample_id}"
            source_record = source_root / "records" / arm / f"{source_stem}.json"
            record = base.read_json(source_record)
            source_latent = Path(record["latent_path"])
            expected = {
                "status": "complete",
                "arm": arm,
                "sample_id": sample_id,
                "prompt_sha256": case["prompt_sha256"],
                "seed": base.SEED,
                "steps": base.STEPS,
                "frames": base.FRAMES,
                "height": base.HEIGHT,
                "width": base.WIDTH,
            }
            mismatch = {
                key: (record.get(key), wanted)
                for key, wanted in expected.items()
                if record.get(key) != wanted
            }
            if mismatch:
                raise RuntimeError(f"source identity mismatch {arm}/{sample_id}: {mismatch}")
            if record.get("attention_config") != configs[arm]:
                raise RuntimeError(f"source config mismatch {arm}/{sample_id}")
            if not source_latent.is_file() or base.sha256(source_latent) != record["latent_sha256"]:
                raise RuntimeError(f"source latent hash mismatch {arm}/{sample_id}")
            target_latent, target_record = base.paths(arm, case)
            if target_latent.exists() or target_record.exists():
                if not base.complete_record(arm, case):
                    raise RuntimeError(f"incomplete existing target {arm}/{sample_id}")
                continue
            base.link_or_copy(source_latent, target_latent)
            base.atomic_json(
                target_record,
                {
                    **record,
                    "case": case["index"],
                    "latent_path": str(target_latent),
                    "reused_from_record": str(source_record),
                    "reused_from_latent": str(source_latent),
                    "reused_source_implementation": source_protocol["implementation"],
                },
            )
            reused.append({
                "arm": arm,
                "sample_id": sample_id,
                "latent_sha256": record["latent_sha256"],
                "source_record": str(source_record),
            })
    base.atomic_json(
        base.ROOT / "reuse_manifest.json",
        {
            "status": "complete",
            "duration": DURATION,
            "records": reused,
            "expected_reused": sum(arm in names for arm in BASELINES) * len(base.CASES),
        },
    )


def prepare(names: tuple[str, ...]) -> None:
    _prepare(names)
    if REUSE:
        seed_existing_results(names)


base.prepare = prepare


def orchestrate(names: tuple[str, ...]) -> None:
    prepare(names)
    base.validate_arm_configs(names)
    base.atomic_json(base.ROOT / "status.json", {"status": "running", "arms": list(names)})
    env_base = {
        **os.environ,
        "H3_IMPL_REPO": str(base.REPO),
        "H3_DIFFUSERS_DIR": str(base.MODEL),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    processes = []
    for gpu in base.GPUS:
        log = (base.ROOT / f"worker_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "worker", "--gpu", str(gpu), "--arms", *names],
            cwd=base.REPO,
            env={**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log))
    failures = []
    for gpu, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failures.append({"gpu": gpu, "exit_code": code})
    if failures:
        base.atomic_json(base.ROOT / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"worker failures: {failures}")
    base.summarize(names)


base.orchestrate = orchestrate


if __name__ == "__main__":
    base.main()
