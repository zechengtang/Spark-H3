#!/usr/bin/env python3
"""Extend the q_from_k exact-block radius comparison to 50 prompts.

Dense and radius=None are reused from the completed 50-prompt experiment.
Radius 0/1 results for the official 25-prompt subset are reused by sample
identity; workers generate only the remaining 25 prompts for each radius.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


os.environ.setdefault(
    "REB_ABLATION_NAME", "exact_block_radius_q_from_k_5s768p_50p_20261004"
)
os.environ.setdefault(
    "REB_ABLATION_ROOT",
    "/mnt/CFS/tangzecheng/experiments/"
    "exact_block_radius_q_from_k_5s768p_50p_20261004",
)
os.environ.setdefault(
    "REB_CONDITIONING_SOURCE_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_50p_20261003",
)
os.environ.setdefault(
    "REB_DENSE_SOURCE_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_50p_20261003",
)

import reblock_ablation_5s768p as base  # noqa: E402


ADAPTER = Path(__file__).resolve()
SAMPLES_25 = base.BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
SAMPLES_50 = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE_50 = Path(
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_50p_20261003"
)
SOURCE_RADIUS_25 = Path(
    "/mnt/CFS/tangzecheng/experiments/"
    "exact_block_radius_q_from_k_5s768p_25p_20261004"
)

base.SAMPLES = SAMPLES_50
base.CASES = tuple(range(1, 51))
rows_25 = json.loads(SAMPLES_25.read_text())
rows_50 = json.loads(SAMPLES_50.read_text())
ids_25 = {row["sample_id"] for row in rows_25}
ids_50 = {row["sample_id"] for row in rows_50}
if len(ids_25) != 25 or len(ids_50) != 50 or not ids_25 <= ids_50:
    raise RuntimeError("invalid official 25/50-prompt sample relationship")
base.TUNING_CASES = tuple(row["index"] for row in rows_50 if row["sample_id"] in ids_25)
base.CONFIRMATION_CASES = tuple(
    index for index in base.CASES if index not in base.TUNING_CASES
)

METHODS = (
    "dense",
    "q_reuses_k_layout",
    "q_reuses_k_exact_radius0",
    "q_reuses_k_exact_radius1",
)
# The base protocol records its baseline config even when baseline is not a
# selected arm, so retain that registry entry for protocol construction.
base.ARMS = {name: base.ARMS[name] for name in (*METHODS, "baseline")}
base.DEFAULT_ARMS = METHODS


def selected_cases():
    return base.import_pipeline().load_cases(
        SAMPLES_50, list(base.CASES), expected_indices=base.CASES
    )


base.selected_cases = selected_cases
_base_fingerprint = base.implementation_fingerprint


def implementation_fingerprint():
    result = _base_fingerprint()
    result["adapter"] = {"path": str(ADAPTER), "sha256": base.sha256(ADAPTER)}
    result["samples_50_sha256"] = base.sha256(SAMPLES_50)
    result["samples_25_sha256"] = base.sha256(SAMPLES_25)
    return result


base.implementation_fingerprint = implementation_fingerprint


def normalized_config(value: dict) -> dict:
    """Normalize historical aliases that are semantically the current default."""
    value = dict(value)
    compression = value.pop("landmark_tree_v2_landmark_compression", None)
    if compression not in (None, "midpoint"):
        raise RuntimeError(f"non-default historical landmark compression: {compression}")
    value.setdefault("sol_exact_block_radius", None)
    return value


def source_records(root: Path, arm: str) -> dict[str, tuple[Path, Path, dict]]:
    records = {}
    for record_path in sorted((root / "records" / arm).glob("*.json")):
        record = base.read_json(record_path)
        sample_id = record["sample_id"]
        if sample_id in records:
            raise RuntimeError(f"duplicate source sample {arm}/{sample_id}")
        latent = Path(record["latent_path"])
        if not latent.is_file() or base.sha256(latent) != record["latent_sha256"]:
            raise RuntimeError(f"invalid source latent {arm}/{sample_id}")
        records[sample_id] = (latent, record_path, record)
    return records


def seed_sparse_results(names: tuple[str, ...]) -> None:
    expected_configs = json.loads(json.dumps(base.validate_arm_configs(names)))
    cases = {case["sample_id"]: case for case in selected_cases()}
    sources = {
        "q_reuses_k_layout": (SOURCE_50, ids_50),
        "q_reuses_k_exact_radius0": (SOURCE_RADIUS_25, ids_25),
        "q_reuses_k_exact_radius1": (SOURCE_RADIUS_25, ids_25),
    }
    reused = []
    for arm, (root, expected_ids) in sources.items():
        if arm not in names:
            continue
        records = source_records(root, arm)
        if set(records) != expected_ids:
            raise RuntimeError(
                f"source identity mismatch for {arm}: got={len(records)} "
                f"expected={len(expected_ids)}"
            )
        for sample_id, (source_latent, source_record_path, record) in records.items():
            case = cases[sample_id]
            expected_identity = {
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
            mismatches = {
                key: (record.get(key), expected)
                for key, expected in expected_identity.items()
                if record.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"source identity mismatch {arm}/{sample_id}: {mismatches}"
                )
            if normalized_config(record["attention_config"]) != normalized_config(
                expected_configs[arm]
            ):
                raise RuntimeError(f"source config mismatch {arm}/{sample_id}")
            target_latent, target_record = base.paths(arm, case)
            if target_latent.exists() or target_record.exists():
                if not base.complete_record(arm, case):
                    raise RuntimeError(f"stale target result {arm}/{sample_id}")
                continue
            base.link_or_copy(source_latent, target_latent)
            copied = {
                **record,
                "case": case["index"],
                "latent_path": str(target_latent),
                "reused_source_case": record["case"],
                "reused_from_record": str(source_record_path),
                "reused_from_latent": str(source_latent),
            }
            base.atomic_json(target_record, copied)
            reused.append(
                {
                    "arm": arm,
                    "sample_id": sample_id,
                    "source_case": record["case"],
                    "target_case": case["index"],
                    "latent_sha256": record["latent_sha256"],
                }
            )
    base.atomic_json(
        base.ROOT / "reuse_manifest.json",
        {
            "status": "complete",
            "identity": "sample_id + prompt_sha256 + seed + config + latent_sha256",
            "records": reused,
            "expected_reused_records": 100,
        },
    )


_base_prepare = base.prepare


def prepare(names: tuple[str, ...]) -> None:
    _base_prepare(names)
    seed_sparse_results(names)


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
        command = [
            sys.executable,
            str(ADAPTER),
            "worker",
            "--gpu",
            str(gpu),
            "--arms",
            *names,
        ]
        process = subprocess.Popen(
            command,
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
