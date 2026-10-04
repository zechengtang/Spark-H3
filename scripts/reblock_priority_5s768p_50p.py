#!/usr/bin/env python3
"""Run the six-method 5s/768p comparison on the official 50-prompt subset.

The official 10%/25-prompt list is a sample-id subset of the 20%/50-prompt
list, but its case indices differ.  This adapter therefore reuses prior
latents by immutable sample identity rather than by case number and computes
only the missing prompt-method pairs (plus the previously absent Sol arm).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


os.environ.setdefault("REB_ABLATION_NAME", "reblock_priority_5s768p_50p_20261003")
os.environ.setdefault(
    "REB_ABLATION_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_50p_20261003",
)

import reblock_ablation_5s768p as base  # noqa: E402


ADAPTER = Path(__file__).resolve()
SAMPLES_25 = base.BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
SAMPLES_50 = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
CONDITIONING_SOURCE = Path(
    "/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002"
)
SOURCE_ROOTS = {
    "dense": CONDITIONING_SOURCE,
    "baseline": CONDITIONING_SOURCE,
    "landmarks64": CONDITIONING_SOURCE,
    "q_reuses_k_layout": Path(
        "/mnt/CFS/tangzecheng/experiments/"
        "reblock_ablation_5s768p_25p_roll11_q_reuses_k_layout_20261002"
    ),
    "k_reuses_q_layout": Path(
        "/mnt/CFS/tangzecheng/experiments/"
        "reblock_ablation_5s768p_25p_roll15_priority4_20261003"
    ),
}

base.SAMPLES = SAMPLES_50
base.CASES = tuple(range(1, 51))
ids_25 = {row["sample_id"] for row in json.loads(SAMPLES_25.read_text())}
rows_50 = json.loads(SAMPLES_50.read_text())
if len(rows_50) != 50 or len({row["sample_id"] for row in rows_50}) != 50:
    raise RuntimeError("invalid official 50-prompt sample list")
if not ids_25 <= {row["sample_id"] for row in rows_50}:
    raise RuntimeError("official 25-prompt list is not a subset of the 50-prompt list")
base.TUNING_CASES = tuple(row["index"] for row in rows_50 if row["sample_id"] in ids_25)
base.CONFIRMATION_CASES = tuple(index for index in base.CASES if index not in base.TUNING_CASES)
base.CONDITIONING_SOURCE_ROOT = None
base.DENSE_SOURCE_ROOT = None

METHODS = (
    "dense",
    "sol",
    "baseline",
    "landmarks64",
    "q_reuses_k_layout",
    "k_reuses_q_layout",
)
base.ARMS = {name: base.ARMS[name] for name in METHODS if name != "sol"}
base.ARMS["sol"] = base.ArmSpec(
    "reference", "Stock Sol-Attn baseline from the synced Benchmark protocol"
)
base.ARMS = {name: base.ARMS[name] for name in METHODS}
base.DEFAULT_ARMS = METHODS


def selected_cases():
    return base.import_pipeline().load_cases(
        SAMPLES_50, list(base.CASES), expected_indices=base.CASES
    )


base.selected_cases = selected_cases
_base_arm_config = base.arm_config


def arm_config(name: str):
    if name != "sol":
        return _base_arm_config(name)
    pipeline = base.import_pipeline()
    return pipeline.H3SparseAttentionConfig.sol(
        base.STEPS, warmup_percent=20.0, sol_dense_layers=1
    )


base.arm_config = arm_config
_base_fingerprint = base.implementation_fingerprint


def implementation_fingerprint():
    result = _base_fingerprint()
    result["adapter"] = {"path": str(ADAPTER), "sha256": base.sha256(ADAPTER)}
    result["samples_50_sha256"] = base.sha256(SAMPLES_50)
    result["samples_25_sha256"] = base.sha256(SAMPLES_25)
    return result


base.implementation_fingerprint = implementation_fingerprint


def migrate_conditioning() -> None:
    """Reuse the 25 known prompts by prompt-key and leave the other 25 pending."""
    base.import_pipeline()
    import _h3_inference_common as common

    target_dir = base.ROOT / "conditioning_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = common._conditioning_model_fingerprint(base.MODEL)
    records = []
    for case in selected_cases():
        target = common._conditioning_cache_path(target_dir, fingerprint, case["prompt"])
        source = common._conditioning_cache_path(
            CONDITIONING_SOURCE / "conditioning_cache", fingerprint, case["prompt"]
        )
        if not target.exists() and source.is_file():
            base.link_or_copy(source, target)
        if not target.is_file():
            continue
        state = common._load_conditioning_state(
            target, model_fingerprint=fingerprint, prompt=case["prompt"]
        )
        summary = common._conditioning_values_summary(
            state.values, prompt=case["prompt"], source=str(target)
        )
        records.append(
            {
                "index": case["index"],
                "sample_id": case["sample_id"],
                "prompt_sha256": case["prompt_sha256"],
                "source": str(source) if source.is_file() else "computed-in-target",
                "path": str(target),
                "sha256": base.sha256(target),
                "summary": summary,
            }
        )
    base.atomic_json(
        base.ROOT / "conditioning_manifest.json",
        {
            "status": "complete" if len(records) == 50 else "partial",
            "schema": 2,
            "model": str(base.MODEL),
            "reused_source": str(CONDITIONING_SOURCE),
            "records": records,
            "expected_records": 50,
        },
    )


base.migrate_conditioning = migrate_conditioning


def source_records(arm: str) -> dict[str, tuple[Path, dict]]:
    root = SOURCE_ROOTS[arm]
    result = {}
    for record_path in sorted((root / "records" / arm).glob("*.json")):
        record = base.read_json(record_path)
        sample_id = record.get("sample_id")
        if sample_id in result:
            raise RuntimeError(f"duplicate source sample for {arm}: {sample_id}")
        latent = Path(record["latent_path"])
        if not latent.is_file():
            raise FileNotFoundError(latent)
        result[sample_id] = (latent, record)
    if set(result) != ids_25:
        raise RuntimeError(
            f"source sample mismatch for {arm}: got={len(result)} expected={len(ids_25)}"
        )
    return result


def seed_existing_results(names: tuple[str, ...]) -> None:
    configs = base.validate_arm_configs(names)
    # Historical records are JSON, so tuple-valued dataclass fields were
    # serialized as lists.  Compare the current render in that same canonical
    # representation instead of treating list-vs-tuple as a semantic change.
    configs = json.loads(json.dumps(configs))
    cases = {case["sample_id"]: case for case in selected_cases()}
    reused = []
    for arm in names:
        if arm == "sol":
            continue
        for sample_id, (source_latent, record) in source_records(arm).items():
            case = cases[sample_id]
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
            mismatches = {
                key: (record.get(key), value)
                for key, value in expected.items()
                if record.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"source identity mismatch {arm}/{sample_id}: {mismatches}")
            if record.get("attention_config") != configs[arm]:
                raise RuntimeError(f"source config mismatch {arm}/{sample_id}")
            source_hash = base.sha256(source_latent)
            if record.get("latent_sha256") != source_hash:
                raise RuntimeError(f"source latent hash mismatch {arm}/{sample_id}")
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
                "reused_from_record": str(
                    SOURCE_ROOTS[arm] / "records" / arm /
                    f"{record['case']:02d}_{sample_id}.json"
                ),
                "reused_from_latent": str(source_latent),
            }
            base.atomic_json(target_record, copied)
            reused.append(
                {
                    "arm": arm,
                    "sample_id": sample_id,
                    "source_case": record["case"],
                    "target_case": case["index"],
                    "latent_sha256": source_hash,
                }
            )
    base.atomic_json(
        base.ROOT / "reuse_manifest.json",
        {
            "status": "complete",
            "identity": "sample_id + prompt_sha256 + seed + config + latent_sha256",
            "records": reused,
            "expected_reused_records": 25 * 5,
        },
    )


_base_prepare = base.prepare


def prepare(names: tuple[str, ...]) -> None:
    _base_prepare(names)
    seed_existing_results(names)


base.prepare = prepare


def populate_conditioning() -> None:
    """Populate the missing 25 prompts once before eight denoisers are launched."""
    import torch
    import _h3_inference_common as common

    pipeline = base.import_pipeline()
    args = SimpleNamespace(model=base.MODEL, output=base.ROOT)
    workflow, states = pipeline.configure_denoise_workflow(args, selected_cases())
    del states, workflow
    common.release_cpu_arenas()
    torch.cuda.empty_cache()
    migrate_conditioning()
    manifest = base.read_json(base.ROOT / "conditioning_manifest.json")
    if manifest["status"] != "complete" or len(manifest["records"]) != 50:
        raise RuntimeError("conditioning population did not produce 50 valid records")


def orchestrate(names: tuple[str, ...]) -> None:
    prepare(names)
    base.validate_arm_configs(names)
    populate_conditioning()
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
            sys.executable, str(ADAPTER), "worker", "--gpu", str(gpu), "--arms", *names
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
