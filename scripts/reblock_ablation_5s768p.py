#!/usr/bin/env python3
"""Resume-safe 5s/768p Spark-Reblock ablation runner on eight GPUs.

The runner intentionally keeps experiment orchestration separate from the
production defaults.  Candidate-only configuration fields are applied with
``dataclasses.replace`` and fail closed until their processor implementation
lands; no unsupported arm is silently mapped back to the baseline.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any


DEFAULT_NAME = "reblock_ablation_5s768p_25p_20261002"
NAME = os.environ.get("REB_ABLATION_NAME", DEFAULT_NAME)
REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
BENCH_SCRIPTS = BENCH / "scripts"
MODEL = Path("/mnt/CFS/tangzecheng/models/MiniMax-H3")
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
LEGACY_CONDITIONING = Path("/mnt/CFS/tangzecheng/calibration_f16_tau_20260919")
ROOT = Path(
    os.environ.get(
        "REB_ABLATION_ROOT",
        str(Path("/mnt/CFS/tangzecheng/experiments") / NAME),
    )
).resolve()
CONDITIONING_SOURCE_ROOT = (
    Path(os.environ["REB_CONDITIONING_SOURCE_ROOT"]).resolve()
    if os.environ.get("REB_CONDITIONING_SOURCE_ROOT")
    else None
)
DENSE_SOURCE_ROOT = (
    Path(os.environ["REB_DENSE_SOURCE_ROOT"]).resolve()
    if os.environ.get("REB_DENSE_SOURCE_ROOT")
    else None
)
GPUS = tuple(range(8))
CASES = tuple(range(1, 26))
TUNING_CASES = (1, 2, 3, 8, 9, 10, 11, 17, 18, 19)
CONFIRMATION_CASES = tuple(index for index in CASES if index not in TUNING_CASES)
SEED = 42
STEPS = 20
WARMUP_STEPS = 6  # five evaluations: four dense warmup plus one sparse
FRAMES, HEIGHT, WIDTH = 120, 768, 1344


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    axis: str
    description: str
    overrides: dict[str, Any] = dataclasses.field(default_factory=dict)


# These names form the narrow adapter between this runner and candidate
# processor work.  Updating an interface should require editing this table,
# not the worker or result code.
F = {
    "m2_sides": "landmark_tree_v2_m2_side",
    "seed_rule": "landmark_tree_v2_proxy_seed_rule",
    "center_update_rule": "landmark_tree_v2_proxy_update_rule",
    "update_rounds": "landmark_tree_v2_proxy_iterations",
    "moment_estimator": "landmark_tree_v2_m2_estimator",
    "layout_reuse": "landmark_tree_v2_layout_reuse",
}


def _arms() -> dict[str, ArmSpec]:
    """Return the frozen single-axis matrix (plus the required M2 2x2)."""

    return {
        "dense": ArmSpec("reference", "Matched dense reference"),
        "baseline": ArmSpec(
            "baseline",
            "Current Spark(20 steps): both-side M2, cosine, flat, L=32, two updates",
        ),
        # M2 weighting x distance: complete 2x2 including the baseline.
        "m2_both_euclidean": ArmSpec(
            "m2_x_distance", "Both-side M2 weighting, Euclidean distance",
            {"landmark_tree_v2_distance": "euclidean"},
        ),
        "m2_none_cosine": ArmSpec(
            "m2_x_distance", "Unweighted clustering features, cosine distance",
            {F["m2_sides"]: "none"},
        ),
        "m2_none_euclidean": ArmSpec(
            "m2_x_distance", "Unweighted clustering features, Euclidean distance",
            {F["m2_sides"]: "none", "landmark_tree_v2_distance": "euclidean"},
        ),
        "m2_q_only_cosine": ArmSpec(
            "m2_side", "Only Q features use opposite-side K M2",
            {F["m2_sides"]: "query"},
        ),
        "m2_k_only_cosine": ArmSpec(
            "m2_side", "Only K features use opposite-side Q M2",
            {F["m2_sides"]: "key"},
        ),
        # One seed-selection alternative and one iterative-center alternative;
        # their definitions/costs must be documented before execution.
        "seed_endpoint_order": ArmSpec(
            "center_rule", "Seed selection: endpoints in inherited token order",
            {F["seed_rule"]: "endpoint_order"},
        ),
        "update_weighted_medoid": ArmSpec(
            "center_rule", "Center update: weighted medoid; baseline seeds unchanged",
            {F["center_update_rule"]: "medoid"},
        ),
        "landmarks64": ArmSpec(
            "landmark_count", "64 midpoint landmarks",
            {"landmark_tree_v2_landmark_count": 64},
        ),
        "landmarks128": ArmSpec(
            "landmark_count", "128 midpoint landmarks",
            {"landmark_tree_v2_landmark_count": 128},
        ),
        "updates0": ArmSpec(
            "update_rounds", "Zero assignment+center-update rounds",
            {F["update_rounds"]: 0},
        ),
        "updates1": ArmSpec(
            "update_rounds", "One assignment+center-update round",
            {F["update_rounds"]: 1},
        ),
        "updates4": ArmSpec(
            "update_rounds", "Four assignment+center-update rounds",
            {F["update_rounds"]: 4},
        ),
        "m2_flat64_block_mean": ArmSpec(
            "m2_estimator", "Flat64 block-mean outer products",
            {F["moment_estimator"]: "flat64_block_mean"},
        ),
        "m2_flat64_midpoint1": ArmSpec(
            "m2_estimator", "One deterministic flat64 midpoint token per block",
            {F["moment_estimator"]: "flat64_midpoint_1"},
        ),
        "m2_flat64_midpoint2": ArmSpec(
            "m2_estimator", "Two deterministic flat64 midpoint tokens per block",
            {F["moment_estimator"]: "flat64_midpoint_2"},
        ),
        "m2_flat64_midpoint4": ArmSpec(
            "m2_estimator", "Four deterministic flat64 midpoint tokens per block",
            {F["moment_estimator"]: "flat64_midpoint_4"},
        ),
        "m2_flat64_mean_diag": ArmSpec(
            "m2_estimator", "Flat64 block means plus exact diagonal compensation",
            {F["moment_estimator"]: "flat64_mean_diag"},
        ),
        "m2_full": ArmSpec(
            "m2_estimator", "Full non-centered second moment over video tokens",
            {F["moment_estimator"]: "full"},
        ),
        "q_reuses_k_layout": ArmSpec(
            "layout_reuse", "Build only K layout and reuse its permutation for Q",
            {F["layout_reuse"]: "q_from_k"},
        ),
        "k_reuses_q_layout": ArmSpec(
            "layout_reuse", "Build only Q layout and reuse its permutation for K/V",
            {F["layout_reuse"]: "k_from_q"},
        ),
        "initial_hilbert_thw": ArmSpec(
            "initial_order", "Root initial order hilbert_thw",
            {"landmark_tree_v2_initial_order": "hilbert_thw"},
        ),
        "initial_tile_t4h4w4": ArmSpec(
            "initial_order", "Root initial order matching FastH3 tile_t4h4w4",
            {"landmark_tree_v2_initial_order": "tile_t4h4w4"},
        ),
    }


ARMS = _arms()
DEFAULT_ARMS = tuple(ARMS)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)
            + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def link_or_copy(source: Path, target: Path) -> None:
    """Materialize an immutable input cheaply, falling back across filesystems."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def command_output(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint() -> dict[str, Any]:
    diff = command_output("git", "diff", "--binary", "HEAD", cwd=REPO)
    relevant = [
        REPO / "h3_sparse_attention/processor.py",
        REPO / "h3_sparse_attention/spark_integration.py",
        REPO / "h3_sparse_attention/landmark_tree_v2.py",
        Path(__file__).resolve(),
    ]
    source_files = sorted(
        path
        for package in (REPO / "h3_sparse_attention", REPO / "sol_attn")
        for path in package.rglob("*.py")
    )
    tree_digest = hashlib.sha256()
    for path in source_files:
        tree_digest.update(str(path.relative_to(REPO)).encode())
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(sha256(path)))
    return {
        "path": str(REPO),
        "revision": command_output("git", "rev-parse", "HEAD", cwd=REPO),
        "status": command_output("git", "status", "--porcelain", cwd=REPO),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "source_tree_sha256": tree_digest.hexdigest(),
        "source_tree_file_count": len(source_files),
        "files": {str(path): sha256(path) for path in relevant},
    }


def gpu_inventory() -> list[dict[str, Any]]:
    lines = command_output(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ).splitlines()
    inventory = []
    for line in lines:
        index, name, uuid, memory_mib = (part.strip() for part in line.split(","))
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "memory_total_mib": int(memory_mib),
            }
        )
    return inventory


def import_pipeline():
    os.environ.setdefault("H3_IMPL_REPO", str(REPO))
    os.environ.setdefault("H3_DIFFUSERS_DIR", str(MODEL))
    if str(BENCH_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BENCH_SCRIPTS))
    import _impl_bootstrap

    if Path(_impl_bootstrap.IMPL_REPO).resolve() != REPO.resolve():
        raise RuntimeError(
            f"implementation mismatch: {_impl_bootstrap.IMPL_REPO} != {REPO}"
        )
    import minimax_h3_vbench_4gpu_pipeline as pipeline

    return pipeline


def selected_cases() -> list[dict[str, Any]]:
    return import_pipeline().load_cases(
        SAMPLES, list(CASES), expected_indices=tuple(range(1, 26))
    )


def normalize_arms(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    names = tuple(names or DEFAULT_ARMS)
    unknown = sorted(set(names) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    # Every sparse comparison needs the one matched dense reference.
    return (("dense",) if "dense" not in names else ()) + names


def arm_config(name: str):
    if name == "dense":
        return None
    p = import_pipeline()
    # Explicitly pin every baseline dimension requested by the protocol.  Two
    # center-update rounds are the current implementation's built-in behavior;
    # use the field too once it exists.
    baseline_kwargs = {
        "landmark_tree_v2_fanout": 16,
        "landmark_tree_v2_landmark_count": 32,
        "landmark_tree_v2_initial_order": "flat",
        "landmark_tree_v2_distance": "cosine",
        "landmark_tree_v2_order_mode": "parent_order",
    }
    config = p.H3SparseAttentionConfig.spark(STEPS, **baseline_kwargs)
    fields = {field.name for field in dataclasses.fields(config)}
    optional_current_defaults = {
        F["m2_sides"]: "both",
        F["seed_rule"]: "farthest_pair",
        F["center_update_rule"]: "mean",
        F["update_rounds"]: 2,
        F["moment_estimator"]: "hilbert_midpoint",
        F["layout_reuse"]: "independent",
    }
    config = dataclasses.replace(
        config, **{key: value for key, value in optional_current_defaults.items() if key in fields}
    )
    overrides = ARMS[name].overrides
    missing = sorted(set(overrides) - fields)
    if missing:
        raise RuntimeError(
            f"arm {name!r} requires processor config fields not yet available: {missing}"
        )
    return dataclasses.replace(config, **overrides)


def validate_arm_configs(names: tuple[str, ...]) -> dict[str, Any]:
    rendered = {}
    failures = {}
    for name in names:
        try:
            config = arm_config(name)
            rendered[name] = None if config is None else dataclasses.asdict(config)
        except Exception as error:  # expose all missing interfaces together
            failures[name] = f"{type(error).__name__}: {error}"
    if failures:
        raise RuntimeError("unsupported arm configuration:\n" + json.dumps(failures, indent=2))
    return rendered


def migrate_conditioning() -> None:
    """Migrate the trusted ten-prompt schema-v1 cache to Benchmark schema v2."""
    import torch
    # ``_h3_inference_common`` lives beside the Benchmark pipeline.  Ensure
    # that directory is registered even when ``prepare`` is the first command
    # invoked in a clean interpreter.
    import_pipeline()
    import _h3_inference_common as common

    selected = selected_cases()
    if CONDITIONING_SOURCE_ROOT is not None:
        source_manifest = read_json(CONDITIONING_SOURCE_ROOT / "conditioning_manifest.json")
        by_index = {int(row["index"]): row for row in source_manifest["records"]}
        target_dir = ROOT / "conditioning_cache"
        target_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = common._conditioning_model_fingerprint(MODEL)
        records = []
        for case in selected:
            metadata = by_index.get(case["index"])
            if metadata is not None and (
                metadata["sample_id"] != case["sample_id"]
                or metadata["prompt_sha256"] != case["prompt_sha256"]
            ):
                raise RuntimeError(f"conditioning identity mismatch for case {case['index']}")
            source = common._conditioning_cache_path(
                CONDITIONING_SOURCE_ROOT / "conditioning_cache", fingerprint, case["prompt"]
            )
            if metadata is not None and sha256(source) != metadata["sha256"]:
                raise RuntimeError(f"conditioning hash mismatch for case {case['index']}")
            # A legacy partial manifest may omit newly populated cases.  The
            # cache payload itself remains authoritative and is validated
            # below against both the model fingerprint and exact prompt.
            if not source.is_file():
                raise FileNotFoundError(f"missing conditioning source for case {case['index']}")
            target = common._conditioning_cache_path(target_dir, fingerprint, case["prompt"])
            link_or_copy(source, target)
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
                    "source": str(source),
                    "path": str(target),
                    "sha256": sha256(target),
                    "summary": summary,
                }
            )
        atomic_json(
            ROOT / "conditioning_manifest.json",
            {
                "status": "complete",
                "schema": 2,
                "model": str(MODEL),
                "reused_from": str(CONDITIONING_SOURCE_ROOT),
                "records": records,
                "expected_records": len(selected),
            },
        )
        return

    legacy = read_json(LEGACY_CONDITIONING / "conditioning_manifest.json")
    by_index = {int(row["index"]): row for row in legacy["records"]}
    target_dir = ROOT / "conditioning_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = common._conditioning_model_fingerprint(MODEL)
    records = []
    for case in selected:
        metadata = by_index.get(case["index"])
        if metadata is None:
            # The trusted legacy cache contains prompts 1--10 only.  Missing
            # entries are populated once by the Benchmark conditioning stage
            # before any eight-GPU denoise/capture workers are launched.
            continue
        if (
            metadata["sample_id"] != case["sample_id"]
            or metadata["prompt_sha256"] != case["prompt_sha256"]
        ):
            raise RuntimeError(f"conditioning identity mismatch for case {case['index']}")
        source = LEGACY_CONDITIONING / "conditioning_cache" / f"{case['index']:02d}.pt"
        target = common._conditioning_cache_path(target_dir, fingerprint, case["prompt"])
        if not target.exists():
            state = torch.load(source, map_location="cpu", weights_only=False)
            common._save_conditioning_state(
                state, target, model_fingerprint=fingerprint, prompt=case["prompt"]
            )
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
                "source": str(source),
                "path": str(target),
                "sha256": sha256(target),
                "summary": summary,
            }
        )
    atomic_json(
        ROOT / "conditioning_manifest.json",
        {
            "status": "complete" if len(records) == len(selected) else "partial",
            "schema": 2,
            "model": str(MODEL),
            "records": records,
            "expected_records": len(selected),
        },
    )


def seed_dense_reference() -> None:
    """Reuse the matched Dense trajectory without rerunning it in rolling roots."""
    if DENSE_SOURCE_ROOT is None:
        return
    for case in selected_cases():
        stem = f"{case['index']:02d}_{case['sample_id']}"
        source_latent = DENSE_SOURCE_ROOT / "latents" / "dense" / f"{stem}.pt"
        source_record = DENSE_SOURCE_ROOT / "records" / "dense" / f"{stem}.json"
        if not source_latent.is_file() or not source_record.is_file():
            raise FileNotFoundError(f"missing Dense source for case {case['index']}")
        record = read_json(source_record)
        if (
            record.get("status") != "complete"
            or record.get("arm") != "dense"
            or record.get("case") != case["index"]
            or record.get("sample_id") != case["sample_id"]
            or record.get("prompt_sha256") != case["prompt_sha256"]
            or record.get("latent_sha256") != sha256(source_latent)
        ):
            raise RuntimeError(f"invalid Dense source record for case {case['index']}")
        target_latent = ROOT / "latents" / "dense" / f"{stem}.pt"
        target_record = ROOT / "records" / "dense" / f"{stem}.json"
        link_or_copy(source_latent, target_latent)
        copied = {
            **record,
            "latent_path": str(target_latent),
            "reused_dense_reference_from": str(source_latent),
        }
        atomic_json(target_record, copied)


def prepare(names: tuple[str, ...]) -> None:
    if not MODEL.is_dir() or not SAMPLES.is_file():
        raise FileNotFoundError(f"missing model or samples: {MODEL}, {SAMPLES}")
    ROOT.mkdir(parents=True, exist_ok=True)
    migrate_conditioning()
    fingerprint = implementation_fingerprint()
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists():
        old = read_json(protocol_path)
        if old["implementation"] != fingerprint:
            raise RuntimeError("implementation changed since protocol was prepared")
        if tuple(old["selected_arms"]) != names:
            raise RuntimeError("selected arms differ from the prepared protocol")
        return
    for name in names:
        for kind in ("records", "latents", "warmup"):
            (ROOT / kind / name).mkdir(parents=True, exist_ok=True)
    seed_dense_reference()
    # Rendering configs is deliberately deferred to workers: prepare remains
    # useful while candidate fields are being integrated, but workers fail
    # before loading 62 GiB of weights if any selected field is absent.
    baseline = dataclasses.asdict(arm_config("baseline"))
    protocol = {
        "name": NAME,
        "status": "prepared",
        "purpose": "Paired Spark-Reblock ablation using the 5s768p proxy",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_gpus": list(GPUS),
        "gpu_inventory": gpu_inventory(),
        "case_assignment": {
            str(gpu): [case["index"] for case in selected_cases()[gpu::len(GPUS)]]
            for gpu in GPUS
        },
        "cases": selected_cases(),
        "tuning_cases": list(TUNING_CASES),
        "confirmation_cases": list(CONFIRMATION_CASES),
        "selected_arms": list(names),
        "arm_registry": {name: dataclasses.asdict(spec) for name, spec in ARMS.items()},
        "settings": {
            "frames": FRAMES,
            "height": HEIGHT,
            "width": WIDTH,
            "seed": SEED,
            "requested_steps": STEPS,
            "expected_transformer_evaluations": STEPS - 1,
            "warmup_steps": WARMUP_STEPS,
            "topk_ratio": 0.1,
            "exact_tile": [64, 64],
            "route_score": "native_mean",
            "model_dtype": "bfloat16",
            "timing": "CUDA-synchronized denoiser; excludes load, warmup, CPU copy/save",
        },
        "baseline_config": baseline,
        "baseline_contract": {
            "fanout": 16,
            "landmark_count": 32,
            "update_rounds": 2,
            "update_rounds_source": (
                "current implementation built-in when config field is absent"
            ),
            "initial_order": "flat",
            "distance": "cosine",
            "m2_weighted_sides": "both",
            "moment_estimator": "hilbert_midpoint",
            "layout_reuse": "independent",
        },
        "model": str(MODEL),
        "samples": str(SAMPLES),
        "conditioning_manifest": str(ROOT / "conditioning_manifest.json"),
        "conditioning_source_root": (
            None if CONDITIONING_SOURCE_ROOT is None else str(CONDITIONING_SOURCE_ROOT)
        ),
        "dense_source_root": None if DENSE_SOURCE_ROOT is None else str(DENSE_SOURCE_ROOT),
        "implementation": fingerprint,
        "allocator": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "pairing": "fixed prompt/seed; every selected arm and dense reference per prompt",
    }
    atomic_json(protocol_path, protocol)
    atomic_json(ROOT / "status.json", {"status": "prepared"})


def paths(name: str, case: dict[str, Any]) -> tuple[Path, Path]:
    stem = f"{case['index']:02d}_{case['sample_id']}"
    return ROOT / "latents" / name / f"{stem}.pt", ROOT / "records" / name / f"{stem}.json"


def complete_record(name: str, case: dict[str, Any]) -> bool:
    latent, record = paths(name, case)
    if not latent.is_file() or not record.is_file():
        return False
    try:
        payload = read_json(record)
        return (
            payload["status"] == "complete"
            and payload["arm"] == name
            and payload["case"] == case["index"]
            and payload["latent_sha256"] == sha256(latent)
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def run_generation(pipe, plugin, state, steps: int):
    import torch

    p = import_pipeline()
    plugin.reset()
    inference_state = p.clone_state(state)
    inference_state.values["prompt_embeds"] = inference_state.values[
        "prompt_embeds"
    ].to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = pipe(
        state=inference_state,
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        output=["latents", "audio_latents"],
    )
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def assert_frozen_protocol(names: tuple[str, ...]) -> dict[str, Any]:
    protocol = read_json(ROOT / "protocol.json")
    if tuple(protocol["selected_arms"]) != names:
        raise RuntimeError("worker arm selection differs from protocol")
    if protocol["implementation"] != implementation_fingerprint():
        raise RuntimeError("implementation fingerprint changed after prepare")
    return protocol


def worker(gpu: int, names: tuple[str, ...]) -> None:
    import torch

    p = import_pipeline()
    import _h3_inference_common as common
    protocol = assert_frozen_protocol(names)
    configs = validate_arm_configs(names)  # fail before the expensive model load
    assigned = selected_cases()[gpu::len(GPUS)]
    if not assigned:
        raise RuntimeError(f"GPU {gpu} has no assigned prompts")
    args = SimpleNamespace(
        model=MODEL,
        output=ROOT,
        transformer_placement="resident",
        transformer_offload_dir=ROOT / "offload" / f"gpu{gpu}",
        resident_blocks=50,
    )
    workflow, states = p.configure_denoise_workflow(args, assigned)
    pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    # Rotate arm order across cards to reduce systematic thermal/order bias.
    ordered = list(names)
    shift = gpu % len(ordered)
    ordered = ordered[shift:] + ordered[:shift]
    try:
        for name in ordered:
            pending = [
                (case, state)
                for case, state in zip(assigned, states, strict=True)
                if not complete_record(name, case)
            ]
            if not pending:
                print(f"REUSE gpu={gpu} arm={name} all assigned cases", flush=True)
                continue
            config = None if name == "dense" else arm_config(name)
            context = p.DensePlugin() if config is None else p.install_h3_sparse_attention(
                pipe.transformer, config
            )
            with context as plugin:
                torch.cuda.empty_cache()
                warm, warm_seconds = run_generation(pipe, plugin, pending[0][1], WARMUP_STEPS)
                del warm
                atomic_json(
                    ROOT / "warmup" / name / f"gpu{gpu}.json",
                    {
                        "status": "complete",
                        "physical_gpu": gpu,
                        "arm": name,
                        "case": pending[0][0]["index"],
                        "steps": WARMUP_STEPS,
                        "seconds": warm_seconds,
                    },
                )
                for case, state in pending:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    result, seconds = run_generation(pipe, plugin, state, STEPS)
                    summary = plugin.summary()
                    if name != "dense" and summary.get("completed_evaluations") != STEPS - 1:
                        raise RuntimeError(f"incomplete attention evaluations: {summary}")
                    payload = {
                        "metadata": {
                            "arm": name,
                            "case": case["index"],
                            "sample_id": case["sample_id"],
                            "prompt_sha256": case["prompt_sha256"],
                            "seed": SEED,
                            "steps": STEPS,
                            "frames": FRAMES,
                            "height": HEIGHT,
                            "width": WIDTH,
                        },
                        "latents": result["latents"].detach().cpu(),
                        "audio_latents": result["audio_latents"].detach().cpu(),
                    }
                    del result
                    latent_path, record_path = paths(name, case)
                    common.atomic_torch_save(payload, latent_path)
                    record = {
                        **payload["metadata"],
                        "status": "complete",
                        "physical_gpu": gpu,
                        "gpu_uuid": protocol["gpu_inventory"][gpu]["uuid"],
                        "denoise_seconds": seconds,
                        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "latent_path": str(latent_path),
                        "latent_sha256": sha256(latent_path),
                        "transformer_placement": placement,
                        "torch_compile": True,
                        "compile_wrapped_blocks": len(
                            getattr(acceleration, "_forward_originals", ())
                        ),
                        "arm_spec": dataclasses.asdict(ARMS[name]),
                        "attention_config": configs[name],
                        "attention_summary": summary,
                    }
                    atomic_json(record_path, record)
                    print(
                        f"DONE gpu={gpu} arm={name} case={case['index']} seconds={seconds:.3f}",
                        flush=True,
                    )
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def latent_metrics(candidate: Path, reference: Path) -> dict[str, Any]:
    import torch

    a = torch.load(candidate, map_location="cpu", weights_only=True)["latents"].float()
    b = torch.load(reference, map_location="cpu", weights_only=True)["latents"].float()
    if a.shape != b.shape:
        raise ValueError(f"latent shape mismatch: {a.shape} != {b.shape}")
    diff = a - b
    mse = float(diff.square().mean())
    dynamic_range = float(b.max() - b.min())
    return {
        "mse": mse,
        "relative_l2": float(diff.norm() / b.norm().clamp_min(1e-12)),
        "latent_psnr_db": float("inf") if mse == 0 else 20 * math.log10(dynamic_range / math.sqrt(mse)),
        "shape": list(a.shape),
    }


def summarize(names: tuple[str, ...]) -> None:
    protocol = assert_frozen_protocol(names)
    rows = []
    for case in selected_cases():
        dense_latent, _ = paths("dense", case)
        if not dense_latent.is_file():
            raise FileNotFoundError(f"missing dense reference for case {case['index']}")
        for name in names:
            latent, record = paths(name, case)
            if not complete_record(name, case):
                raise FileNotFoundError(f"incomplete arm={name} case={case['index']}")
            row = read_json(record)
            row["latent_metrics_vs_dense"] = (
                {"mse": 0.0, "relative_l2": 0.0, "latent_psnr_db": None}
                if name == "dense"
                else latent_metrics(latent, dense_latent)
            )
            rows.append(row)
    means = {}
    for name in names:
        arm_rows = [row for row in rows if row["arm"] == name]
        means[name] = {
            "samples": len(arm_rows),
            "mean_seconds": statistics.fmean(row["denoise_seconds"] for row in arm_rows),
            "median_seconds": statistics.median(row["denoise_seconds"] for row in arm_rows),
            "mean_peak_cuda_allocated_gib": statistics.fmean(
                row["peak_cuda_allocated_gib"] for row in arm_rows
            ),
            "mean_relative_l2": statistics.fmean(
                row["latent_metrics_vs_dense"]["relative_l2"] for row in arm_rows
            ),
            "mean_latent_psnr_db": None if name == "dense" else statistics.fmean(
                row["latent_metrics_vs_dense"]["latent_psnr_db"] for row in arm_rows
            ),
        }
    result = {
        "status": "complete",
        "protocol": str(ROOT / "protocol.json"),
        "sample_count": len(selected_cases()),
        "means": means,
        "records": rows,
    }
    atomic_json(ROOT / "results.json", result)
    protocol["status"] = "complete"
    atomic_json(ROOT / "protocol.json", protocol)
    atomic_json(ROOT / "status.json", {"status": "complete"})


def record_failure(stage: str, error: BaseException, gpu: int | None = None) -> None:
    payload = {
        "status": "failed",
        "stage": stage,
        "gpu": gpu,
        "pid": os.getpid(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    suffix = f"gpu{gpu}" if gpu is not None else "orchestrator"
    atomic_json(ROOT / "failures" / f"{suffix}_{time.time_ns()}.json", payload)
    atomic_json(ROOT / "last_error.json", payload)


def orchestrate(names: tuple[str, ...]) -> None:
    prepare(names)
    # Validate candidate interfaces before launching eight 62-GiB model loads.
    validate_arm_configs(names)
    atomic_json(ROOT / "status.json", {"status": "running", "arms": list(names)})
    env_base = {
        **os.environ,
        "H3_IMPL_REPO": str(REPO),
        "H3_DIFFUSERS_DIR": str(MODEL),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    processes = []
    for gpu in GPUS:
        log = (ROOT / f"worker_gpu{gpu}.log").open("a")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--gpu",
            str(gpu),
            "--arms",
            *names,
        ]
        process = subprocess.Popen(
            command,
            cwd=REPO,
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
        atomic_json(ROOT / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"worker failures: {failures}")
    summarize(names)


def status() -> None:
    payload = {
        "root": str(ROOT),
        "status": read_json(ROOT / "status.json") if (ROOT / "status.json").exists() else None,
        "gpus": [],
    }
    for gpu in GPUS:
        log = ROOT / f"worker_gpu{gpu}.log"
        done = []
        if log.exists():
            done = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("DONE ")]
        payload["gpus"].append(
            {"gpu": gpu, "completed": len(done), "last": done[-1] if done else None}
        )
    print(json.dumps(payload, indent=2))


def list_arms() -> None:
    print(json.dumps({name: dataclasses.asdict(spec) for name, spec in ARMS.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "worker", "summarize", "status", "list-arms")
    )
    parser.add_argument("--gpu", type=int, choices=GPUS)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARMS),
        help="Selected arms; dense is added automatically. Default: the full frozen matrix.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "status":
        status()
        return
    if args.command == "list-arms":
        list_arms()
        return
    names = normalize_arms(args.arms)
    try:
        if args.command == "prepare":
            prepare(names)
        elif args.command == "run":
            orchestrate(names)
        elif args.command == "worker":
            if args.gpu is None:
                parser.error("worker requires --gpu")
            worker(args.gpu, names)
        elif args.command == "summarize":
            summarize(names)
    except Exception as error:
        record_failure(args.command, error, args.gpu)
        raise


if __name__ == "__main__":
    main()
