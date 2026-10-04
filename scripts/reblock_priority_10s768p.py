#!/usr/bin/env python3
"""Run the priority Reblock variants on the 25-prompt 10s/768p benchmark.

This is a narrow adapter over the frozen 5s ablation runner.  It changes only
the duration/output identity and adds the stock Sol arm required by the VBench
comparison protocol.  The A800 runtime warmup remains the established
four-Dense-plus-one-sparse pass (six requested sigma-grid points); the new
three-point short warmup is currently validated only on SM120.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


os.environ.setdefault("REB_ABLATION_NAME", "reblock_priority_10s768p_25p_20261003")
os.environ.setdefault(
    "REB_ABLATION_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_priority_10s768p_25p_20261003",
)
os.environ.setdefault(
    "REB_CONDITIONING_SOURCE_ROOT",
    "/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002",
)

import reblock_ablation_5s768p as base  # noqa: E402


base.FRAMES = 240
base.WARMUP_STEPS = 6

priority = (
    "dense",
    "sol",
    "baseline",
    "landmarks64",
    "q_reuses_k_layout",
    "k_reuses_q_layout",
)
base.ARMS = {
    name: spec
    for name, spec in base.ARMS.items()
    if name in priority and name != "sol"
}
base.ARMS["sol"] = base.ArmSpec(
    "reference",
    "Stock Sol-Attn baseline required by the VBench core-five protocol",
)
base.ARMS = {name: base.ARMS[name] for name in priority}
base.DEFAULT_ARMS = priority

_base_arm_config = base.arm_config


def arm_config(name: str):
    if name != "sol":
        return _base_arm_config(name)
    pipeline = base.import_pipeline()
    return pipeline.H3SparseAttentionConfig.sol(
        base.STEPS,
        warmup_percent=20.0,
        sol_dense_layers=1,
    )


base.arm_config = arm_config

_base_fingerprint = base.implementation_fingerprint


def implementation_fingerprint():
    result = _base_fingerprint()
    path = Path(__file__).resolve()
    result["adapter"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return result


base.implementation_fingerprint = implementation_fingerprint


SCHEDULER_PLAN = base.ROOT / "arm_major_scheduler_plan.json"


def build_scheduler_plan(names: tuple[str, ...]) -> dict:
    """Freeze a strict all-GPU, one-arm-at-a-time execution plan."""
    protocol = base.assert_frozen_protocol(names)
    cases = base.selected_cases()
    fallback = {
        "dense": 994.0,
        "sol": 687.0,
        "baseline": 598.0,
        "landmarks64": 600.0,
        "q_reuses_k_layout": 572.0,
        "k_reuses_q_layout": 569.0,
    }
    costs = {}
    pending = {}
    completed_identity = []
    for arm in names:
        observed = []
        arm_pending = []
        for case in cases:
            if base.complete_record(arm, case):
                _, record_path = base.paths(arm, case)
                record = base.read_json(record_path)
                observed.append(float(record["denoise_seconds"]))
                completed_identity.append(
                    [arm, case["index"], case["sample_id"], record["latent_sha256"]]
                )
            else:
                arm_pending.append(case["index"])
        costs[arm] = statistics.fmean(observed) if observed else fallback[arm]
        pending[arm] = arm_pending

    # Run expensive arms first.  Every worker observes hard phase barriers, so
    # no GPU starts another arm while a peer is still measuring this one.
    arm_order = sorted(names, key=lambda arm: (-costs[arm], names.index(arm)))
    estimated_rounds = {
        arm: (len(pending[arm]) + len(base.GPUS) - 1) // len(base.GPUS)
        for arm in arm_order
    }
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    estimated_makespan = sum(
        estimated_rounds[arm] * costs[arm] for arm in arm_order
    )
    plan = {
        "schema_version": 1,
        "status": "prepared",
        "strategy": (
            "strict arm-major: all 8 GPUs share one arm; synchronized warmup; "
            "dynamic prompt claims; hard barrier between arms"
        ),
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_arms": list(names),
        "arm_order": arm_order,
        "estimated_seconds_per_job": costs,
        "pending_counts": {arm: len(rows) for arm, rows in pending.items()},
        "pending_case_indices": pending,
        "estimated_rounds": estimated_rounds,
        "estimated_makespan_seconds": estimated_makespan,
        "completed_identity_sha256": hashlib.sha256(
            json.dumps(sorted(completed_identity), separators=(",", ":")).encode()
        ).hexdigest(),
        "protocol": str(base.ROOT / "protocol.json"),
        "protocol_implementation": protocol["implementation"],
    }
    base.atomic_json(SCHEDULER_PLAN, plan)
    return plan


def create_claim(path: Path, payload: dict) -> bool:
    """Atomically claim one prompt on the shared filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return True


def phase_barrier(run_root: Path, phase: str, arm: str, gpu: int) -> None:
    marker_root = run_root / phase / arm
    base.atomic_json(
        marker_root / f"gpu{gpu}.json",
        {"status": "complete", "arm": arm, "physical_gpu": gpu},
    )
    expected = [marker_root / f"gpu{item}.json" for item in base.GPUS]
    abort = run_root / "abort.json"
    while not all(path.is_file() for path in expected):
        if abort.is_file():
            raise RuntimeError(f"scheduler aborted during {phase}/{arm}: {abort.read_text()}")
        time.sleep(1.0)


def scheduled_worker(gpu: int, names: tuple[str, ...]) -> None:
    """Persistent worker using dynamic prompt claims inside strict arm phases."""
    import torch

    pipeline = base.import_pipeline()
    import _h3_inference_common as common

    protocol = base.assert_frozen_protocol(names)
    configs = base.validate_arm_configs(names)
    plan = base.read_json(SCHEDULER_PLAN)
    if tuple(plan["selected_arms"]) != names:
        raise RuntimeError("scheduler plan arm selection differs from worker")
    if plan["protocol_implementation"] != protocol["implementation"]:
        raise RuntimeError("scheduler plan implementation differs from protocol")
    cases_by_index = {case["index"]: case for case in base.selected_cases()}
    assigned = list(cases_by_index.values())
    run_root = base.ROOT / "scheduler_runs" / plan["run_id"]
    args = SimpleNamespace(
        model=base.MODEL,
        output=base.ROOT,
        transformer_placement="resident",
        transformer_offload_dir=base.ROOT / "offload" / f"gpu{gpu}",
        resident_blocks=50,
    )
    workflow, states = pipeline.configure_denoise_workflow(args, assigned)
    state_by_index = {
        case["index"]: state for case, state in zip(assigned, states, strict=True)
    }
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)
    try:
        for arm in plan["arm_order"]:
            pending_indices = list(plan["pending_case_indices"][arm])
            if not pending_indices:
                print(f"SCHEDULER_REUSE_ALL gpu={gpu} arm={arm}", flush=True)
                continue
            config = None if arm == "dense" else base.arm_config(arm)
            context = (
                pipeline.DensePlugin()
                if config is None
                else pipeline.install_h3_sparse_attention(pipe.transformer, config)
            )
            with context as plugin:
                warm_case = cases_by_index[pending_indices[0]]
                torch.cuda.empty_cache()
                warm, warm_seconds = base.run_generation(
                    pipe, plugin, state_by_index[warm_case["index"]], base.WARMUP_STEPS
                )
                del warm
                base.atomic_json(
                    base.ROOT / "warmup" / arm / f"gpu{gpu}.json",
                    {
                        "status": "complete",
                        "scheduler": "strict_arm_major_dynamic",
                        "scheduler_run_id": plan["run_id"],
                        "physical_gpu": gpu,
                        "arm": arm,
                        "case": warm_case["index"],
                        "steps": base.WARMUP_STEPS,
                        "seconds": warm_seconds,
                    },
                )
                phase_barrier(run_root, "warmup_done", arm, gpu)

                # Rotate scan order to reduce claim collisions.  The exclusive
                # claim file makes each prompt exactly-once within this run;
                # the output identity check makes resume across runs safe.
                offset = gpu % len(pending_indices)
                scan_order = pending_indices[offset:] + pending_indices[:offset]
                while True:
                    case = None
                    for case_index in scan_order:
                        candidate = cases_by_index[case_index]
                        if base.complete_record(arm, candidate):
                            continue
                        claim_path = run_root / "claims" / arm / f"{case_index:02d}.json"
                        if create_claim(
                            claim_path,
                            {
                                "status": "claimed",
                                "arm": arm,
                                "case": case_index,
                                "physical_gpu": gpu,
                                "pid": os.getpid(),
                            },
                        ):
                            case = candidate
                            break
                    if case is None:
                        break
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    result, seconds = base.run_generation(
                        pipe, plugin, state_by_index[case["index"]], base.STEPS
                    )
                    summary = plugin.summary()
                    if arm != "dense" and summary.get("completed_evaluations") != base.STEPS - 1:
                        raise RuntimeError(f"incomplete attention evaluations: {summary}")
                    payload = {
                        "metadata": {
                            "arm": arm,
                            "case": case["index"],
                            "sample_id": case["sample_id"],
                            "prompt_sha256": case["prompt_sha256"],
                            "seed": base.SEED,
                            "steps": base.STEPS,
                            "frames": base.FRAMES,
                            "height": base.HEIGHT,
                            "width": base.WIDTH,
                        },
                        "latents": result["latents"].detach().cpu(),
                        "audio_latents": result["audio_latents"].detach().cpu(),
                    }
                    del result
                    latent_path, record_path = base.paths(arm, case)
                    common.atomic_torch_save(payload, latent_path)
                    record = {
                        **payload["metadata"],
                        "status": "complete",
                        "scheduler": "strict_arm_major_dynamic",
                        "scheduler_run_id": plan["run_id"],
                        "physical_gpu": gpu,
                        "gpu_uuid": protocol["gpu_inventory"][gpu]["uuid"],
                        "denoise_seconds": seconds,
                        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "latent_path": str(latent_path),
                        "latent_sha256": base.sha256(latent_path),
                        "transformer_placement": placement,
                        "torch_compile": True,
                        "compile_wrapped_blocks": len(
                            getattr(acceleration, "_forward_originals", ())
                        ),
                        "arm_spec": dataclasses.asdict(base.ARMS[arm]),
                        "attention_config": configs[arm],
                        "attention_summary": summary,
                    }
                    base.atomic_json(record_path, record)
                    print(
                        f"DONE gpu={gpu} arm={arm} case={case['index']} seconds={seconds:.3f}",
                        flush=True,
                    )
                phase_barrier(run_root, "phase_done", arm, gpu)
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


base.worker = scheduled_worker


def orchestrate(names: tuple[str, ...]) -> None:
    base.prepare(names)
    base.validate_arm_configs(names)
    plan = build_scheduler_plan(names)
    base.atomic_json(
        base.ROOT / "status.json",
        {
            "status": "running",
            "arms": list(names),
            "scheduler": plan["strategy"],
            "estimated_makespan_seconds": plan["estimated_makespan_seconds"],
        },
    )
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
        command = [sys.executable, str(Path(__file__).resolve()), "worker",
                   "--gpu", str(gpu), "--arms", *names]
        process = subprocess.Popen(
            command, cwd=base.REPO,
            env={**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log))
    failures = []
    run_root = base.ROOT / "scheduler_runs" / plan["run_id"]
    remaining = {gpu: (process, log) for gpu, process, log in processes}
    while remaining:
        for gpu, (process, log) in list(remaining.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del remaining[gpu]
            if code:
                failures.append({"gpu": gpu, "exit_code": code})
        if failures and not (run_root / "abort.json").is_file():
            base.atomic_json(
                run_root / "abort.json",
                {"status": "aborted", "worker_failures": failures},
            )
            for process, _ in remaining.values():
                process.terminate()
        if remaining:
            time.sleep(2.0)
    if failures:
        base.atomic_json(base.ROOT / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"worker failures: {failures}")
    base.summarize(names)


base.orchestrate = orchestrate


if __name__ == "__main__":
    base.main()
