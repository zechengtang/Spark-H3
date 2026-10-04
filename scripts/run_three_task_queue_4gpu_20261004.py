#!/usr/bin/env python3
"""Persistent, strictly sequential generation + scoring queue for all 3 tasks.

Attach to the already running route-matrix coordinator, then run task1 VBench,
Ref2VA (no VBench), and FP32-anchor generation/RGB quality/VBench. No manual
approval or agent turn is needed between tasks. Resume guards preserve outputs.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_route_mode_quality_25prompt_20261003 as study
import run_route_2x2_full50_3gpu_20261004 as route

REPO = Path(__file__).resolve().parents[1]
ROOT = study.EXPERIMENTS / "three_task_queue_4gpu_20261004"
ROUTE_NAME = "route_2x2_full50_4gpu_20261004"
ANCHOR_NAME = "fp32_anchor_full50_4gpu_20261004"
ANCHOR_METHOD = "legacy_threshold_fp32_anchor"
GPUS = (0, 1, 2, 3)
ENV = {**os.environ, **study.base.ENV, "H3_EXPERIMENT_GPUS": "0,1,2,3",
       "H3_ROUTE_FULL50_NAME": ROUTE_NAME, "H3_WEIGHT_LOAD_LOCK_SLOTS": "4"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def state(stage, **extra):
    payload = dict(status="running", stage=stage, coordinator_pid=os.getpid(), updated_utc=now(), **extra)
    study.base.write(ROOT / "status.json", payload)
    print(json.dumps(payload), flush=True)


def result_path(name, duration):
    return study.EXPERIMENTS / f"{name}_{duration}s768p/results.json"


def rgb_complete(name):
    return all(result_path(name, d).exists() and
               study.base.read(result_path(name, d)).get("full50", {}).get("status") == "complete" for d in (5, 10))


def process_is_route(pid):
    if not pid:
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
        return "run_route_2x2_full50_3gpu_20261004.py run" in cmd
    except (FileNotFoundError, ProcessLookupError):
        return False


def run_command(stage, command):
    for attempt in range(1, 3):
        with (ROOT / f"{stage}.log").open("a") as log:
            proc = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                state(stage, command=command, child_pid=proc.pid, child_process_group=proc.pid, attempt=attempt)
                time.sleep(30)
            code = proc.wait()
        if code == 0:
            state(stage + "_complete", returncode=0)
            return
        state(stage + "_retry", returncode=code, attempt=attempt)
    raise RuntimeError(f"{stage}: two failed attempts, see {ROOT / (stage + '.log')}")


def vbench(name, duration, manifests, stage, cache_experiments=()):
    exp = result_path(name, duration).parent
    if (exp / "vbench/results.json").exists():
        result = study.base.read(exp / "vbench/results.json")
        protocol = study.base.read(exp / "vbench/protocol.json")
        if result.get("status") == "complete" and protocol["manifests"] == manifests:
            for path, digest in protocol["source_sha256"].items():
                assert study.base.sha(path) == digest, "completed VBench source changed"
            state(stage + "_reused_complete")
            return
    cfg = ROOT / f"{stage}_input.json"
    study.base.write(cfg, dict(experiment=str(exp), duration=duration, manifests=manifests,
                              gpus=list(GPUS), cache_experiments=list(map(str, cache_experiments))))
    run_command(stage, [sys.executable, str(REPO / "scripts/run_assigned_full50_vbench_20261004.py"), "--config", str(cfg)])
    result = study.base.read(exp / "vbench/results.json")
    assert result["status"] == "complete"


def main(pid, accept_runtime_repair=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [Path(__file__), REPO / "scripts/run_assigned_full50_vbench_20261004.py",
             REPO / "scripts/run_fp32_anchor_full50_20261004.py", REPO / "scripts/run_ref2va_8fps_phase_bias_20261004.py",
             Path(route.__file__), Path(study.__file__)]
    hashes = {str(p): study.base.sha(p) for p in paths}
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists():
        protocol = study.base.read(protocol_path)
        if protocol["source_sha256"] != hashes:
            assert accept_runtime_repair, "queue source changed"
            changed = {p for p, h in protocol["source_sha256"].items() if hashes.get(p) != h}
            allowed = {str(Path(__file__)), str(REPO / "scripts/run_ref2va_8fps_phase_bias_20261004.py")}
            assert changed <= allowed, changed
            previous_digest = protocol["source_sha256"][str(Path(__file__))]
            study.base.write(ROOT / f"protocol.before_runtime_repair.{previous_digest[:12]}.json", protocol)
            protocol["source_sha256"] = hashes
            previous_repairs = protocol.get("runtime_repairs", [])
            if isinstance(previous_repairs, dict):
                previous_repairs = [previous_repairs]
            previous_repairs.append(dict(utc=now(), changed_paths=sorted(changed),
                reason="Ref2VA audio dependency, conditioning lifetime and multi-GPU loading; no experiment parameter/core changes",
                task1_generation_and_scores_preserved=True))
            protocol["runtime_repairs"] = previous_repairs
            study.base.write(protocol_path, protocol)
    else:
        study.base.write(protocol_path, dict(created_utc=now(), source_sha256=hashes, gpus=list(GPUS),
            ordered_tasks=["route_2x2_full50_plus_RGB_and_assigned_VBench",
                           "ref2va_8fps_Dense_ABC_then_C_Spark_plus_RGB_only",
                           "FP32_anchor_only_full50_plus_RGB_and_assigned_VBench"],
            required_metrics={"task1": ["PSNR", "SSIM", "LPIPS", "assigned-core5-VBench"],
                              "task2": ["PSNR", "SSIM", "LPIPS"],
                              "task3": ["PSNR", "SSIM", "LPIPS", "assigned-core5-VBench"]},
            task1_attached_pid=pid, stop_between_tasks=False,
            resume_policy="Verify input/source hashes; preserve old outputs; at most two automatic transient-failure attempts"))
    while process_is_route(pid):
        state("task1_existing_generation_and_RGB", attached_pid=pid, task1_rgb_complete=rgb_complete(ROUTE_NAME))
        time.sleep(30)
    if not rgb_complete(ROUTE_NAME):
        run_command("task1_resume_generation_and_RGB", [sys.executable, str(Path(route.__file__)), "run"])
    assert rgb_complete(ROUTE_NAME)
    for duration in (5, 10):
        matrix = study.base.read(result_path(ROUTE_NAME, duration))["full50"]
        manifests = {"dense": str(route.reference_manifest(duration)), **matrix["combined_manifests"]}
        vbench(ROUTE_NAME, duration, manifests, f"task1_vbench_{duration}s")
    state("task1_complete_all_required_scores")
    refroot = study.EXPERIMENTS / "ref2va_8fps_phase_bias_20261004"
    if not (refroot / "results.json").exists() or study.base.read(refroot / "results.json")["status"] != "complete":
        run_command("task2_ref2va", [sys.executable, str(REPO / "scripts/run_ref2va_8fps_phase_bias_20261004.py"), "run"])
    assert study.base.read(refroot / "results.json")["status"] == "complete"
    state("task2_complete_RGB_only")
    if not rgb_complete(ANCHOR_NAME):
        run_command("task3_fp32_anchor_generation_and_RGB", [sys.executable, str(REPO / "scripts/run_fp32_anchor_full50_20261004.py"), "run"])
    assert rgb_complete(ANCHOR_NAME)
    for duration in (5, 10):
        matrix = study.base.read(result_path(ROUTE_NAME, duration))["full50"]
        anchor = study.base.read(result_path(ANCHOR_NAME, duration))["full50"]
        manifests = {"dense": str(route.reference_manifest(duration)),
                     "legacy_threshold": matrix["combined_manifests"]["legacy_threshold"],
                     ANCHOR_METHOD: anchor["candidate_manifest"]}
        vbench(ANCHOR_NAME, duration, manifests, f"task3_vbench_{duration}s",
               cache_experiments=[result_path(ROUTE_NAME, duration).parent])
    result = dict(status="complete", completed_utc=now(),
        task1={str(d): str(result_path(ROUTE_NAME, d)) for d in (5, 10)},
        task2=str(refroot / "results.json"), task3={str(d): str(result_path(ANCHOR_NAME, d)) for d in (5, 10)},
        source_sha256=hashes, notes="All generation, RGB fidelity and requested assigned-dimension VBench are complete")
    study.base.write(ROOT / "results.json", result)
    study.base.write(ROOT / "status.json", dict(status="complete", stage="all_three_tasks_complete", completed_utc=now()))
    print("ALL THREE TASKS AND SCORES COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-pid", type=int, default=0)
    parser.add_argument("--accept-runtime-repair", action="store_true")
    args = parser.parse_args()
    try:
        main(args.task1_pid, args.accept_runtime_repair)
    except BaseException as error:
        if ROOT.exists():
            study.base.write(ROOT / "status.json", dict(status="failed", error=str(error), updated_utc=now(),
                traceback=traceback.format_exc()))
        raise
