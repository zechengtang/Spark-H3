#!/usr/bin/env python3
"""Orthogonal legacy + packed-external-no-route-QK study at 5s/10s 768p.

This continuation deliberately keeps the landmark-tree midpoint direction
builder on ``legacy`` while changing only the Top-K execution path. Run 5s to
completion before 10s; the command refuses to overwrite either result root.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import run_route_mode_quality_25prompt_20261003 as study


METHOD = "legacy_packed_external_no_route_qk"
NAME = "legacy_packed_external_no_route_qk_25prompt_20261003"
PREVIOUS_NAME = "route_mode_quality_25prompt_20261003"
OLD_10S = Path("/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920")


def configure() -> None:
    study.NAME = NAME
    study.ALL_METHODS = (METHOD,)
    study.GENERATED = {5: (METHOD,), 10: (METHOD,)}
    study.GPUS = tuple(int(value) for value in os.environ.get("H3_EXPERIMENT_GPUS", "0,1,2,3,4,5,6,7").split(","))
    if not study.GPUS:
        raise ValueError("H3_EXPERIMENT_GPUS must contain at least one GPU")
    # spawn() resolves __file__ in the imported module; point workers back to
    # this continuation so they receive the same orthogonal configuration.
    study.__file__ = str(Path(__file__).resolve())

    original_prepare = study.prepare

    def prepare(duration: int) -> None:
        original_prepare(duration)
        root, _ = study.roots(duration)
        protocol = study.base.read(root / "protocol.json")
        protocol.update(
            purpose="Hold legacy midpoint directions fixed while evaluating packed_external_no_route_qk",
            baseline_method="legacy_threshold",
            baseline_source=str(
                study.EXPERIMENTS / f"{PREVIOUS_NAME}_5s768p"
                if duration == 5 else OLD_10S
            ),
            reused_10s_methods=["dense", "legacy_threshold"] if duration == 10 else [],
            ordering_constraint="5s must complete before 10s starts",
            weight_load_slots=int(os.environ.get("H3_WEIGHT_LOAD_LOCK_SLOTS", "4")),
        )
        study.base.write(root / "protocol.json", protocol)

    study.prepare = prepare

    def quality(duration: int) -> None:
        root, out = study.roots(duration)
        reference_manifest = (
            study.OUTPUTS / f"{PREVIOUS_NAME}_5s768p" / "dense" / "generation_manifest.json"
            if duration == 5 else study.SOURCE / "dense" / "generation_manifest.json"
        )
        work = root / "quality" / METHOD
        quality_config = {
            "work_dir": str(work),
            "reference_manifest": str(reference_manifest),
            "candidate_manifest": str(out / METHOD / "generation_manifest.json"),
            "method": METHOD,
            "cases": list(study.CASES),
            "workers": len(study.GPUS),
        }
        config_path = root / "quality" / f"{METHOD}_config.json"
        study.base.write(config_path, quality_config)
        subprocess.run(
            [str(study.base.PYTHON), str(study.QUALITY_SCRIPT), "run", "--config", str(config_path)],
            check=True,
            env={
                **os.environ,
                **study.base.ENV,
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, study.GPUS)),
                "H3_NUM_GPUS": str(len(study.GPUS)),
            },
        )
        summary = study.base.read(work / f"{METHOD}_quality_results.json")["summary"]
        result = study.base.read(root / "results.json")
        result.update(status="complete", quality={METHOD: summary})
        study.base.write(root / "results.json", result)
        protocol = study.base.read(root / "protocol.json")
        protocol["status"] = "complete"
        study.base.write(root / "protocol.json", protocol)
        study.base.write(root / "status.json", {"status": "complete", "stage": "complete"})

    study.quality = quality

    def config(method: str, steps: int = 20):
        study.base.pipeline()
        from h3_sparse_attention import H3SparseAttentionConfig

        if method != METHOD:
            raise ValueError(f"unsupported method: {method}")
        return H3SparseAttentionConfig.spark(
            steps,
            warmup_percent=20.0 if steps == 20 else 33.0,
            sol_dense_layers=1,
            sol_route_topk_ratio=0.1,
            sol_log_density=False,
            landmark_tree_v2_midpoint_direction_mode="legacy",
            sol_route_topk_execution="packed_external_no_route_qk",
        )

    study.config = config


def previous_legacy_rows(duration: int) -> list[dict]:
    if duration == 5:
        root = study.EXPERIMENTS / f"{PREVIOUS_NAME}_5s768p" / "quality" / "legacy_threshold"
        return [study.base.read(root / f"legacy_threshold_{case:02}.json") for case in study.CASES]
    root = OLD_10S / "quality_work" / "quality"
    return [study.base.read(root / f"topk10_reblock_global_reweight_{case:02}.json") for case in study.CASES]


def candidate_rows(duration: int) -> list[dict]:
    root, _ = study.roots(duration)
    quality = root / "quality" / METHOD
    return [study.base.read(quality / f"{METHOD}_{case:02}.json") for case in study.CASES]


def paired_summary(candidate: list[dict], legacy: list[dict]) -> dict:
    result = {}
    for metric in ("psnr_db", "ssim", "lpips"):
        delta = [new[metric] - old[metric] for new, old in zip(candidate, legacy, strict=True)]
        mean = statistics.mean(delta)
        stdev = statistics.stdev(delta)
        sem = stdev / math.sqrt(len(delta))
        # t(24, 0.975); record the interval without requiring scipy at runtime.
        result[metric] = {
            "mean_delta": mean,
            "stdev_delta": stdev,
            "ci95": [mean - 2.0639 * sem, mean + 2.0639 * sem],
            "improved": sum(value > 0 for value in delta),
            "degraded": sum(value < 0 for value in delta),
            "values": delta,
        }
    return result


def finalize(duration: int) -> None:
    root, _ = study.roots(duration)
    result = study.base.read(root / "results.json")
    result["comparison"] = {
        "candidate": METHOD,
        "baseline": "legacy_threshold",
        "baseline_source": str(
            study.EXPERIMENTS / f"{PREVIOUS_NAME}_5s768p"
            if duration == 5 else OLD_10S
        ),
        "paired": paired_summary(candidate_rows(duration), previous_legacy_rows(duration)),
    }
    study.base.write(root / "results.json", result)


def main() -> None:
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        duration = int(sys.argv[2])
        if duration == 10:
            five_results = study.EXPERIMENTS / f"{NAME}_5s768p" / "results.json"
            if not five_results.exists() or study.base.read(five_results).get("status") != "complete":
                raise RuntimeError("5s experiment must complete before 10s starts")
        study.run(duration)
        finalize(duration)
    elif command in {"generate_worker", "decode_worker"}:
        getattr(study, command)(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
