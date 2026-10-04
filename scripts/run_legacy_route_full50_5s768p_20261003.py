#!/usr/bin/env python3
"""Complete cases 26--50 for the legacy route comparison at 5s/768p.

Cases 1--25 are reused from the two completed 2026-10-03 experiments.  This
runner generates, decodes, and scores only the missing half, then writes
combined 50-prompt manifests and paired RGB summaries without copying videos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys

import run_route_mode_quality_25prompt_20261003 as study


NAME = "legacy_route_full50_5s768p_20261003"
CANDIDATE = "legacy_packed_external_no_route_qk"
METHODS = ("dense", "legacy_threshold", CANDIDATE)
FIRST_EXP = Path("/autodl-fs/data/h3_experiments/route_2x2_full50_4gpu_20261004_5s768p/provenance/route_mode_quality_25prompt_20261003_5s768p")
FIRST_OUT = Path("/autodl-fs/data/h3_outputs/route_2x2_full50_4gpu_20261004_5s768p/provenance_outputs/route_mode_quality_25prompt_20261003_5s768p")
CANDIDATE_EXP = Path("/autodl-fs/data/h3_experiments/route_2x2_full50_4gpu_20261004_5s768p/provenance/legacy_packed_external_no_route_qk_25prompt_20261003_5s768p")
CANDIDATE_OUT = Path("/autodl-fs/data/h3_outputs/route_2x2_full50_4gpu_20261004_5s768p/provenance_outputs/legacy_packed_external_no_route_qk_25prompt_20261003_5s768p")


def configure() -> None:
    study.NAME = NAME
    study.CASES = tuple(range(26, 51))
    study.ALL_METHODS = METHODS
    study.GENERATED = {5: METHODS, 10: ()}
    study.GPUS = tuple(int(value) for value in os.environ.get(
        "H3_EXPERIMENT_GPUS", "0,1,2,3,4,5,6,7"
    ).split(","))
    study.__file__ = str(Path(__file__).resolve())

    def config(method: str, steps: int = 20):
        study.base.pipeline()
        from h3_sparse_attention import H3SparseAttentionConfig

        if method == "dense":
            return None
        kwargs = dict(
            warmup_percent=20.0 if steps == 20 else 33.0,
            sol_dense_layers=1,
            sol_route_topk_ratio=0.1,
            sol_log_density=False,
            landmark_tree_v2_midpoint_direction_mode="legacy",
        )
        kwargs["sol_route_topk_execution"] = {
            "legacy_threshold": "threshold",
            CANDIDATE: "packed_external_no_route_qk",
        }[method]
        return H3SparseAttentionConfig.spark(steps, **kwargs)

    study.config = config
    original_prepare = study.prepare

    def prepare(duration: int) -> None:
        if duration != 5:
            raise ValueError("this continuation is 5s-only")
        for required in (
            FIRST_EXP / "results.json",
            FIRST_OUT / "dense" / "generation_manifest.json",
            FIRST_OUT / "legacy_threshold" / "generation_manifest.json",
            CANDIDATE_EXP / "results.json",
            CANDIDATE_OUT / CANDIDATE / "generation_manifest.json",
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        root, out = study.roots(duration)
        if root.exists() or out.exists():
            if not (root.exists() and out.exists() and (root / "protocol.json").exists()):
                raise FileExistsError(f"incomplete resume roots: {root} / {out}")
            protocol = study.base.read(root / "protocol.json")
            if protocol.get("case_indices") != list(study.CASES):
                raise ValueError("resume protocol case set does not match")
            print("RESUMING", root, flush=True)
            return
        original_prepare(duration)
        protocol = study.base.read(root / "protocol.json")
        protocol.update(
            purpose="Complete the 20pct 50-prompt 5s/768p legacy route comparison",
            generated_case_indices=list(study.CASES),
            reused_case_indices=list(range(1, 26)),
            reused_sources={
                "dense_and_legacy_threshold": str(FIRST_EXP),
                CANDIDATE: str(CANDIDATE_EXP),
            },
            weight_load_slots=int(os.environ.get("H3_WEIGHT_LOAD_LOCK_SLOTS", "4")),
            vbench_scope="Only each prompt's declared vbench_dimensions on the full 50-prompt manifest",
        )
        study.base.write(root / "protocol.json", protocol)

    study.prepare = prepare


def source_manifest(method: str) -> Path:
    if method == CANDIDATE:
        return CANDIDATE_OUT / method / "generation_manifest.json"
    return FIRST_OUT / method / "generation_manifest.json"


def source_quality(method: str, case: int) -> Path:
    root = CANDIDATE_EXP if method == CANDIDATE else FIRST_EXP
    return root / "quality" / method / f"{method}_{case:02}.json"


def source_runtime(method: str, case: int) -> Path:
    root = CANDIDATE_EXP if method == CANDIDATE else FIRST_EXP
    return root / "records" / method / f"case_{case:02}.json"


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def finalize() -> None:
    root, out = study.roots(5)
    combined_root = out / "full50"
    combined_root.mkdir(parents=True, exist_ok=True)
    combined_manifests = {}
    for method in METHODS:
        first = study.base.read(source_manifest(method))
        second = study.base.read(out / method / "generation_manifest.json")
        records = [*first["records"], *second["records"]]
        indices = [int(row["index"]) for row in records]
        if indices != list(range(1, 51)):
            raise RuntimeError(f"{method}: combined indices are not exactly 1--50: {indices}")
        target = combined_root / method / "generation_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        study.base.write(target, {
            "schema_version": 1,
            "status": "passed",
            "method": method,
            "sample_count": 50,
            "settings": second["settings"],
            "records": records,
            "reused_cases": list(range(1, 26)),
            "generated_cases": list(range(26, 51)),
        })
        combined_manifests[method] = str(target)

    quality_rows = {}
    quality_summary = {}
    for method in METHODS[1:]:
        rows = [study.base.read(source_quality(method, case)) for case in range(1, 26)]
        rows += [study.base.read(root / "quality" / method / f"{method}_{case:02}.json") for case in range(26, 51)]
        quality_rows[method] = rows
        quality_summary[method] = {
            metric: statistics.fmean(row[metric] for row in rows)
            for metric in ("psnr_db", "ssim", "lpips")
        }

    paired = {}
    legacy_by_case = {int(row["case"]): row for row in quality_rows["legacy_threshold"]}
    candidate_by_case = {int(row["case"]): row for row in quality_rows[CANDIDATE]}
    for metric in ("psnr_db", "ssim", "lpips"):
        delta = [candidate_by_case[case][metric] - legacy_by_case[case][metric] for case in range(1, 51)]
        paired[metric] = {
            "mean_delta": statistics.fmean(delta),
            "stdev_delta": statistics.stdev(delta),
            "improved": sum(value > 0 for value in delta),
            "degraded": sum(value < 0 for value in delta),
            "values": delta,
        }

    runtime = {}
    for method in METHODS:
        rows = [study.base.read(source_runtime(method, case)) for case in range(1, 26)]
        rows += [study.base.read(root / "records" / method / f"case_{case:02}.json") for case in range(26, 51)]
        runtime[method] = summarize([float(row["denoise_seconds"]) for row in rows])

    result = study.base.read(root / "results.json")
    result.update(
        status="complete",
        full50={
            "status": "quality_complete_vbench_pending",
            "case_indices": list(range(1, 51)),
            "combined_manifests": combined_manifests,
            "runtime": runtime,
            "quality_vs_dense": quality_summary,
            "candidate_vs_legacy_threshold": paired,
        },
    )
    study.base.write(root / "results.json", result)


def main() -> None:
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        study.run(5)
        finalize()
    elif command in {"generate_worker", "decode_worker"}:
        getattr(study, command)(5, int(sys.argv[2] if len(sys.argv) == 3 else sys.argv[3]))
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
