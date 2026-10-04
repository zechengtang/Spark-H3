#!/usr/bin/env python3
"""Assigned-dimension VBench evaluation for the full 5s/768p 50-prompt run."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import statistics
import sys


EXPERIMENTS_CODE = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Experiments/scripts")
sys.path.insert(0, str(EXPERIMENTS_CODE))
import score_vbench20pct_768p10s as base


EXP = Path("/autodl-fs/data/h3_experiments/legacy_route_full50_5s768p_20261003_5s768p")
OUT = Path("/autodl-fs/data/h3_outputs/legacy_route_full50_5s768p_20261003_5s768p/full50")
SAMPLES = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark/vbench_core5_percent_subsets/20pct/samples.json")
ROOT = EXP / "vbench"
METHODS = ("dense", "legacy_threshold", "legacy_packed_external_no_route_qk")
DIMS = (*base.DIMS, "aesthetic_quality")
EXPECTED = {
    "subject_consistency": 14,
    "background_consistency": 17,
    "motion_smoothness": 14,
    "imaging_quality": 19,
    "aesthetic_quality": 19,
}
read, write = base.read, base.write


def prepare() -> None:
    main_result = read(EXP / "results.json")
    if main_result.get("full50", {}).get("status") != "quality_complete_vbench_pending":
        raise RuntimeError("full 50-prompt RGB quality must complete before VBench")
    cases = read(SAMPLES)
    if len(cases) != 50 or [int(row["index"]) for row in cases] != list(range(1, 51)):
        raise ValueError("unexpected 20pct sample manifest")
    case_by_index = {int(row["index"]): row for row in cases}
    counts = {
        dim: sum(dim in row["vbench_dimensions"] for row in cases)
        for dim in DIMS
    }
    if counts != EXPECTED:
        raise ValueError(f"unexpected assigned-dimension counts: {counts}")

    videos_by_dimension = {dim: [] for dim in DIMS}
    manifests = {}
    for method in METHODS:
        manifest_path = OUT / method / "generation_manifest.json"
        manifest = read(manifest_path)
        if manifest.get("status") != "passed" or manifest.get("sample_count") != 50:
            raise ValueError(f"incomplete manifest: {manifest_path}")
        manifests[method] = str(manifest_path)
        for row in manifest["records"]:
            case = case_by_index[int(row["index"])]
            prompt_sha256 = hashlib.sha256(case["generation_prompt"].encode()).hexdigest()
            if row["sample_id"] != case["sample_id"] or row["prompt_sha256"] != prompt_sha256:
                raise ValueError(f"video/prompt mismatch: {method}/{row['index']}")
            video = {
                "method": method,
                "case": int(row["index"]),
                "sample_id": row["sample_id"],
                "video_path": row["output_path"],
                "video_sha256": row["sha256"],
            }
            if base.sha(video["video_path"]) != video["video_sha256"]:
                raise ValueError(f"video hash mismatch: {method}/{row['index']}")
            assigned = set(case["vbench_dimensions"])
            if not assigned or not assigned <= set(DIMS):
                raise ValueError(f"invalid dimensions: {case['index']} {sorted(assigned)}")
            for dim in assigned:
                videos_by_dimension[dim].append(video)
    for dim, count in EXPECTED.items():
        if len(videos_by_dimension[dim]) != count * len(METHODS):
            raise RuntimeError(f"{dim}: incomplete assigned jobs")

    ROOT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "protocol.json", {
        "status": "prepared",
        "dimensions": list(DIMS),
        "methods": list(METHODS),
        "prompt_counts_by_dimension": counts,
        "videos_by_dimension": videos_by_dimension,
        "cache": {},
        "manifests": manifests,
        "evaluation": (
            "Full 20pct 50-prompt manifest. Each video is scored only on the "
            "dimensions declared by its source prompt. Scores are normalized means "
            "times 100, not an official overall VBench aggregate."
        ),
    })
    print("VBENCH PREPARED", sum(map(len, videos_by_dimension.values())), "jobs", flush=True)


def aggregate() -> None:
    protocol = read(ROOT / "protocol.json")
    dimensions = {}
    scores = {method: {} for method in METHODS}
    counts = {method: {} for method in METHODS}
    comparisons = (
        ("legacy_packed_external_no_route_qk", "legacy_threshold"),
        ("legacy_threshold", "dense"),
        ("legacy_packed_external_no_route_qk", "dense"),
    )
    paired = {f"{left}_vs_{right}": {} for left, right in comparisons}
    for dim in DIMS:
        result = read(ROOT / "scores" / f"{dim}.json")
        expected = {
            (row["method"], int(row["case"]), row["video_sha256"])
            for row in protocol["videos_by_dimension"][dim]
        }
        actual = {
            (row["method"], int(row["case"]), row["video_sha256"])
            for row in result["records"]
        }
        if result.get("status") != "complete" or actual != expected:
            raise RuntimeError(f"{dim}: incomplete or mismatched score set")
        dimensions[dim] = result
        lookup = {(row["method"], int(row["case"])): 100 * row["score"] for row in result["records"]}
        case_ids = sorted({int(row["case"]) for row in result["records"]})
        for method in METHODS:
            values = [lookup[(method, case)] for case in case_ids]
            scores[method][dim] = statistics.fmean(values)
            counts[method][dim] = len(values)
        for left, right in comparisons:
            delta = [lookup[(left, case)] - lookup[(right, case)] for case in case_ids]
            paired[f"{left}_vs_{right}"][dim] = {
                "count": len(delta),
                "cases": case_ids,
                "mean_delta_percentage_points": statistics.fmean(delta),
                "wins": sum(value > 0 for value in delta),
                "losses": sum(value < 0 for value in delta),
                "ties": sum(value == 0 for value in delta),
                "values": delta,
            }
    payload = {
        "status": "complete",
        "official_total": False,
        "scores_percent_assigned_prompts": scores,
        "counts": counts,
        "paired_assigned_prompt_scores": paired,
        "dimensions": dimensions,
    }
    write(ROOT / "results.json", payload)
    main = read(EXP / "results.json")
    main["full50"]["status"] = "complete"
    main["full50"]["vbench"] = payload
    write(EXP / "results.json", main)
    print("VBENCH COMPLETE", json.dumps(scores, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    globals()[command]()
