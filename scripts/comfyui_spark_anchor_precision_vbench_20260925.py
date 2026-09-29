"""Assigned core-5 VBench adapter for the ComfyUI anchor-precision experiment."""
from pathlib import Path
import json
import statistics
import sys


EXP = Path(__file__).resolve().parent
BENCH = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark")
sys.path.insert(0, str(BENCH / "scripts"))
import score_vbench20pct_768p10s as base

ROOT = EXP / "vbench"
DIMS = (
    "subject_consistency", "background_consistency", "motion_smoothness",
    "imaging_quality", "aesthetic_quality",
)
ARMS = (
    "5s_sol_tau1_extra0", "5s_spark_fp32", "5s_spark_bf16",
    "10s_sol_tau1_extra0", "10s_spark_fp32", "10s_spark_bf16",
)
read, write = base.read, base.write


def _videos():
    protocol = read(EXP / "protocol.json")
    assigned = {int(c["index"]): set(c["vbench_dimensions"]) for c in protocol["cases"]}
    by_dimension = {dim: [] for dim in DIMS}
    for arm in ARMS:
        seconds, method = arm.split("s_", 1)
        manifest = read(EXP / "manifests" / f"{seconds}s" / f"{method}.json")
        for row in manifest["records"]:
            video = {
                "method": arm, "case": int(row["index"]),
                "video_path": row["output_path"], "video_sha256": row["sha256"],
            }
            for dim in assigned[int(row["index"])]:
                if dim in by_dimension:
                    by_dimension[dim].append(video)
    return by_dimension


def _existing_cache():
    old = Path("/autodl-fs/data/h3_experiments/comfyui_spark_sol_10prompt_5s768p_20260924/vbench/results.json")
    if not old.exists():
        return {}
    result = read(old)
    cache = {}
    for dim, payload in result["dimensions"].items():
        for row in payload["records"]:
            if row["method"] != "sol_tau1_extra0":
                continue
            cache[f"{dim}:{row['video_sha256']}"] = {
                "score": row["score"], "native_score": row["native_score"],
                "source": str(old),
            }
    return cache


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    videos = _videos()
    write(ROOT / "protocol.json", {
        "status": "prepared", "dimensions": DIMS, "cache": _existing_cache(),
        "videos_by_dimension": videos,
        "evaluation": "Source-assigned VBench core-5; 5s Sol reuses hash-verified prior scores.",
    })
    print("VBENCH PREPARED", {k: len(v) for k, v in videos.items()}, flush=True)


def aggregate():
    protocol = read(ROOT / "protocol.json")
    dimensions, scores = {}, {arm: {} for arm in ARMS}
    for dim in DIMS:
        result = read(ROOT / "scores" / f"{dim}.json")
        expected = {(v["method"], v["case"]) for v in protocol["videos_by_dimension"][dim]}
        actual = {(v["method"], v["case"]) for v in result["records"]}
        assert result["status"] == "complete" and actual == expected
        dimensions[dim] = result
        for arm in ARMS:
            rows = [r["score"] for r in result["records"] if r["method"] == arm]
            scores[arm][dim] = 100 * statistics.fmean(rows)
    nested = {"5s": {}, "10s": {}}
    for arm, value in scores.items():
        seconds, method = arm.split("s_", 1)
        nested[f"{seconds}s"][method] = value
    write(ROOT / "results.json", {
        "status": "complete", "scores_percent_assigned_prompts": scores,
        "scores_by_duration": nested, "dimensions": dimensions,
        "official_total": False,
    })
    print("VBENCH COMPLETE", json.dumps(nested), flush=True)
