"""VBench module for the four-prompt ComfyUI Sol comparison.

This file is copied to the experiment root as ``vbench.py`` before evaluation.
Only each prompt's assigned VBench dimensions are evaluated.
"""
from pathlib import Path
import json
import statistics
import sys


EXP = Path(__file__).resolve().parent
BENCH = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark")
sys.path.insert(0, str(BENCH / "scripts"))
import score_vbench20pct_768p10s as base

ROOT = EXP / "vbench"
DIMS = ("subject_consistency", "motion_smoothness", "imaging_quality", "aesthetic_quality")
read, write = base.read, base.write


def _videos():
    protocol = read(EXP / "protocol.json")
    assigned = {int(c["index"]): set(c["vbench_dimensions"]) for c in protocol["cases"]}
    by_dimension = {dim: [] for dim in DIMS}
    for method in protocol["methods"]:
        manifest = read(Path("/autodl-fs/data/h3_outputs") / protocol["name"] / method / "generation_manifest.json")
        for row in manifest["records"]:
            video = {
                "method": method, "case": int(row["index"]),
                "video_path": row["output_path"], "video_sha256": row["sha256"],
            }
            for dim in assigned[int(row["index"])]:
                if dim in by_dimension:
                    by_dimension[dim].append(video)
    return by_dimension


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    videos_by_dimension = _videos()
    write(ROOT / "protocol.json", {
        "status": "prepared", "dimensions": DIMS, "cache": {},
        "videos_by_dimension": videos_by_dimension,
        "evaluation": "Only the VBench dimensions assigned to each source prompt are evaluated.",
    })
    print("VBENCH PREPARED", {k: len(v) for k, v in videos_by_dimension.items()}, flush=True)


def aggregate():
    protocol = read(ROOT / "protocol.json")
    methods = list(read(EXP / "protocol.json")["methods"])
    dimensions = {}
    scores = {method: {} for method in methods}
    for dim in DIMS:
        result = read(ROOT / "scores" / f"{dim}.json")
        expected = {(v["method"], v["case"]) for v in protocol["videos_by_dimension"][dim]}
        actual = {(v["method"], v["case"]) for v in result["records"]}
        assert result["status"] == "complete" and actual == expected
        dimensions[dim] = result
        for method in methods:
            rows = [r["score"] for r in result["records"] if r["method"] == method]
            scores[method][dim] = 100 * statistics.mean(rows)
    write(ROOT / "results.json", {
        "status": "complete", "scores_percent_assigned_prompts": scores,
        "dimensions": dimensions, "official_total": False,
    })
    print("VBENCH COMPLETE", json.dumps(scores), flush=True)
