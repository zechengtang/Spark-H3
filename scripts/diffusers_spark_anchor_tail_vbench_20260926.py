"""Assigned core-five VBench adapter for Diffusers Spark anchor/tail ablation."""
from pathlib import Path
import json
import statistics
import sys

EXP = Path(__file__).resolve().parent
BENCH = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark")
OUT = Path("/autodl-fs/data/h3_outputs/diffusers_spark_anchor_tail_10prompt_10s768p_aligned_20260926")
sys.path.insert(0, str(BENCH / "scripts"))
import score_vbench20pct_768p10s as base

ROOT = EXP / "vbench"
DIMS = ("subject_consistency", "background_consistency", "motion_smoothness",
        "imaging_quality", "aesthetic_quality")
ARMS = ("spark_threshold", "spark_block", "spark_fp32")
read, write = base.read, base.write


def prepare():
    protocol = read(EXP / "protocol.json")
    assigned = {int(c["index"]): set(c["vbench_dimensions"]) for c in protocol["cases"]}
    videos = {dim: [] for dim in DIMS}
    for arm in ARMS:
        manifest = read(OUT / arm / "generation_manifest.json")
        for row in manifest["records"]:
            video = dict(method=arm, case=int(row["index"]), video_path=row["output_path"],
                         video_sha256=row["sha256"])
            for dim in assigned[int(row["index"])]:
                if dim in videos:
                    videos[dim].append(video)
    ROOT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "protocol.json", dict(status="prepared", dimensions=DIMS,
        videos_by_dimension=videos, cache={}, evaluation="source-assigned core-five VBench"))


def aggregate():
    protocol = read(ROOT / "protocol.json")
    results, scores = {}, {arm: {} for arm in ARMS}
    for dim in DIMS:
        result = read(ROOT / "scores" / f"{dim}.json")
        expected = {(v["method"], v["case"]) for v in protocol["videos_by_dimension"][dim]}
        assert result["status"] == "complete"
        assert {(v["method"], v["case"]) for v in result["records"]} == expected
        results[dim] = result
        for arm in ARMS:
            values = [row["score"] for row in result["records"] if row["method"] == arm]
            scores[arm][dim] = 100 * statistics.fmean(values)
    write(ROOT / "results.json", dict(status="complete", official_total=False,
        scores_percent_assigned_prompts=scores, dimensions=results))
