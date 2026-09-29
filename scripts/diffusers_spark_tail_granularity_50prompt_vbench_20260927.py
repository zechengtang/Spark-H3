"""Assigned core-five VBench adapter for BF16 Spark query/block 50 prompts."""
from pathlib import Path
import statistics
import sys

EXP = Path(__file__).resolve().parent
BENCH = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark")
OUT = Path("/autodl-fs/data/h3_outputs/diffusers_spark_tail_granularity_50prompt_10s768p_20260927")
sys.path.insert(0, str(BENCH / "scripts"))
import score_vbench20pct_768p10s as base

ROOT = EXP / "vbench"
DIMS = ("subject_consistency", "background_consistency", "motion_smoothness",
        "imaging_quality", "aesthetic_quality")
ARMS = ("query", "block")
read, write = base.read, base.write


def prepare():
    protocol = read(EXP / "protocol.json")
    cases = {int(case["index"]): case for case in protocol["cases"]}
    videos = {dim: [] for dim in DIMS}
    for arm in ARMS:
        manifest = read(OUT / arm / "generation_manifest.json")
        if manifest["sample_count"] != len(cases):
            raise ValueError(f"incomplete {arm} video manifest")
        for row in manifest["records"]:
            case = cases[int(row["index"])]
            if row["sample_id"] != case["sample_id"] or row["prompt_sha256"] != case["prompt_sha256"]:
                raise ValueError(f"video/prompt mismatch: {arm}/{row['index']}")
            video = dict(method=arm, case=int(row["index"]), video_path=row["output_path"],
                         video_sha256=row["sha256"], sample_id=case["sample_id"])
            for dim in case["vbench_dimensions"]:
                if dim in videos:
                    videos[dim].append(video)
    ROOT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "protocol.json", dict(status="prepared", dimensions=DIMS,
        videos_by_dimension=videos, cache={},
        evaluation="source-assigned core-five VBench; 50 prompts per arm"))


def aggregate():
    protocol = read(ROOT / "protocol.json")
    results = {}
    scores = {arm: {} for arm in ARMS}
    counts = {arm: {} for arm in ARMS}
    paired = {}
    for dim in DIMS:
        result = read(ROOT / "scores" / f"{dim}.json")
        expected = {(v["method"], v["case"], v["video_sha256"])
                    for v in protocol["videos_by_dimension"][dim]}
        actual = {(v["method"], v["case"], v["video_sha256"])
                  for v in result["records"]}
        if result["status"] != "complete" or actual != expected:
            raise ValueError(f"incomplete/mismatched {dim} scores")
        results[dim] = result
        for arm in ARMS:
            values = [row["score"] for row in result["records"] if row["method"] == arm]
            scores[arm][dim] = 100 * statistics.fmean(values)
            counts[arm][dim] = len(values)
        lookup = {(row["method"], row["case"]): 100 * row["score"]
                  for row in result["records"]}
        case_ids = sorted({row["case"] for row in result["records"]})
        diffs = [lookup[("block", case)] - lookup[("query", case)] for case in case_ids]
        paired[dim] = dict(count=len(diffs), cases=case_ids,
            mean_block_minus_query_percentage_points=statistics.fmean(diffs),
            block_wins=sum(value > 0 for value in diffs),
            query_wins=sum(value < 0 for value in diffs),
            ties=sum(value == 0 for value in diffs))
    write(ROOT / "results.json", dict(status="complete", official_total=False,
        scores_percent_assigned_prompts=scores, counts=counts,
        paired_assigned_prompt_scores=paired, dimensions=results))
