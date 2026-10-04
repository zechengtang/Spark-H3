"""VBench core-five module for the fixed temporal-chunk Table 3/4 experiment.

This file is copied to ``<experiment>/vbench.py`` before the queue starts.
"""

from pathlib import Path
import json
import statistics
import sys


EXP = Path(__file__).resolve().parent
CODE = EXP / "vbench_code"
try:
    import _bench_bootstrap  # noqa: F401
except ImportError:
    pass
sys.path.insert(0, str(CODE))
import score_vbench20pct_768p10s as base

ROOT = EXP / "vbench"
DIMS = (*base.DIMS, "aesthetic_quality")
read, write = base.read, base.write
CASES = list(range(1, 51))
ARMS = ("chunk5_topk10_reblock", "chunk10_topk10_reblock")
EXPECTED_CASE_COUNTS = {
    "subject_consistency": 14,
    "background_consistency": 17,
    "motion_smoothness": 14,
    "imaging_quality": 19,
    "aesthetic_quality": 19,
}
HISTORICAL_SCORES = [
    Path("/autodl-fs/data/h3_experiments/strict_splitter_tau1_20260918/vbench/scores"),
    Path("/autodl-fs/data/h3_experiments/vbench20pct_768p10s_seed42_20260913/vbench"),
]
DENSE_MANIFEST = "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json"
SOL_MANIFEST = "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/sol/generation_manifest.json"


def select_videos_by_dimension(videos):
    """Return only the videos whose prompt is assigned to each VBench metric."""
    selected = {dim: [] for dim in DIMS}
    known = set(DIMS)
    for video in videos:
        assigned = set(video.get("vbench_dimensions", ()))
        if not assigned or not assigned <= known:
            raise ValueError(
                f"invalid VBench dimensions for case {video.get('case')}: "
                f"{sorted(assigned)}"
            )
        for dim in assigned:
            selected[dim].append(video)
    return selected


def prepare():
    ROOT.mkdir(exist_ok=True)
    experiment = read(EXP / "protocol.json")
    arms = tuple(experiment.get("arms", ARMS))
    if not arms or len(set(arms)) != len(arms):
        raise ValueError(f"invalid experiment arms: {arms}")
    case_dimensions = {
        row["index"]: tuple(row["vbench_dimensions"])
        for row in experiment["cases"]
    }
    if len(experiment["cases"]) != len(CASES) or set(case_dimensions) != set(CASES):
        raise ValueError("experiment protocol does not contain exactly cases 1--50")
    case_counts = {
        dim: sum(dim in assigned for assigned in case_dimensions.values())
        for dim in DIMS
    }
    if case_counts != EXPECTED_CASE_COUNTS:
        raise ValueError(
            f"unexpected prompt-to-dimension assignment: {case_counts}"
        )
    videos = []
    for arm in arms:
        for case in CASES:
            row = read(EXP / "quality_work/quality" / f"{arm}_{case:02}.json")
            assert row["status"] == "complete" and row["frames"] == 240
            videos.append(
                dict(
                    method=arm,
                    case=case,
                    video_path=row["video_path"],
                    video_sha256=row["video_sha256"],
                    vbench_dimensions=case_dimensions[case],
                )
            )
    for method, manifest in (("dense", DENSE_MANIFEST), ("sol", SOL_MANIFEST)):
        for row in read(manifest)["records"]:
            if row["index"] in CASES:
                videos.append(
                    dict(
                        method=method,
                        case=row["index"],
                        video_path=row["output_path"],
                        video_sha256=row["sha256"],
                        vbench_dimensions=case_dimensions[row["index"]],
                    )
                )
    videos_by_dimension = select_videos_by_dimension(videos)
    expected_method_count = len(arms) + 2
    for dim, expected_cases in EXPECTED_CASE_COUNTS.items():
        actual = videos_by_dimension[dim]
        if len(actual) != expected_method_count * expected_cases:
            raise RuntimeError(
                f"{dim}: expected {expected_method_count * expected_cases} "
                f"method/case videos, found {len(actual)}"
            )
    weights = read(
        "/autodl-fs/data/h3_experiments/unsplit_f16_fp16_10prompt_20260915/vbench/model_weights.json"
    )
    write(
        ROOT / "model_weights.json",
        {p: dict(sha256=base.sha(p), bytes=Path(p).stat().st_size) for p in weights},
    )
    sources = [
        Path(__file__),
        Path(base.__file__),
        CODE / "score_vbench20pct_aesthetic.py",
        EXP / "h3_vbench_queue.py",
        base.VBENCH / "vbench/utils.py",
    ]
    sources += [base.VBENCH / "vbench" / f"{dim}.py" for dim in DIMS]
    cache = {}
    for dim in DIMS:
        current = ROOT / "scores" / f"{dim}.json"
        paths = [current, *(directory / f"{dim}.json" for directory in HISTORICAL_SCORES)]
        for path in paths:
            if not path.exists():
                continue
            for row in read(path)["records"]:
                cache.setdefault(
                    dim + ":" + row["video_sha256"],
                    dict(
                        score=row["score"],
                        native_score=row["native_score"],
                        source=str(path),
                    ),
                )
    write(
        ROOT / "protocol.json",
        dict(
            status="prepared",
            arms=arms,
            dimensions=DIMS,
            gpus=list(range(4)),
            cache=cache,
            videos=videos,
            videos_by_dimension=videos_by_dimension,
            prompt_counts_by_dimension=case_counts,
            source_sha256={str(path): base.sha(path) for path in sources},
            model_weights=str(ROOT / "model_weights.json"),
            evaluation=(
                "Each VBench dimension is evaluated only on prompts assigned to that "
                "dimension by the experiment case metadata: subject/motion 14, "
                "background 17, imaging/aesthetic 19 prompts per method. Both chunk "
                "arms, Dense, and Sol-H3 use identical subsets. Normalized means x100; "
                "not an official overall VBench score. Cache reuse requires exact SHA-256."
            ),
        ),
    )
    print(
        "VBENCH PREPARED",
        len(videos),
        "videos",
        sum(map(len, videos_by_dimension.values())),
        "dimension jobs,",
        len(cache),
        "cache entries",
        flush=True,
    )


def aggregate():
    protocol = read(ROOT / "protocol.json")
    payload = {dim: read(ROOT / "scores" / f"{dim}.json") for dim in DIMS}
    for dim, result in payload.items():
        assert result["status"] == "complete"
        expected = {
            (video["method"], video["case"])
            for video in protocol["videos_by_dimension"][dim]
        }
        actual = {(row["method"], row["case"]) for row in result["records"]}
        if len(actual) != len(result["records"]):
            raise RuntimeError(f"{dim}: duplicate method/case score records found")
        if actual != expected:
            raise RuntimeError(
                f"{dim}: scored method/case set does not match assigned prompts; "
                f"expected {len(expected)}, found {len(actual)}"
            )
    methods = [*protocol.get("arms", ARMS), "dense", "sol"]
    scores = {
        method: {
            dim: 100
            * statistics.mean(
                row["score"]
                for row in payload[dim]["records"]
                if row["method"] == method
            )
            for dim in DIMS
        }
        for method in methods
    }
    result = dict(
        status="complete", scores_percent=scores, dimensions=payload, official_total=False
    )
    write(ROOT / "results.json", result)
    combined = read(EXP / "results.json")
    combined["vbench"] = result
    write(EXP / "results.json", combined)
    lines = [
        "",
        "## VBench Core Five",
        "",
        "Each dimension was evaluated only on its assigned prompts for both chunk arms, Dense, and Sol-H3: Subject/Motion 14, Background 17, and Imaging/Aesthetic 19 prompts. Scores are normalized means x100 (higher is better), not the official full VBench aggregate.",
        "",
        "| Arm | Subject | Background | Motion | Imaging | Aesthetic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(f"{scores[method][dim]:.4f}" for dim in DIMS)
            + " |"
        )
    report = EXP / "REPORT.md"
    previous = report.read_text().split("\n## VBench Core Five")[0].rstrip()
    report.write_text(previous + "\n" + "\n".join(lines) + "\n")
    print("VBENCH COMPLETE", json.dumps(scores), flush=True)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
