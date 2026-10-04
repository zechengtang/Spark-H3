"""VBench core-five module that evaluates only each prompt's assigned dimensions.

This module is copied into an experiment directory as ``vbench.py`` and used
by MiniMax-H3-Experiments' persistent GPU queue.  Unlike the generic
all-five-per-video diagnostic template, the queue contains only the official
``samples.json`` memberships.
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
read, write = base.read, base.write
DIMS = (*base.DIMS, "aesthetic_quality")


def config():
    return read(ROOT / "module_config.json")


def membership(cfg):
    raw = read(cfg["samples"])
    entries = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw
    return {int(row["index"]): set(row["vbench_dimensions"]) for row in entries}


def videos(cfg):
    rows = []
    for source in cfg["video_sources"]:
        if source["from"] != "manifest":
            raise ValueError("official-subset module accepts manifest video sources only")
        manifest = read(source["path"])
        if manifest.get("status") != "passed":
            raise ValueError(f"incomplete manifest: {source['path']}")
        by_case = {int(row["index"]): row for row in manifest["records"]}
        for case in cfg["cases"]:
            row = by_case[case]
            rows.append({
                "method": source["method"], "case": case,
                "video_path": row["output_path"], "video_sha256": row["sha256"],
            })
    return rows


def cache(cfg):
    result = {}
    for directory in [ROOT / "scores", *map(Path, cfg.get("historical_scores", []))]:
        for dim in DIMS:
            path = directory / f"{dim}.json"
            if not path.exists():
                continue
            for row in read(path)["records"]:
                result.setdefault(
                    dim + ":" + row["video_sha256"],
                    {"score": row["score"], "native_score": row["native_score"],
                     "source": str(path)},
                )
    return result


def prepare():
    cfg = config()
    ROOT.mkdir(exist_ok=True)
    all_videos = videos(cfg)
    memberships = membership(cfg)
    videos_by_dimension = {
        dim: [row for row in all_videos if dim in memberships[row["case"]]]
        for dim in DIMS
    }
    weights = read(cfg["model_weights"])
    write(ROOT / "model_weights.json", {
        path: {"sha256": base.sha(path), "bytes": Path(path).stat().st_size}
        for path in weights
    })
    sources = [Path(__file__), Path(base.__file__), CODE / "score_vbench20pct_aesthetic.py",
               EXP / "h3_vbench_queue.py", base.VBENCH / "vbench/utils.py"]
    sources += [base.VBENCH / "vbench" / f"{dim}.py" for dim in DIMS]
    protocol = {
        "status": "prepared", "dimensions": DIMS, "gpus": list(range(8)),
        "cache": cache(cfg), "videos": all_videos,
        "videos_by_dimension": videos_by_dimension,
        "source_sha256": {str(path): base.sha(path) for path in sources},
        "model_weights": str(ROOT / "model_weights.json"),
        "evaluation": "Only each prompt's samples.json vbench_dimensions are evaluated.",
        "dimension_job_counts": {dim: len(rows) for dim, rows in videos_by_dimension.items()},
    }
    write(ROOT / "protocol.json", protocol)
    print("VBENCH OFFICIAL PREPARED", sum(map(len, videos_by_dimension.values())),
          "dimension jobs", protocol["dimension_job_counts"], flush=True)


def aggregate():
    cfg = config()
    protocol = read(ROOT / "protocol.json")
    payload = {dim: read(ROOT / "scores" / f"{dim}.json") for dim in DIMS}
    scores = {}
    methods = [source["method"] for source in cfg["video_sources"]]
    for dim, result in payload.items():
        if result.get("status") != "complete":
            raise RuntimeError(f"incomplete dimension: {dim}")
        expected = {(row["method"], row["case"])
                    for row in protocol["videos_by_dimension"][dim]}
        actual = {(row["method"], row["case"]) for row in result["records"]}
        if actual != expected:
            raise RuntimeError(f"dimension membership mismatch: {dim}")
    for method in methods:
        scores[method] = {
            dim: 100 * statistics.fmean(
                row["score"] for row in payload[dim]["records"] if row["method"] == method
            )
            for dim in DIMS
        }
    result = {
        "status": "complete", "official_total": False,
        "evaluation": "samples.json per-prompt assigned dimensions only",
        "scores_percent_official_subsets": scores,
        "dimension_prompt_counts": {
            dim: len(protocol["videos_by_dimension"][dim]) // len(methods) for dim in DIMS
        },
        "dimensions": payload,
    }
    write(ROOT / "results.json", result)
    labels = cfg.get("labels", {})
    lines = [
        "# VBench Core Five — official prompt memberships", "",
        "Only each prompt's declared `vbench_dimensions` were evaluated.", "",
        "| Arm | Subject | Background | Motion | Imaging | Aesthetic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append("| " + labels.get(method, method) + " | "
                     + " | ".join(f"{scores[method][dim]:.4f}" for dim in DIMS) + " |")
    (EXP / "VBENCH_REPORT.md").write_text("\n".join(lines) + "\n")
    print("VBENCH OFFICIAL COMPLETE", json.dumps(scores), flush=True)
