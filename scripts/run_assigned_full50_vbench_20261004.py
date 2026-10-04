#!/usr/bin/env python3
"""Full50 source-assigned core-five VBench, using the persistent SQLite queue.

Copied to an experiment as vbench.py for the existing queue loader. Fresh
scores are used unless that exact experiment is resumed; historical scores
without matching metric provenance are deliberately not used as a cache.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

CODE = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Experiments/scripts")
sys.path.insert(0, str(CODE))
import score_vbench20pct_768p10s as base
from scipy import stats

read, write = base.read, base.write
DIMS = (*base.DIMS, "aesthetic_quality")
EXPECTED = dict(subject_consistency=14, background_consistency=17,
                motion_smoothness=14, imaging_quality=19, aesthetic_quality=19)
SAMPLES = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark/vbench_core5_percent_subsets/20pct/samples.json")


def source_hashes():
    paths = [Path(__file__), CODE / "h3_vbench_queue.py", Path(base.__file__),
             CODE / "score_vbench20pct_aesthetic.py", base.VBENCH / "vbench/utils.py"]
    paths += [base.VBENCH / "vbench" / f"{d}.py" for d in DIMS]
    return {str(p): base.sha(p) for p in paths}


def prepare(exp, cfg):
    cases = read(SAMPLES)
    assert [int(r["index"]) for r in cases] == list(range(1, 51))
    counts = {d: sum(d in c["vbench_dimensions"] for c in cases) for d in DIMS}
    assert counts == EXPECTED, counts
    selected = {d: [] for d in DIMS}
    for method, path in cfg["manifests"].items():
        manifest = read(path)
        assert manifest["status"] == "passed" and manifest["sample_count"] == 50
        rows = {int(r["index"]): r for r in manifest["records"]}
        assert sorted(rows) == list(range(1, 51))
        for case in cases:
            row = rows[int(case["index"])]
            assert row["sample_id"] == case["sample_id"]
            assert row["prompt_sha256"] == hashlib.sha256(case["generation_prompt"].encode()).hexdigest()
            assert base.sha(row["output_path"]) == row["sha256"]
            assert row.get("video", {}).get("frames", cfg["duration"] * 24) == cfg["duration"] * 24
            assigned = set(case["vbench_dimensions"])
            assert assigned and assigned <= set(DIMS)
            video = dict(method=method, case=int(case["index"]), sample_id=case["sample_id"],
                         video_path=row["output_path"], video_sha256=row["sha256"])
            for dim in assigned:
                selected[dim].append(video)
    hashes, cache = source_hashes(), {}
    for cached_exp in cfg.get("cache_experiments", []):
        cached_exp = Path(cached_exp)
        old = read(cached_exp / "vbench/protocol.json")
        # Only our byte-identical evaluator with the exact metric source hashes
        # can seed a cache. Historical protocols lacking hashes are not reused.
        old_hashes = {p: h for p, h in old["source_sha256"].items() if Path(p).name != "vbench.py"}
        current_hashes = {p: h for p, h in hashes.items() if p != str(Path(__file__))}
        assert old_hashes == current_hashes, "cached VBench metric provenance changed"
        assert old["source_sha256"][str(cached_exp / "vbench.py")] == base.sha(__file__)
        assert old["duration"] == cfg["duration"]
        for dim in DIMS:
            path = cached_exp / "vbench/scores" / f"{dim}.json"
            scores = read(path)
            assert scores["status"] == "complete"
            expected = {(v["method"], v["case"], v["video_sha256"]) for v in old["videos_by_dimension"][dim]}
            actual = {(v["method"], v["case"], v["video_sha256"]) for v in scores["records"]}
            assert actual == expected
            for row in scores["records"]:
                cache[dim + ":" + row["video_sha256"]] = dict(score=row["score"], native_score=row["native_score"], source=str(path))
    protocol = dict(status="prepared", dimensions=list(DIMS), methods=list(cfg["manifests"]),
        prompt_counts_by_dimension=counts, videos_by_dimension=selected, cache=cache,
        manifests=cfg["manifests"], source_sha256=hashes, duration=cfg["duration"],
        evaluation="Only source-assigned core-five dimensions; scores x100; NOT an official overall VBench score")
    path = exp / "vbench/protocol.json"
    if path.exists():
        previous = read(path)
        for key in ("videos_by_dimension", "source_sha256", "manifests", "duration", "cache"):
            if previous[key] != protocol[key]:
                raise ValueError(f"VBench resume changed {key}")
    else:
        write(path, protocol)


def aggregate():
    # Called by h3_vbench_queue after loading this copied file as vbench.py.
    exp = Path(__file__).resolve().parent
    protocol = read(exp / "vbench/protocol.json")
    methods = protocol["methods"]
    scores, counts = {m: {} for m in methods}, {m: {} for m in methods}
    comparisons = {f"{a}_minus_{b}": {} for a, b in itertools.combinations(methods, 2)}
    for dim in DIMS:
        result = read(exp / "vbench/scores" / f"{dim}.json")
        expected = {(r["method"], int(r["case"]), r["video_sha256"])
                    for r in protocol["videos_by_dimension"][dim]}
        actual = {(r["method"], int(r["case"]), r["video_sha256"]) for r in result["records"]}
        assert result["status"] == "complete" and actual == expected
        assert len(result["records"]) == len(expected)
        lookup = {(r["method"], int(r["case"])): 100 * r["score"] for r in result["records"]}
        ids = sorted({int(r["case"]) for r in result["records"]})
        for method in methods:
            scores[method][dim] = statistics.fmean(lookup[method, i] for i in ids)
            counts[method][dim] = len(ids)
        for a, b in itertools.combinations(methods, 2):
            values = [lookup[a, i] - lookup[b, i] for i in ids]
            mean = statistics.fmean(values)
            sem = stats.sem(values)
            ci = stats.t.interval(.95, len(values) - 1, loc=mean, scale=sem) if sem else (mean, mean)
            pvalue = float(stats.ttest_1samp(values, 0).pvalue) if sem else (1.0 if mean == 0 else 0.0)
            comparisons[f"{a}_minus_{b}"][dim] = dict(count=len(ids), cases=ids,
                mean_delta_percentage_points=mean, ci95=list(map(float, ci)), pvalue=pvalue,
                wins=sum(x > 0 for x in values), losses=sum(x < 0 for x in values), values=values)
    payload = dict(status="complete", official_total=False, scores_percent_assigned_prompts=scores,
                   counts=counts, paired_assigned_prompt_scores=comparisons,
                   statistical_caveat="Exploratory unadjusted paired tests; nonsignificance is not equivalence",
                   source_sha256=protocol["source_sha256"])
    write(exp / "vbench/results.json", payload)
    main_path = exp / "results.json"
    if main_path.exists():
        main = read(main_path)
        main.setdefault("full50", {})["vbench"] = payload
        write(main_path, main)
    print("ASSIGNED FULL50 VBENCH COMPLETE", json.dumps(scores), flush=True)


def run(config_path):
    cfg = read(config_path)
    exp = Path(cfg["experiment"])
    copy = exp / "vbench.py"
    exp.mkdir(parents=True, exist_ok=True)
    if copy.exists() and base.sha(copy) != base.sha(__file__):
        raise ValueError("copied VBench evaluator changed")
    if not copy.exists():
        shutil.copy2(__file__, copy)
    # Prepare through the copied source, so provenance records the same paths
    # when the queue subsequently imports that module.
    import importlib.util
    spec = importlib.util.spec_from_file_location("assigned_vbench", copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(exp, cfg)
    subprocess.run([sys.executable, str(CODE / "h3_vbench_queue.py"), "run", "--experiment", str(exp),
                    "--gpus", *map(str, cfg["gpus"])], check=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"})
    assert read(exp / "vbench/results.json")["status"] == "complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
