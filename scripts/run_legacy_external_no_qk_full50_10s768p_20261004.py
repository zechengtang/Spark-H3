#!/usr/bin/env python3
"""Complete 10s/768p legacy external-no-QK cases 26--50 and combine all 50."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import run_legacy_packed_external_no_route_qk_25prompt_20261003 as previous

study = previous.study
METHOD = previous.METHOD
NAME = "legacy_external_no_qk_full50_20261004"
FIRST_EXP = study.EXPERIMENTS / f"{previous.NAME}_10s768p"
FIRST_OUT = study.OUTPUTS / f"{previous.NAME}_10s768p"
METRICS = ("psnr_db", "ssim", "lpips")


def configure() -> None:
    previous.configure()
    study.NAME = NAME
    study.CASES = tuple(range(26, 51))
    study.__file__ = str(Path(__file__).resolve())
    study.ALL_METHODS = (METHOD,)
    study.GENERATED = {5: (), 10: (METHOD,)}
    original_prepare = study.prepare

    def prepare(duration: int) -> None:
        if duration != 10:
            raise ValueError("this continuation is 10s-only")
        first = study.base.read(FIRST_EXP / "results.json")
        if first.get("status") != "complete":
            raise RuntimeError("first 25 cases must complete before extension")
        first_manifest = study.base.read(FIRST_OUT / METHOD / "generation_manifest.json")
        if [int(row["index"]) for row in first_manifest["records"]] != list(range(1, 26)):
            raise ValueError("first manifest is not exactly cases 1--25")
        for case in (1, 25):
            recorded = study.base.read(FIRST_EXP / "records" / METHOD / f"case_{case:02}.json")
            import dataclasses
            expected_config = json.loads(json.dumps(dataclasses.asdict(study.config(METHOD))))
            if recorded["config"] != expected_config:
                raise ValueError(f"configuration differs from reused case {case}")
        root, out = study.roots(10)
        if root.exists() or out.exists():
            if not (root.exists() and out.exists() and (root / "protocol.json").exists()):
                raise FileExistsError("incomplete resume roots")
            protocol = study.base.read(root / "protocol.json")
            if protocol["case_indices"] != list(study.CASES) or protocol["generated_methods"] != [METHOD]:
                raise ValueError("resume protocol does not match")
            print("RESUMING", root, flush=True)
            return
        original_prepare(10)
        protocol = study.base.read(root / "protocol.json")
        protocol.update(
            purpose="Complete all 20pct 50 prompts for legacy packed external no-route-QK at 10s/768p",
            generated_case_indices=list(study.CASES),
            reused_case_indices=list(range(1, 26)),
            reused_candidate_source=str(FIRST_EXP),
            reused_10s_methods=["dense", "legacy_threshold"],
            source_sha256={str(path): study.base.sha(path) for path in (
                Path(__file__).resolve(),
                Path(study.base.IMPL) / "h3_sparse_attention/processor.py",
                Path(study.base.IMPL) / "h3_sparse_attention/sol_topk_cutoff.py",
                Path(study.base.IMPL) / "h3_sparse_attention/sol_numerator_virtual_q.py",
                Path(study.base.IMPL) / "h3_sparse_attention/spark_reweight_sm120.py",
            )},
            baseline_caveat="legacy_threshold and Dense are historical reused references, not newly generated same-snapshot arms",
        )
        study.base.write(root / "protocol.json", protocol)

    study.prepare = prepare


def paired(values: list[float], *, lower_is_better: bool = False) -> dict:
    from scipy.stats import t, ttest_1samp
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = float(t.ppf(.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {
        "count": len(values), "mean_delta": mean, "stdev_delta": sd,
        "ci95": [mean - half, mean + half],
        "paired_t_pvalue": float(ttest_1samp(values, 0).pvalue),
        "improved": sum(value < 0 if lower_is_better else value > 0 for value in values),
        "degraded": sum(value > 0 if lower_is_better else value < 0 for value in values),
        "values": values,
    }


def finalize() -> None:
    root, out = study.roots(10)
    manifests = [study.base.read(path / METHOD / "generation_manifest.json") for path in (FIRST_OUT, out)]
    records = [row for manifest in manifests for row in manifest["records"]]
    if [int(row["index"]) for row in records] != list(range(1, 51)):
        raise ValueError("combined candidate indices must be exactly 1--50")
    source_cases = study.base.read(study.SAMPLES)
    refs = {int(row["index"]): row for row in study.base.read(study.SOURCE / "dense/generation_manifest.json")["records"]}
    candidate_quality, baseline_quality, runtime = [], [], []
    for case, video in zip(source_cases, records, strict=True):
        index = int(case["index"])
        prompt_hash = hashlib.sha256(case["generation_prompt"].encode()).hexdigest()
        if video["sample_id"] != case["sample_id"] or video["prompt_sha256"] != prompt_hash:
            raise ValueError(f"candidate identity mismatch: {index}")
        location = FIRST_EXP if index <= 25 else root
        quality = study.base.read(location / "quality" / METHOD / f"{METHOD}_{index:02}.json")
        baseline = study.base.read(previous.OLD_10S / "quality_work/quality" / f"topk10_reblock_global_reweight_{index:02}.json")
        if quality["case"] != index or baseline["case"] != index:
            raise ValueError(f"quality case mismatch: {index}")
        if quality.get("reference_sha256") != refs[index]["sha256"]:
            raise ValueError(f"candidate reference mismatch: {index}")
        if baseline.get("reference_sha256") != refs[index]["sha256"]:
            raise ValueError(f"baseline reference mismatch: {index}")
        if quality.get("video_sha256") != video["sha256"] or study.base.sha(video["output_path"]) != video["sha256"]:
            raise ValueError(f"candidate video mismatch: {index}")
        candidate_quality.append(quality)
        baseline_quality.append(baseline)
        runtime.append(float(study.base.read(location / "records" / METHOD / f"case_{index:02}.json")["denoise_seconds"]))
    combined_manifest = out / "full50" / METHOD / "generation_manifest.json"
    study.base.write(combined_manifest, {
        "schema_version": 1, "status": "passed", "method": METHOD,
        "sample_count": 50, "settings": manifests[1]["settings"], "records": records,
        "reused_cases": list(range(1, 26)), "generated_cases": list(range(26, 51)),
    })
    old_runtime = study.base.read(previous.OLD_10S / "results.json")["summary"]["topk10_reblock_global_reweight"]["per_prompt_seconds"]
    baseline_runtime = [float(old_runtime[str(case)]) for case in range(1, 51)]
    candidate_mean, baseline_mean = statistics.fmean(runtime), statistics.fmean(baseline_runtime)
    full50 = {
        "status": "complete", "case_indices": list(range(1, 51)),
        "combined_candidate_manifest": str(combined_manifest),
        "reference_manifest": str(study.SOURCE / "dense/generation_manifest.json"),
        "baseline_source": str(previous.OLD_10S),
        "baseline_caveat": "historical reused Dense and legacy_threshold; candidate first half reused and second half newly generated",
        "runtime": {
            METHOD: {"count": 50, "mean_seconds": candidate_mean, "values": runtime},
            "legacy_threshold": {"count": 50, "mean_seconds": baseline_mean, "values": baseline_runtime},
        },
        "quality_vs_dense": {
            METHOD: {metric: statistics.fmean(row[metric] for row in candidate_quality) for metric in METRICS},
            "legacy_threshold": {metric: statistics.fmean(row[metric] for row in baseline_quality) for metric in METRICS},
        },
        "candidate_vs_legacy_threshold": {
            metric: paired([new[metric] - old[metric] for new, old in zip(candidate_quality, baseline_quality, strict=True)], lower_is_better=metric == "lpips")
            for metric in METRICS
        },
        "runtime_candidate_vs_legacy_threshold": {
            **paired([new-old for new,old in zip(runtime,baseline_runtime,strict=True)], lower_is_better=True),
            "time_reduction_percent": 100 * (1 - candidate_mean / baseline_mean),
        },
        "vbench": {"status": "not_requested_for_this_extension"},
    }
    result = study.base.read(root / "results.json")
    result.update(status="complete", full50=full50)
    study.base.write(root / "results.json", result)
    study.base.write(root / "status.json", {"status": "complete", "stage": "full50_complete", "sample_count": 50})
    print("FULL50 COMPLETE", full50["quality_vs_dense"], full50["runtime_candidate_vs_legacy_threshold"], flush=True)


def main() -> None:
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        study.run(10)
        finalize()
    elif command == "finalize":
        finalize()
    elif command in ("generate_worker", "decode_worker"):
        getattr(study, command)(10, int(sys.argv[-1]))
    else:
        raise ValueError(command)


if __name__ == "__main__":
    main()
