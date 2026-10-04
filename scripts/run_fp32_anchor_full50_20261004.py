#!/usr/bin/env python3
"""FP32-anchor-only ablation: same 50 prompts, 5s then 10s, BF16 model."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import run_route_mode_quality_25prompt_20261003 as study
import run_route_2x2_full50_3gpu_20261004 as route
from run_legacy_external_no_qk_full50_10s768p_20261004 import paired

NAME = "fp32_anchor_full50_4gpu_20261004"
METHOD = "legacy_threshold_fp32_anchor"
METRICS = ("psnr_db", "ssim", "lpips")
ORIGINAL_CONFIG = study.config


def config(method, steps=20):
    assert method == METHOD
    baseline = ORIGINAL_CONFIG("legacy_threshold", steps)
    result = dataclasses.replace(baseline, sol_global_anchor_dtype="float32")
    left, right = dataclasses.asdict(baseline), dataclasses.asdict(result)
    changed = [k for k in left if left[k] != right[k]]
    assert changed == ["sol_global_anchor_dtype"], changed
    assert baseline.sol_global_anchor_dtype == "bfloat16"
    assert result.landmark_tree_v2_midpoint_direction_mode == "legacy"
    assert result.sol_route_topk_execution == "threshold"
    assert result.sol_reweight_summary_math == baseline.sol_reweight_summary_math == "tensorcore"
    return result


def source_hashes():
    paths = [Path(__file__), Path(study.__file__), Path(route.__file__),
             Path(study.base.IMPL) / "h3_sparse_attention/processor.py",
             Path(study.base.IMPL) / "h3_sparse_attention/sol_numerator_virtual_q.py",
             Path(study.base.IMPL) / "h3_sparse_attention/spark_reweight_sm120.py"]
    return {str(p): study.base.sha(p) for p in paths}


def configure():
    hashes = source_hashes()
    study.NAME, study.CASES = NAME, tuple(range(1, 51))
    study.GPUS = tuple(int(x) for x in os.environ.get("H3_EXPERIMENT_GPUS", "0,1,2,3").split(","))
    assert len(study.GPUS) == 4 and len(set(study.GPUS)) == 4
    study.ALL_METHODS, study.GENERATED = (METHOD,), {5: (METHOD,), 10: (METHOD,)}
    original_prepare = study.prepare
    study.config, study.__file__ = config, str(Path(__file__).resolve())

    def prepare(duration):
        if duration == 10:
            assert study.base.read(study.roots(5)[0] / "results.json")["full50"]["status"] == "complete"
        root, out = study.roots(duration)
        if root.exists() or out.exists():
            protocol = study.base.read(root / "protocol.json")
            assert protocol["source_sha256"] == hashes, "anchor resume source changed"
            assert protocol["configs"][METHOD] == json.loads(json.dumps(dataclasses.asdict(config(METHOD))))
            print("RESUMING ANCHOR", duration, flush=True)
            return
        original_prepare(duration)
        protocol = study.base.read(root / "protocol.json")
        protocol.update(source_sha256=hashes, purpose="Change only sol_global_anchor_dtype from BF16 to FP32",
            only_changed_config_fields=["sol_global_anchor_dtype"], model_dtype="bfloat16",
            baseline_config=dataclasses.asdict(ORIGINAL_CONFIG("legacy_threshold")),
            dense_reference_manifest=str(route.reference_manifest(duration)),
            vbench="assigned core-five required by sequential supervisor", ordering="5s before 10s")
        study.base.write(root / "protocol.json", protocol)
    study.prepare = prepare

    def quality(duration):
        root, out = study.roots(duration)
        work = root / "quality" / METHOD
        cfg = root / "quality" / f"{METHOD}_config.json"
        study.base.write(cfg, dict(work_dir=str(work), reference_manifest=str(route.reference_manifest(duration)),
            candidate_manifest=str(out / METHOD / "generation_manifest.json"), method=METHOD,
            cases=list(study.CASES), workers=len(study.GPUS)))
        subprocess.run([str(study.base.PYTHON), str(study.QUALITY_SCRIPT), "run", "--config", str(cfg)], check=True,
            env={**os.environ, **study.base.ENV, "CUDA_VISIBLE_DEVICES": ",".join(map(str, study.GPUS)), "H3_NUM_GPUS": str(len(study.GPUS))})
        result = study.base.read(root / "results.json")
        result.update(status="quality_complete", quality={METHOD: study.base.read(work / f"{METHOD}_quality_results.json")["summary"]})
        study.base.write(root / "results.json", result)
    study.quality = quality


def finalize(duration):
    root, out = study.roots(duration)
    baseline_name = "legacy_threshold"
    refs = {int(x["index"]): x for x in study.base.read(route.reference_manifest(duration))["records"]}
    candidate, baseline, times, baseline_times = [], [], [], []
    old_runtime = study.base.read(study.OLD_10S / "results.json")["summary"]["topk10_reblock_global_reweight"]["per_prompt_seconds"] if duration == 10 else {}
    for i in range(1, 51):
        score = study.base.read(root / "quality" / METHOD / f"{METHOD}_{i:02}.json")
        record = study.base.read(root / "records" / METHOD / f"case_{i:02}.json")
        path, recpath = route.locations(duration, baseline_name, i)
        before = study.base.read(path)
        assert score["case"] == before["case"] == i and score["frames"] == before["frames"] == duration * 24
        assert score["reference_sha256"] == before["reference_sha256"] == refs[i]["sha256"]
        for row in (score, before):
            assert study.base.sha(row["video_path"]) == row["video_sha256"]
        assert record["config"] == json.loads(json.dumps(dataclasses.asdict(config(METHOD))))
        if recpath:
            prior = study.base.read(recpath)
            assert prior["config"] == json.loads(json.dumps(dataclasses.asdict(ORIGINAL_CONFIG(baseline_name))))
            assert record["sample_id"] == prior["sample_id"] and record["prompt_sha256"] == prior["prompt_sha256"]
            seconds = prior["denoise_seconds"]
        else:
            seconds = float(old_runtime[str(i)])
        candidate.append(score)
        baseline.append(before)
        times.append(record["denoise_seconds"])
        baseline_times.append(seconds)
    means = {m: {k: statistics.fmean(r[k] for r in rows) for k in METRICS}
             for m, rows in [(METHOD, candidate), (baseline_name, baseline)]}
    quality_delta = {k: paired([a[k] - b[k] for a, b in zip(candidate, baseline, strict=True)],
                       lower_is_better=k == "lpips") for k in METRICS}
    result = study.base.read(root / "results.json")
    result.update(status="complete", full50=dict(status="complete", sample_count=50,
        quality_vs_dense=means, paired_anchor_minus_baseline=quality_delta,
        runtime={METHOD: dict(count=50, mean_seconds=statistics.fmean(times), values=times),
                 baseline_name: dict(count=50, mean_seconds=statistics.fmean(baseline_times), values=baseline_times)},
        paired_runtime=paired([a - b for a, b in zip(times, baseline_times, strict=True)], lower_is_better=True),
        candidate_manifest=str(out / METHOD / "generation_manifest.json"),
        reference_manifest=str(route.reference_manifest(duration)),
        baseline_caveat="Historical BF16 anchor outputs reused; timing is descriptive across runs, not a same-session microbenchmark",
        vbench=dict(status="pending")))
    study.base.write(root / "results.json", result)
    study.base.write(root / "status.json", dict(status="complete", stage="rgb_quality_complete_vbench_pending"))
    print("FP32 ANCHOR FULL50 RGB COMPLETE", duration, means, flush=True)


def main():
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        getattr(study, command)(int(sys.argv[2]), int(sys.argv[3]))
        return
    assert command == "run"
    prior = study.EXPERIMENTS / "ref2va_8fps_phase_bias_20261004/results.json"
    assert study.base.read(prior)["status"] == "complete", "finish task2 first"
    for duration in (5, 10):
        root, _ = study.roots(duration)
        if (root / "results.json").exists() and study.base.read(root / "results.json").get("full50", {}).get("status") == "complete":
            continue
        study.run(duration)
        finalize(duration)


if __name__ == "__main__":
    main()
