#!/usr/bin/env python3
"""Complete the missing fused halves, sequentially at 5s then 10s, on 3/4 GPUs.

Reuse the existing Dense and legacy arms; never copy or overwrite old outputs.
The full50 object is the complete four-arm result, unlike the new-half summaries.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import os
from pathlib import Path
import statistics
import subprocess
import sys

import run_route_mode_quality_25prompt_20261003 as study
from run_legacy_external_no_qk_full50_10s768p_20261004 import paired

NAME = os.environ.get("H3_ROUTE_FULL50_NAME", "route_2x2_full50_3gpu_20261004")
FIRST_NAME = "route_mode_quality_25prompt_20261003"
FUSED = ("fused_threshold", "fused_packed_external_no_route_qk")
LEGACY = ("legacy_threshold", "legacy_packed_external_no_route_qk")
ARMS = (LEGACY[0], FUSED[0], LEGACY[1], FUSED[1])
METRICS = ("psnr_db", "ssim", "lpips")
LEGACY5 = study.EXPERIMENTS / "legacy_route_full50_5s768p_20261003_5s768p"
LEGACY10 = study.EXPERIMENTS / "legacy_external_no_qk_full50_20261004_10s768p"
FIRST_LEGACY = "legacy_packed_external_no_route_qk_25prompt_20261003"


def first_root(duration):
    return study.EXPERIMENTS / f"{FIRST_NAME}_{duration}s768p"


def first_out(duration):
    return study.OUTPUTS / f"{FIRST_NAME}_{duration}s768p"


def reference_manifest(duration):
    if duration == 5:
        return Path(study.base.read(LEGACY5 / "results.json")["full50"]["combined_manifests"]["dense"])
    return study.SOURCE / "dense/generation_manifest.json"


def core_hashes():
    paths = [Path(__file__), Path(study.__file__),
             Path(study.base.IMPL) / "h3_sparse_attention/processor.py",
             Path(study.base.IMPL) / "h3_sparse_attention/sol_topk_cutoff.py",
             Path(study.base.IMPL) / "h3_sparse_attention/sol_numerator_virtual_q.py",
             Path(study.base.IMPL) / "h3_sparse_attention/spark_reweight_sm120.py"]
    return {str(p): study.base.sha(p) for p in paths}


def configure():
    study.NAME = NAME
    study.CASES = tuple(range(26, 51))
    study.GPUS = tuple(int(x) for x in os.environ.get("H3_EXPERIMENT_GPUS", "0,1,2").split(","))
    if len(study.GPUS) not in (3, 4) or len(set(study.GPUS)) != len(study.GPUS):
        raise ValueError("this runner requires three or four distinct GPUs")
    study.ALL_METHODS = FUSED
    study.GENERATED = {5: FUSED, 10: FUSED}
    original_file = study.__file__
    original_prepare = study.prepare
    study.__file__ = str(Path(__file__).resolve())

    def prepare(duration):
        if duration == 10:
            five = study.base.read(study.roots(5)[0] / "results.json")
            if five.get("full50", {}).get("status") != "complete":
                raise RuntimeError("finish all 5s generation, decoding, scoring and aggregation first")
        if study.base.read(first_root(duration) / "results.json").get("status") != "complete":
            raise RuntimeError("first-half fused results must be complete")
        refs = study.base.read(reference_manifest(duration))
        if [int(r["index"]) for r in refs["records"]] != list(range(1, 51)):
            raise ValueError("Dense reference is not exactly cases 1--50")
        for method in FUSED:
            manifest = study.base.read(first_out(duration) / method / "generation_manifest.json")
            if [int(r["index"]) for r in manifest["records"]] != list(range(1, 26)):
                raise ValueError(f"invalid reused manifest: {method}")
            expected = study.base.read(first_root(duration) / "protocol.json")["configs"][method]
            if expected != __import__("json").loads(__import__("json").dumps(dataclasses.asdict(study.config(method)))):
                raise ValueError(f"configuration changed: {method}")
        root, out = study.roots(duration)
        if root.exists() or out.exists():
            p = study.base.read(root / "protocol.json")
            if p["case_indices"] != list(study.CASES) or p["generated_methods"] != list(FUSED):
                raise ValueError("resume protocol mismatch")
            for path, digest in p["source_sha256"].items():
                if study.base.sha(path) != digest:
                    raise ValueError(f"resume source changed: {path}")
            print("RESUMING", root, flush=True)
            return
        original_prepare(duration)
        p = study.base.read(root / "protocol.json")
        p.update(purpose="Complete legacy/fused x threshold/external-no-QK full50 matrix",
                 reused_case_indices=list(range(1, 26)), generated_case_indices=list(study.CASES),
                 dense_reference_manifest=str(reference_manifest(duration)),
                 reused_first_half=str(first_root(duration)),
                 weight_load_slots=int(os.environ.get("H3_WEIGHT_LOAD_LOCK_SLOTS", str(len(study.GPUS)))),
                 source_sha256={**core_hashes(), original_file: study.base.sha(original_file)},
                 baseline_caveat="Historical reused legacy arms and first-half fused arms; not a newly rerun same-snapshot timing matrix",
                 vbench="not included", ordering_constraint="5s fully completes before 10s starts")
        study.base.write(root / "protocol.json", p)

    study.prepare = prepare

    def quality(duration):
        root, out = study.roots(duration)
        summaries = {}
        for method in FUSED:
            work = root / "quality" / method
            cfg = dict(work_dir=str(work), reference_manifest=str(reference_manifest(duration)),
                       candidate_manifest=str(out / method / "generation_manifest.json"),
                       method=method, cases=list(study.CASES), workers=len(study.GPUS))
            config_path = root / "quality" / f"{method}_config.json"
            study.base.write(config_path, cfg)
            subprocess.run([str(study.base.PYTHON), str(study.QUALITY_SCRIPT), "run", "--config", str(config_path)],
                           check=True, env={**os.environ, **study.base.ENV,
                                            "CUDA_VISIBLE_DEVICES": ",".join(map(str, study.GPUS)), "H3_NUM_GPUS": str(len(study.GPUS))})
            summaries[method] = study.base.read(work / f"{method}_quality_results.json")["summary"]
        result = study.base.read(root / "results.json")
        result.update(status="new_half_quality_complete", quality=summaries)
        study.base.write(root / "results.json", result)
        study.base.write(root / "status.json", dict(status="running", stage="full50_audit"))

    study.quality = quality


def locations(duration, method, index):
    root, _ = study.roots(duration)
    if method in FUSED:
        location = first_root(duration) if index <= 25 else root
    elif duration == 5:
        location = (first_root(5) if method == LEGACY[0] else
                    study.EXPERIMENTS / f"{FIRST_LEGACY}_5s768p") if index <= 25 else LEGACY5
    elif method == LEGACY[1]:
        location = study.EXPERIMENTS / f"{FIRST_LEGACY}_10s768p" if index <= 25 else LEGACY10
    else:
        quality = study.OLD_10S / "quality_work/quality" / f"topk10_reblock_global_reweight_{index:02}.json"
        return quality, None
    return location / "quality" / method / f"{method}_{index:02}.json", location / "records" / method / f"case_{index:02}.json"


def finalize(duration):
    root, out = study.roots(duration)
    refs = {int(x["index"]): x for x in study.base.read(reference_manifest(duration))["records"]}
    cases = study.base.read(study.SAMPLES)
    rows, runtime, manifests = {}, {}, {}
    old_runtime = study.base.read(study.OLD_10S / "results.json")["summary"]["topk10_reblock_global_reweight"]["per_prompt_seconds"] if duration == 10 else {}
    for method in ARMS:
        rows[method], runtime[method], records = [], [], []
        for case in cases:
            index = int(case["index"])
            quality_path, record_path = locations(duration, method, index)
            quality = study.base.read(quality_path)
            if quality["case"] != index or quality["frames"] != duration * 24:
                raise ValueError(f"quality case/frames mismatch: {method}/{index}")
            if quality["reference_sha256"] != refs[index]["sha256"]:
                raise ValueError(f"Dense reference mismatch: {method}/{index}")
            video = Path(quality["video_path"])
            if study.base.sha(video) != quality["video_sha256"]:
                raise ValueError(f"video mismatch: {method}/{index}")
            if record_path is not None:
                record = study.base.read(record_path)
                prompt_hash = hashlib.sha256(case["generation_prompt"].encode()).hexdigest()
                if record["case"] != index or record["sample_id"] != case["sample_id"] or record["prompt_sha256"] != prompt_hash:
                    raise ValueError(f"identity mismatch: {method}/{index}")
                seconds = record["denoise_seconds"]
            else:
                seconds = float(old_runtime[str(index)])
            rows[method].append(quality)
            runtime[method].append(float(seconds))
            records.append({**refs[index], "output_path": str(video), "sha256": quality["video_sha256"]})
        manifest_path = out / "full50" / method / "generation_manifest.json"
        study.base.write(manifest_path, dict(schema_version=1, status="passed", method=method, sample_count=50,
            settings=dict(seed=42, steps=20, frames=duration * 24, height=768, width=1344, fps=24), records=records))
        manifests[method] = str(manifest_path)
    def comparison(a, b):
        return {metric: paired([x[metric] - y[metric] for x, y in zip(rows[a], rows[b], strict=True)],
                               lower_is_better=metric == "lpips") for metric in METRICS}
    comparisons = {f"{a}_minus_{b}": comparison(a, b) for a, b in itertools.combinations(ARMS, 2)}
    interaction = {
        metric: paired([(rows[FUSED[1]][i][metric] - rows[FUSED[0]][i][metric]) -
                        (rows[LEGACY[1]][i][metric] - rows[LEGACY[0]][i][metric]) for i in range(50)],
                       lower_is_better=metric == "lpips") for metric in METRICS
    }
    result = study.base.read(root / "results.json")
    result.update(status="complete", full50=dict(status="complete", sample_count=50, case_indices=list(range(1, 51)),
        methods=list(ARMS), combined_manifests=manifests, reference_manifest=str(reference_manifest(duration)),
        runtime={m: dict(count=50, mean_seconds=statistics.fmean(v), stdev_seconds=statistics.stdev(v), values=v) for m, v in runtime.items()},
        quality_vs_dense={m: {metric: statistics.fmean(x[metric] for x in rows[m]) for metric in METRICS} for m in ARMS},
        paired_quality=comparisons, factorial_interaction=interaction,
        statistical_caveat="Exploratory unadjusted paired tests; nonsignificance does not establish equivalence",
        baseline_caveat="Legacy arms and first-half fused arms reused from historical runs; hardware/concurrency differences may affect timing",
        vbench=dict(status="not_requested")))
    study.base.write(root / "results.json", result)
    protocol = study.base.read(root / "protocol.json")
    protocol["status"] = "complete"
    study.base.write(root / "protocol.json", protocol)
    study.base.write(root / "status.json", dict(status="complete", stage="full50_complete", sample_count=50))
    print("FULL50 COMPLETE", duration, result["full50"]["quality_vs_dense"], flush=True)


def main():
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        durations = [int(sys.argv[2])] if len(sys.argv) > 2 else [5, 10]
        for duration in durations:
            root, _ = study.roots(duration)
            if (root / "results.json").exists() and study.base.read(root / "results.json").get("full50", {}).get("status") == "complete":
                print("ALREADY COMPLETE", duration, flush=True)
                continue
            study.run(duration)
            finalize(duration)
    elif command == "finalize":
        finalize(int(sys.argv[2]))
    elif command in ("generate_worker", "decode_worker"):
        getattr(study, command)(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise ValueError(command)


if __name__ == "__main__":
    main()
