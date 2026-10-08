#!/usr/bin/env python3
"""Resume-safe 50-prompt Spark reweight component ablation on GPU0-3.

Matches the completed 10-prompt numeric/component settings. Reuses existing
matching full (25) and other-component (10 each) artifacts by sample ID;
generates only the missing 145 method/prompt pairs.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import pytorch_spark_route_execution_10prompt_5s768p_20260925 as route
import diffusers_spark_reweight_components_10prompt_20260926 as previous
import diffusers_spark_reweight_precision_25prompt_20260926 as precision


NAME = "diffusers_spark_reweight_components_50prompt_10s768p_20260926"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = route.base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
DENSE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json")
COND = DENSE.parent.parent
METHODS = ("full", "weights_only", "bias_only", "none")
GPUS = (0, 1, 2, 3)
CASES = tuple(range(1, 51))


def config(method):
    return previous.config(method)


def configure():
    route.ROOT, route.OUT = ROOT, OUT
    route.SAMPLES, route.CASES = SAMPLES, CASES
    route.METHODS, route.GPUS = METHODS, GPUS
    route.EXECUTIONS = {
        method: "packed_external_no_route_qk" for method in METHODS
    }
    route.DENSE_OUT = ROOT / "dense_reference"
    route.spark_config = config
    route.cases = cases
    route.base.FRAMES = 240
    route.base.INTERNAL_FRAMES = 244


def cases():
    return route.base.pipeline().load_cases(SAMPLES, list(CASES), expected_indices=CASES)


def _source(method):
    if method == "full":
        return (
            precision.ROOT / "records" / "spark_comfy_fp32",
            precision.OUT / "spark_comfy_fp32" / "generation_manifest.json",
        )
    return (
        previous.ROOT / "records" / method,
        previous.OUT / method / "generation_manifest.json",
    )


def _reusable(method):
    records, manifest_path = _source(method)
    source_protocol = json.loads((records.parents[1] / "protocol.json").read_text())
    # A matching attention config is not enough: the old sources explicitly
    # disabled block compilation, which changes the generated latents.
    if source_protocol.get("torch_compile") is not True:
        return {}
    manifest = json.loads(manifest_path.read_text())
    by_id = {}
    for row in manifest["records"]:
        old_index = int(row["index"])
        record = json.loads((records / f"case_{old_index:02}.json").read_text())
        if record["sample_id"] != row["sample_id"]:
            raise ValueError(f"source record mismatch: {method}/{old_index}")
        source_config = dict(record["config"])
        if record.get("torch_compile") is not True:
            raise ValueError(f"source compile provenance mismatch: {method}/{old_index}")
        source_config.setdefault("sol_reweight_components", "full")
        # JSON records serialize tuple-valued config fields as lists.
        expected_config = json.loads(json.dumps(dataclasses.asdict(config(method))))
        if source_config != expected_config:
            raise ValueError(f"source config mismatch: {method}/{old_index}")
        if not Path(row["output_path"]).is_file() or not Path(record["latent_path"]).is_file():
            raise FileNotFoundError(f"source artifact missing: {method}/{old_index}")
        by_id[row["sample_id"]] = (record, row)
    return by_id


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    selected = cases()
    dense = json.loads(DENSE.read_text())
    dense_by_id = {row["sample_id"]: row for row in dense["records"]}
    if len(selected) != 50 or len(dense_by_id) != 50:
        raise ValueError("50-prompt dataset or dense reference is incomplete")
    reuse = {method: _reusable(method) for method in METHODS}
    for method in METHODS:
        for folder in (ROOT / "records" / method, ROOT / "warmup" / method,
                       ROOT / "decode_records" / method,
                       OUT / method / "latents", OUT / method / "videos"):
            folder.mkdir(parents=True, exist_ok=True)
    (ROOT / "dense_reference").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(COND / name, target_is_directory=name == "conditioning_cache")
    dense_rows = []
    reused = {method: 0 for method in METHODS}
    for case in selected:
        dense_row = dict(dense_by_id[case["sample_id"]])
        if dense_row["prompt_sha256"] != case["prompt_sha256"] or not Path(dense_row["output_path"]).is_file():
            raise ValueError(f"dense mismatch: {case['sample_id']}")
        dense_row["index"] = case["index"]
        dense_rows.append(dense_row)
        for method in METHODS:
            found = reuse[method].get(case["sample_id"])
            if found is None:
                continue
            old_record, old_video = found
            if old_record["prompt_sha256"] != case["prompt_sha256"] or old_video["prompt_sha256"] != case["prompt_sha256"]:
                raise ValueError(f"reused prompt mismatch: {method}/{case['sample_id']}")
            record = dict(old_record, method=method, case=case["index"],
                          reused_from=str(_source(method)[1]))
            route.base.write(ROOT / "records" / method / f"case_{case['index']:02}.json", record)
            route.base.write(ROOT / "decode_records" / method / f"case_{case['index']:02}.json",
                             dict(status="complete", method=method, case=case["index"],
                                  sample_id=case["sample_id"], output_path=old_video["output_path"],
                                  sha256=old_video["sha256"], reused_from=str(_source(method)[1])))
            reused[method] += 1
    route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=50,
        settings=dense.get("settings"), records=dense_rows, source_manifest=str(DENSE)))
    route.base.write(ROOT / "protocol.json", dict(
        name=NAME, status="prepared", pipeline="native PyTorch/Diffusers",
        purpose="Matched 50-prompt global-reweight weights/bias ablation",
        dataset="VBench core-five 20pct first 50; same dense reference as historical reweight run",
        cases=selected, methods=METHODS,
        configs={method: dataclasses.asdict(config(method)) for method in METHODS},
        dense_reference_manifest=str(DENSE), conditioning_source=str(COND),
        seed=42, requested_steps=20, actual_evaluations=19,
        torch_compile=True,
        frames=240, resolution="1344x768", gpus=GPUS,
        reused_counts=reused, newly_required=200-sum(reused.values()),
        timing="synchronized denoise; reused runtimes from source experiments, not paired with new runs",
        warmup="one excluded full denoise per method/GPU",
        metrics="PSNR/SSIM/LPIPS versus dense plus source-assigned core-five VBench for every arm",
    ))
    route.base.pipeline().configure_denoise_workflow(route.parse_args("denoise"), selected)


def _record_ready(method, case):
    path = ROOT / "records" / method / f"case_{case['index']:02}.json"
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    if record["sample_id"] != case["sample_id"] or record["prompt_sha256"] != case["prompt_sha256"]:
        raise ValueError(f"record mismatch: {method}/{case['index']}")
    if not Path(record["latent_path"]).is_file():
        raise FileNotFoundError(record["latent_path"])
    pipeline = route.base.pipeline()
    latent = Path(record["latent_path"])
    provenance_path = latent.parent.parent / "denoise_records" / f"{latent.stem}.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"missing compile provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("torch_compile") is not True
            or provenance.get("compile_wrapped_blocks") != pipeline.EXPECTED_TRANSFORMER_BLOCKS
            or provenance.get("latent_sha256") != record.get("latent_sha256")):
        raise RuntimeError(f"invalid compile provenance: {provenance_path}")
    return True


def generate_worker(rank):
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assigned = cases()[rank::len(GPUS)]
    p = route.base.pipeline()
    workflow, states = p.configure_denoise_workflow(route.parse_args("denoise"), assigned)
    pipe, manager, acceleration, placement = p.load_denoiser(route.parse_args("denoise"), workflow)
    try:
        for method in route.method_order(assigned[0]["index"], rank):
            row = route.run_one(pipe, p, states[0], assigned[0], method, placement, warmup=True)
            route.base.write(ROOT / "warmup" / method / f"gpu{GPUS[rank]}.json", row)
            print("WARMUP", rank, method, f"{row['seconds']:.3f}s", flush=True)
        for case, state in zip(assigned, states, strict=True):
            for method in route.method_order(case["index"], rank):
                if _record_ready(method, case):
                    continue
                row = route.run_one(pipe, p, state, case, method, placement)
                print("MEASURED", rank, method, case["index"], f"{row['denoise_seconds']:.3f}s", flush=True)
        route.base.write(ROOT / f"generate_gpu{GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def aggregate():
    rows = {method: [json.loads((ROOT / "records" / method / f"case_{case:02}.json").read_text())
                     for case in CASES] for method in METHODS}
    summary = {}
    for method, records in rows.items():
        seconds = [record["denoise_seconds"] for record in records]
        summary[method] = dict(mean_seconds=statistics.mean(seconds),
                               median_seconds=statistics.median(seconds),
                               reused=sum("reused_from" in record for record in records),
                               measured=sum("reused_from" not in record for record in records))
    route.base.write(ROOT / "results.json", dict(status="runtime_complete", summary=summary,
        caveat="mixed reused/new runtimes are not a paired speed comparison"))


def decode_worker(rank):
    """Serialize per-GPU decode for safe overlap with late denoising."""
    import fcntl

    lock_path = ROOT / f"decode_gpu{GPUS[rank]}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _decode_worker_unlocked(rank)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _decode_worker_unlocked(rank):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    if str(route.FV_EVAL) not in sys.path:
        sys.path.insert(0, str(route.FV_EVAL))
    from fasth3_vbench_archive import archive, file_sha

    p = route.base.pipeline()
    assigned = cases()[rank::len(GPUS)]
    pipe, manager, acceleration = p.load_decoder(route.parse_args("decode"))
    try:
        vae_dtype = next(pipe.vae.parameters()).dtype
        audio_vae_dtype = next(pipe.audio_vae.parameters()).dtype
        print("DECODE_MODEL_DTYPES", rank, str(vae_dtype), str(audio_vae_dtype), flush=True)
        route.base.write(ROOT / f"decode_model_dtypes_gpu{GPUS[rank]}.json", dict(
            status="checked", load_components_dtype="torch.bfloat16",
            acceleration_vae_fp16=False, vae_dtype=str(vae_dtype),
            audio_vae_dtype=str(audio_vae_dtype)))
        # A live preflight after load_decoder(dtype=BF16) confirmed both VAE
        # modules remain FP32. The prior no-op acceleration never converted
        # those weights, and the new hook must preserve that runtime dtype.
        if vae_dtype != torch.float32 or audio_vae_dtype != torch.float32:
            raise RuntimeError(
                "decode VAE dtype changed from prior no-op FP32 runtime path: "
                f"video={vae_dtype}, audio={audio_vae_dtype}"
            )
        for case in assigned:
            stem = p.case_stem(case)
            for method in METHODS:
                record_path = ROOT / "decode_records" / method / f"case_{case['index']:02}.json"
                if record_path.is_file():
                    record = json.loads(record_path.read_text())
                    if record["sample_id"] != case["sample_id"] or not Path(record["output_path"]).is_file():
                        raise ValueError(f"invalid reused decode record: {method}/{case['index']}")
                    continue
                payload = torch.load(OUT / method / "latents" / f"{stem}.pt",
                                     map_location="cpu", weights_only=True)
                with torch.inference_mode():
                    result = pipe(latents=payload["latents"].to("cuda"),
                                  audio_latents=payload["audio_latents"].to("cuda"),
                                  output_type="np", output=["videos", "audio", "sampling_rate"])
                frames = np.clip(np.round(result["videos"][0][:route.base.FRAMES] * 255), 0, 255).astype(np.uint8)
                target = OUT / method / "videos" / f"{stem}.mkv"
                metadata = archive(frames, result["audio"][0],
                                   int(result["sampling_rate"]), target,
                                   fps=route.base.FPS, threads=8)
                route.base.write(record_path, dict(
                    status="complete", method=method, case=case["index"],
                    sample_id=case["sample_id"], output_path=str(target),
                    sha256=file_sha(target), archive=metadata))
                print("DECODED", rank, method, case["index"], flush=True)
        route.base.write(ROOT / f"decode_gpu{GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def manifests():
    for method in METHODS:
        rows = []
        for case in cases():
            rec = json.loads((ROOT / "decode_records" / method /
                              f"case_{case['index']:02}.json").read_text())
            rows.append(dict(index=case["index"], sample_id=case["sample_id"],
                             prompt_sha256=case["prompt_sha256"],
                             evaluation_prompt=case["original_prompt"],
                             vbench_dimensions=case["vbench_dimensions"],
                             output_path=rec["output_path"], sha256=rec["sha256"],
                             video=dict(frames=route.base.FRAMES,
                                        width=route.base.WIDTH, height=route.base.HEIGHT,
                                        fps=float(route.base.FPS))))
        route.base.write(OUT / method / "generation_manifest.json", dict(
            schema_version=1, status="passed", method=method, sample_count=50,
            settings=dict(seed=route.base.SEED, steps=route.base.STEPS,
                          frames=route.base.FRAMES, height=route.base.HEIGHT,
                          width=route.base.WIDTH, fps=route.base.FPS), records=rows))


def quality():
    for method in METHODS:
        work = ROOT / "quality" / method
        config_path = ROOT / "quality" / f"{method}_config.json"
        route.base.write(config_path, dict(
            work_dir=str(work),
            reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
            candidate_manifest=str(OUT / method / "generation_manifest.json"),
            method=method, cases=list(CASES), workers=len(GPUS)))
        subprocess.run([str(route.base.PYTHON), str(route.QUALITY_SCRIPT), "run",
                        "--config", str(config_path)], check=True,
                       env={**os.environ, **route.base.ENV,
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3", "H3_NUM_GPUS": "4"})
    data = json.loads((ROOT / "results.json").read_text())
    data["quality"] = {method: json.loads((ROOT / "quality" / method /
                                           f"{method}_quality_results.json").read_text())["summary"]
                       for method in METHODS}
    data["status"] = "quality_complete"
    route.base.write(ROOT / "results.json", data)


def paired():
    """Summarize per-prompt paired quality without pooling video frames."""
    metric_names = ("psnr_db", "ssim", "lpips")
    by_method = {
        method: {
            case: json.loads((ROOT / "quality" / method / f"{method}_{case:02}.json").read_text())
            for case in CASES
        }
        for method in METHODS
    }
    comparisons = (
        ("full", "none"),
        ("full", "weights_only"),
        ("full", "bias_only"),
        ("weights_only", "none"),
        ("bias_only", "none"),
    )
    pairs = {}
    runtime_by_method = {
        method: {
            case: json.loads((ROOT / "records" / method / f"case_{case:02}.json").read_text())
            for case in CASES
        }
        for method in METHODS
    }
    runtime_pairs = {}
    for left, right in comparisons:
        metrics = {}
        for metric in metric_names:
            diffs = [by_method[left][case][metric] - by_method[right][case][metric]
                     for case in CASES]
            favorable = diffs if metric != "lpips" else [-value for value in diffs]
            metrics[metric] = dict(
                mean_delta=statistics.mean(diffs),
                median_delta=statistics.median(diffs),
                wins=sum(value > 0 for value in favorable),
                losses=sum(value < 0 for value in favorable),
                ties=sum(value == 0 for value in favorable),
                # Approximate two-sided 95% Student-t interval for n=50.
                ci95=[statistics.mean(diffs) - 2.0096 * statistics.stdev(diffs) / math.sqrt(len(diffs)),
                      statistics.mean(diffs) + 2.0096 * statistics.stdev(diffs) / math.sqrt(len(diffs))],
            )
        pairs[f"{left}_vs_{right}"] = metrics
        # Only newly measured runs are comparable: reused records came from
        # different jobs, GPUs, and dates. Both arms for a prompt share a GPU.
        runtime_cases = [
            case for case in CASES
            if "reused_from" not in runtime_by_method[left][case]
            and "reused_from" not in runtime_by_method[right][case]
        ]
        left_times = [runtime_by_method[left][case]["denoise_seconds"] for case in runtime_cases]
        right_times = [runtime_by_method[right][case]["denoise_seconds"] for case in runtime_cases]
        differences = [a - b for a, b in zip(left_times, right_times)]
        n = len(differences)
        # 95% Student-t critical values: df 24 for full comparisons, df 39
        # for comparisons among arms with only ten reused records.
        critical = 2.0639 if n == 25 else 2.0227 if n == 40 else None
        if critical is None:
            raise ValueError(f"unexpected paired runtime sample count {n} for {left}/{right}")
        margin = critical * statistics.stdev(differences) / math.sqrt(n)
        runtime_pairs[f"{left}_vs_{right}"] = dict(
            cases=runtime_cases, count=n,
            left_mean_seconds=statistics.mean(left_times),
            right_mean_seconds=statistics.mean(right_times),
            difference_seconds=statistics.mean(differences),
            difference_ci95=[statistics.mean(differences) - margin,
                             statistics.mean(differences) + margin],
            ratio=statistics.mean(left_times) / statistics.mean(right_times),
            left_faster=sum(value < 0 for value in differences),
            right_faster=sum(value > 0 for value in differences),
            ties=sum(value == 0 for value in differences),
        )
    route.base.write(ROOT / "paired_results.json", dict(
        status="complete", cases=list(CASES), metric_orientation=dict(
            psnr_db="higher", ssim="higher", lpips="lower"), pairs=pairs,
        runtime_caveat="Only prompts newly generated in both arms are paired; reused runtimes excluded",
        runtime_pairs=runtime_pairs))
    return pairs


def vbench():
    """Score the source-assigned core-five dimensions on GPU0-3."""
    adapter = ROOT / "vbench.py"
    source = Path(__file__).with_name(
        "diffusers_spark_reweight_components_50prompt_vbench_20260926.py")
    if not adapter.exists():
        shutil.copy2(source, adapter)
    spec = importlib.util.spec_from_file_location("experiment_vbench", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not (ROOT / "vbench" / "protocol.json").exists():
        module.prepare()
    subprocess.run([
        str(route.base.PYTHON), str(route.base.BENCH / "scripts" / "h3_vbench_queue.py"),
        "run", "--experiment", str(ROOT), "--gpus", "0", "1", "2", "3",
    ], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    result = json.loads((ROOT / "results.json").read_text())
    result["vbench"] = json.loads((ROOT / "vbench" / "results.json").read_text())[
        "scores_percent_assigned_prompts"]
    result["status"] = "vbench_complete"
    route.base.write(ROOT / "results.json", result)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(route.base.PYTHON), str(Path(__file__).resolve()), stage, str(rank)],
            env={**os.environ, **route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT)
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def run():
    configure()
    if not ROOT.exists() and not OUT.exists():
        prepare()
    elif not (ROOT / "protocol.json").is_file():
        raise RuntimeError("experiment directory exists without protocol; refusing to overwrite")
    spawn("generate_worker")
    aggregate()
    spawn("decode_worker")
    manifests()
    quality()
    paired()
    vbench()


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        globals()[command](int(sys.argv[2]))
    elif command in ("run", "prepare", "aggregate", "manifests", "quality", "paired", "vbench"):
        globals()[command]()
    else:
        raise ValueError(command)
