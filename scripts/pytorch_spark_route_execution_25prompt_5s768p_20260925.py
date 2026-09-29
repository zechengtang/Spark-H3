#!/usr/bin/env python3
"""Extend the native-PyTorch Spark route-execution comparison to 25 prompts."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import pytorch_spark_route_execution_10prompt_5s768p_20260925 as exp


NAME = "pytorch_spark_route_execution_25prompt_5s768p_seed42_20260925"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
OLD_ROOT = exp.ROOT
OLD_OUT = exp.OUT
CASES = tuple(range(1, 26))
REUSED_CASES = tuple(range(1, 11))
NEW_CASES = tuple(range(11, 26))
TOTAL = len(CASES) * len(exp.METHODS)

# The imported runner helpers resolve these values in their defining module.
exp.NAME = NAME
exp.ROOT = ROOT
exp.OUT = OUT
exp.CASES = CASES


def cases():
    return exp.base.pipeline().load_cases(
        exp.SAMPLES, list(CASES), expected_indices=tuple(range(1, 51))
    )


exp.cases = cases


def link(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(source).resolve())


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    all_cases = cases()
    by_index = {case["index"]: case for case in all_cases}
    for method in exp.METHODS:
        (ROOT / "records" / method).mkdir(parents=True)
        (ROOT / "warmup" / method).mkdir(parents=True)
        (ROOT / "decode_records" / method).mkdir(parents=True)
        (ROOT / "quality" / method).mkdir(parents=True)
        (OUT / method / "latents").mkdir(parents=True)
        (OUT / method / "videos").mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        link(exp.base.CONDITIONING_SOURCE / name, OUT / name)

    for case_index in REUSED_CASES:
        case = by_index[case_index]
        stem = exp.base.pipeline().case_stem(case)
        for method in exp.METHODS:
            old_record_path = OLD_ROOT / "records" / method / f"case_{case_index:02}.json"
            old_record = exp.base.read(old_record_path)
            latent = OUT / method / "latents" / f"{stem}.pt"
            link(old_record["latent_path"], latent)
            exp.base.write(
                ROOT / "records" / method / f"case_{case_index:02}.json",
                {**old_record, "latent_path": str(latent), "reused_from": str(old_record_path)},
            )

            old_decode_path = OLD_ROOT / "decode_records" / method / f"case_{case_index:02}.json"
            old_decode = exp.base.read(old_decode_path)
            video = OUT / method / "videos" / f"{stem}.mkv"
            link(old_decode["output_path"], video)
            exp.base.write(
                ROOT / "decode_records" / method / f"case_{case_index:02}.json",
                {**old_decode, "output_path": str(video), "reused_from": str(old_decode_path)},
            )

            old_quality = OLD_ROOT / "quality" / method / f"{method}_{case_index:02}.json"
            shutil.copy2(old_quality, ROOT / "quality" / method / old_quality.name)

    shutil.copy2(__file__, ROOT / "runner_source.py")
    exp.base.write(ROOT / "protocol.json", dict(
        name=NAME,
        purpose="Extend the threshold/packed-external/fused Spark route comparison to 25 prompts",
        pipeline="native PyTorch/Diffusers",
        dataset="first 25 prompts of VBench core-five 20% subset",
        cases=all_cases,
        reused_case_indices=list(REUSED_CASES),
        generated_case_indices=list(NEW_CASES),
        reused_from=str(OLD_ROOT),
        methods=list(exp.METHODS),
        spark_configs={m: dataclasses.asdict(exp.spark_config(m)) for m in exp.METHODS},
        dense_reference_manifest=str(exp.DENSE_OUT / "generation_manifest.json"),
        dense_reference_reused=True,
        seed=exp.base.SEED,
        requested_steps=exp.base.STEPS,
        transformer_evaluations=exp.base.STEPS - 1,
        requested_frames=exp.base.FRAMES,
        internal_vae_aligned_frames=exp.base.INTERNAL_FRAMES,
        fps=exp.base.FPS,
        height=exp.base.HEIGHT,
        width=exp.base.WIDTH,
        video_tokens=37296,
        video_token_remainder=48,
        gpus=list(exp.GPUS),
        torch_compile=True,
        timing="synchronized denoise only; loading, conditioning, CPU transfer and saving excluded",
        warmup="one excluded full generation per route mode per GPU",
        pairing="all modes for a prompt run on the same GPU with rotating order",
        total_records=TOTAL,
        environment=exp.base.ENV,
    ))
    exp.base.pipeline().configure_denoise_workflow(exp.parse_args("denoise"), all_cases)
    exp.base.write(ROOT / "status.json", dict(
        status="running", stage="prepared", completed_records=30, total_records=TOTAL
    ))


def generate_worker(rank):
    import torch

    rank = int(rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = exp.base.pipeline()
    assigned = [case for case in cases() if case["index"] in NEW_CASES][rank::len(exp.GPUS)]
    workflow, states = p.configure_denoise_workflow(exp.parse_args("denoise"), assigned)
    pipe, manager, acceleration, placement = p.load_denoiser(exp.parse_args("denoise"), workflow)
    try:
        for method in exp.method_order(assigned[0]["index"], rank):
            row = exp.run_one(pipe, p, states[0], assigned[0], method, placement, warmup=True)
            exp.base.write(ROOT / "warmup" / method / f"gpu{exp.GPUS[rank]}.json", row)
            print("WARMUP", rank, method, f"{row['seconds']:.3f}s", flush=True)
        for case, state in zip(assigned, states, strict=True):
            for method in exp.method_order(case["index"], rank):
                row = exp.run_one(pipe, p, state, case, method, placement)
                print("MEASURED", rank, method, case["index"], f"{row['denoise_seconds']:.3f}s", flush=True)
        exp.base.write(ROOT / f"generate_gpu{exp.GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def decode_worker(rank):
    import numpy as np
    import torch

    rank = int(rank)
    torch.set_num_threads(4)
    if str(exp.FV_EVAL) not in sys.path:
        sys.path.insert(0, str(exp.FV_EVAL))
    from fasth3_vbench_archive import archive, file_sha

    p = exp.base.pipeline()
    assigned = [case for case in cases() if case["index"] in NEW_CASES][rank::len(exp.GPUS)]
    pipe, manager, acceleration = p.load_decoder(exp.parse_args("decode"))
    try:
        for case in assigned:
            stem = p.case_stem(case)
            for method in exp.METHODS:
                payload = torch.load(
                    OUT / method / "latents" / f"{stem}.pt",
                    map_location="cpu", weights_only=True,
                )
                with torch.inference_mode():
                    result = pipe(
                        latents=payload["latents"].to("cuda"),
                        audio_latents=payload["audio_latents"].to("cuda"),
                        output_type="np", output=["videos", "audio", "sampling_rate"],
                    )
                frames = np.clip(
                    np.round(result["videos"][0][:exp.base.FRAMES] * 255), 0, 255
                ).astype(np.uint8)
                target = OUT / method / "videos" / f"{stem}.mkv"
                metadata = archive(
                    frames, result["audio"][0], int(result["sampling_rate"]),
                    target, fps=exp.base.FPS, threads=8,
                )
                exp.base.write(
                    ROOT / "decode_records" / method / f"case_{case['index']:02}.json",
                    dict(status="complete", method=method, case=case["index"],
                         sample_id=case["sample_id"], output_path=str(target),
                         sha256=file_sha(target), archive=metadata),
                )
                print("DECODED", rank, method, case["index"], flush=True)
        exp.base.write(ROOT / f"decode_gpu{exp.GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(exp.GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(exp.base.PYTHON), str(__file__), stage, str(rank)],
            env={**os.environ, **exp.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def aggregate_runtime():
    records = {
        method: [exp.base.read(ROOT / "records" / method / f"case_{case:02}.json") for case in CASES]
        for method in exp.METHODS
    }
    summary = {}
    for method, rows in records.items():
        values = [row["denoise_seconds"] for row in rows]
        summary[method] = dict(
            mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
            stdev_seconds=statistics.stdev(values), min_seconds=min(values), max_seconds=max(values),
            mean_peak_memory_gib=statistics.mean(row["peak_memory_bytes"] / 2**30 for row in rows),
            values=values,
        )
    paired = {}
    baseline = "spark_threshold"
    for method in exp.METHODS[1:]:
        deltas = [records[method][i]["denoise_seconds"] - records[baseline][i]["denoise_seconds"] for i in range(len(CASES))]
        paired[method] = dict(
            versus=baseline, mean_delta_seconds=statistics.mean(deltas),
            median_delta_seconds=statistics.median(deltas),
            mean_ratio=summary[method]["mean_seconds"] / summary[baseline]["mean_seconds"],
            faster_prompts=sum(delta < 0 for delta in deltas), values=deltas,
        )
    exp.base.write(ROOT / "results.json", dict(status="runtime_complete", summary=summary, paired_runtime=paired))
    exp.base.write(ROOT / "status.json", dict(status="running", stage="runtime_complete", completed_records=TOTAL, total_records=TOTAL))


def quality():
    # The first ten per-case records were copied and are hash-verified/reused.
    exp.quality()
    results = exp.base.read(ROOT / "results.json")
    exp.base.write(ROOT / "status.json", dict(status="complete", stage="complete", completed_records=TOTAL, total_records=TOTAL))
    results["status"] = "complete"
    exp.base.write(ROOT / "results.json", results)


def run():
    os.environ.update(exp.base.ENV)
    prepare()
    exp.base.write(ROOT / "status.json", dict(status="running", stage="generating", completed_records=30, total_records=TOTAL))
    spawn("generate_worker")
    aggregate_runtime()
    exp.base.write(ROOT / "status.json", dict(status="running", stage="decoding", completed_records=TOTAL, total_records=TOTAL))
    spawn("decode_worker")
    exp.manifests()
    exp.base.write(ROOT / "status.json", dict(status="running", stage="quality", completed_records=TOTAL, total_records=TOTAL))
    quality()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in {"generate_worker", "decode_worker"}:
        globals()[command](int(sys.argv[2]))
    elif command in globals():
        globals()[command]()
    else:
        raise SystemExit(f"unknown command: {command}")
