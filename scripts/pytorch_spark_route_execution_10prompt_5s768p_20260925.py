#!/usr/bin/env python3
"""10-prompt native-PyTorch comparison of Spark route-score execution modes."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base


NAME = "pytorch_spark_route_execution_10prompt_5s768p_seed42_20260925"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
DENSE_OUT = Path("/autodl-fs/data/h3_outputs/pytorch_spark_tail_ablation_50prompt_5s768p_seed42_20260925/dense")
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
CASES = tuple(range(1, 11))
METHODS = ("spark_threshold", "spark_external_no_qk", "spark_fused")
EXECUTIONS = {
    "spark_threshold": "threshold",
    "spark_external_no_qk": "packed_external_no_route_qk",
    "spark_fused": "fused",
}
GPUS = (2, 3)
QUALITY_SCRIPT = base.SCRIPTS / "h3_quality_video_pair.py"
FV_EVAL = base.IMPL.parent / "FastVideo/examples/inference/eval"


def cases():
    return base.pipeline().load_cases(SAMPLES, list(CASES), expected_indices=tuple(range(1, 51)))


def parse_args(command):
    return base.pipeline().build_parser().parse_args([
        command,
        "--samples", str(SAMPLES),
        "--output", str(OUT),
        "--method", "dense",
        "--steps", str(base.STEPS),
        "--frames", str(base.FRAMES),
        "--height", str(base.HEIGHT),
        "--width", str(base.WIDTH),
        "--workers", "1",
    ])


def spark_config(method):
    base.pipeline()
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        base.STEPS,
        landmark_tree_v2_fanout=64,
        landmark_tree_v2_landmark_count=64,
        sol_log_density=False,
        sol_video_tail_mode="dense",
        sol_route_topk_execution=EXECUTIONS[method],
    )


def method_order(case_index, rank):
    offset = (case_index - 1 + rank) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    all_cases = cases()
    for method in METHODS:
        (ROOT / "records" / method).mkdir(parents=True)
        (ROOT / "warmup" / method).mkdir(parents=True)
        (ROOT / "decode_records" / method).mkdir(parents=True)
        (OUT / method / "latents").mkdir(parents=True)
        (OUT / method / "videos").mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(base.CONDITIONING_SOURCE / name)
    shutil.copy2(__file__, ROOT / "runner_source.py")
    base.write(ROOT / "protocol.json", dict(
        name=NAME,
        purpose="Compare threshold, packed external-no-QK, and CTA-local fused Spark route execution",
        pipeline="native PyTorch/Diffusers",
        dataset="first 10 prompts of VBench core-five 20% subset",
        cases=all_cases,
        methods=list(METHODS),
        spark_configs={m: dataclasses.asdict(spark_config(m)) for m in METHODS},
        dense_reference_manifest=str(DENSE_OUT / "generation_manifest.json"),
        dense_reference_reused=True,
        seed=base.SEED,
        requested_steps=base.STEPS,
        transformer_evaluations=base.STEPS - 1,
        requested_frames=base.FRAMES,
        internal_vae_aligned_frames=base.INTERNAL_FRAMES,
        fps=base.FPS,
        height=base.HEIGHT,
        width=base.WIDTH,
        video_tokens=37296,
        video_token_remainder=48,
        gpus=list(GPUS),
        torch_compile=True,
        timing="synchronized denoise only; loading, conditioning, CPU transfer and saving excluded",
        warmup="one excluded full generation per route mode per GPU",
        pairing="all modes for a prompt run on the same GPU with rotating order",
        environment=base.ENV,
    ))
    base.pipeline().configure_denoise_workflow(parse_args("denoise"), all_cases)
    base.write(ROOT / "status.json", dict(
        status="running", stage="prepared", completed_records=0, total_records=len(CASES) * len(METHODS)
    ))


def run_one(pipe, p, original_state, case, method, placement, *, warmup=False):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    state = p.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    config = spark_config(method)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with install_h3_sparse_attention(pipe.transformer, config) as plugin, torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=state,
            num_frames=base.FRAMES,
            height=base.HEIGHT,
            width=base.WIDTH,
            num_inference_steps=base.STEPS,
            generator=torch.Generator(device="cpu").manual_seed(base.SEED),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary()
    calls = summary["processor_calls"]
    assert summary["completed_evaluations"] == 19, summary
    assert calls.get("sparse:sol") == 735, calls
    if warmup:
        del result, state
        return dict(method=method, seconds=seconds)
    payload = {key: result[key].detach().cpu().contiguous() for key in ("latents", "audio_latents")}
    assert all(torch.isfinite(value).all() for value in payload.values())
    latent = OUT / method / "latents" / f"{p.case_stem(case)}.pt"
    p.atomic_torch_save(payload, latent)
    latent_sha256 = base.sha(latent)
    base.write(
        OUT / method / "denoise_records" / f"{p.case_stem(case)}.json",
        dict(
            method=method,
            sample_id=case["sample_id"],
            prompt_sha256=case["prompt_sha256"],
            latent_path=str(latent),
            latent_sha256=latent_sha256,
            torch_compile=True,
            compile_wrapped_blocks=p.EXPECTED_TRANSFORMER_BLOCKS,
        ),
    )
    record = dict(
        method=method,
        route_execution=EXECUTIONS[method],
        case=case["index"],
        sample_id=case["sample_id"],
        prompt_sha256=case["prompt_sha256"],
        config=dataclasses.asdict(config),
        attention_summary=summary,
        latent_path=str(latent),
        latent_sha256=latent_sha256,
        seed=base.SEED,
        steps=base.STEPS,
        # The Benchmark parser now requires compilation and no longer exposes
        # a no-compile flag. load_denoiser verifies the live compile hook.
        torch_compile=True,
        frames=base.FRAMES,
        height=base.HEIGHT,
        width=base.WIDTH,
        denoise_seconds=seconds,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        placement=placement,
    )
    base.write(ROOT / "records" / method / f"case_{case['index']:02}.json", record)
    del result, payload, state
    return record


def generate_worker(rank):
    import torch

    rank = int(rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = base.pipeline()
    assigned = cases()[rank::len(GPUS)]
    workflow, states = p.configure_denoise_workflow(parse_args("denoise"), assigned)
    pipe, manager, acceleration, placement = p.load_denoiser(parse_args("denoise"), workflow)
    try:
        for method in method_order(assigned[0]["index"], rank):
            row = run_one(pipe, p, states[0], assigned[0], method, placement, warmup=True)
            base.write(ROOT / "warmup" / method / f"gpu{GPUS[rank]}.json", row)
            print("WARMUP", rank, method, f"{row['seconds']:.3f}s", flush=True)
        for case, state in zip(assigned, states, strict=True):
            for method in method_order(case["index"], rank):
                row = run_one(pipe, p, state, case, method, placement)
                print("MEASURED", rank, method, case["index"], f"{row['denoise_seconds']:.3f}s", flush=True)
        base.write(ROOT / f"generate_gpu{GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(base.PYTHON), str(__file__), stage, str(rank)],
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def aggregate_runtime():
    records = {
        method: [base.read(ROOT / "records" / method / f"case_{case:02}.json") for case in CASES]
        for method in METHODS
    }
    summary = {}
    for method, rows in records.items():
        values = [row["denoise_seconds"] for row in rows]
        summary[method] = dict(
            mean_seconds=statistics.mean(values),
            median_seconds=statistics.median(values),
            stdev_seconds=statistics.stdev(values),
            min_seconds=min(values),
            max_seconds=max(values),
            mean_peak_memory_gib=statistics.mean(row["peak_memory_bytes"] / 2**30 for row in rows),
            values=values,
        )
    paired = {}
    baseline = "spark_threshold"
    for method in METHODS[1:]:
        deltas = [records[method][i]["denoise_seconds"] - records[baseline][i]["denoise_seconds"] for i in range(len(CASES))]
        paired[method] = dict(
            versus=baseline,
            mean_delta_seconds=statistics.mean(deltas),
            median_delta_seconds=statistics.median(deltas),
            mean_ratio=summary[method]["mean_seconds"] / summary[baseline]["mean_seconds"],
            faster_prompts=sum(delta < 0 for delta in deltas),
            values=deltas,
        )
    base.write(ROOT / "results.json", dict(status="runtime_complete", summary=summary, paired_runtime=paired))
    base.write(ROOT / "status.json", dict(status="running", stage="runtime_complete", completed_records=30, total_records=30))


def decode_worker(rank):
    import numpy as np
    import torch

    rank = int(rank)
    torch.set_num_threads(4)
    if str(FV_EVAL) not in sys.path:
        sys.path.insert(0, str(FV_EVAL))
    from fasth3_vbench_archive import archive, file_sha

    p = base.pipeline()
    assigned = cases()[rank::len(GPUS)]
    pipe, manager, acceleration = p.load_decoder(parse_args("decode"))
    try:
        for case in assigned:
            stem = p.case_stem(case)
            for method in METHODS:
                payload = __import__("torch").load(OUT / method / "latents" / f"{stem}.pt", map_location="cpu", weights_only=True)
                with torch.inference_mode():
                    result = pipe(
                        latents=payload["latents"].to("cuda"),
                        audio_latents=payload["audio_latents"].to("cuda"),
                        output_type="np",
                        output=["videos", "audio", "sampling_rate"],
                    )
                frames = np.clip(np.round(result["videos"][0][:base.FRAMES] * 255), 0, 255).astype(np.uint8)
                target = OUT / method / "videos" / f"{stem}.mkv"
                metadata = archive(frames, result["audio"][0], int(result["sampling_rate"]), target, fps=base.FPS, threads=8)
                base.write(ROOT / "decode_records" / method / f"case_{case['index']:02}.json", dict(
                    status="complete", method=method, case=case["index"], sample_id=case["sample_id"],
                    output_path=str(target), sha256=file_sha(target), archive=metadata,
                ))
                print("DECODED", rank, method, case["index"], flush=True)
        base.write(ROOT / f"decode_gpu{GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def manifests():
    all_cases = cases()
    for method in METHODS:
        rows = []
        for case in all_cases:
            rec = base.read(ROOT / "decode_records" / method / f"case_{case['index']:02}.json")
            rows.append(dict(
                index=case["index"], sample_id=case["sample_id"], prompt_sha256=case["prompt_sha256"],
                evaluation_prompt=case["original_prompt"], vbench_dimensions=case["vbench_dimensions"],
                output_path=rec["output_path"], sha256=rec["sha256"],
                video=dict(frames=base.FRAMES, width=base.WIDTH, height=base.HEIGHT, fps=float(base.FPS)),
            ))
        base.write(OUT / method / "generation_manifest.json", dict(
            schema_version=1, status="passed", method=method, sample_count=len(CASES),
            settings=dict(seed=base.SEED, steps=base.STEPS, frames=base.FRAMES, height=base.HEIGHT, width=base.WIDTH, fps=base.FPS),
            records=rows,
        ))


def quality():
    summaries = {}
    for method in METHODS:
        work = ROOT / "quality" / method
        config = dict(
            work_dir=str(work), reference_manifest=str(DENSE_OUT / "generation_manifest.json"),
            candidate_manifest=str(OUT / method / "generation_manifest.json"),
            method=method, cases=list(CASES), workers=len(GPUS),
        )
        config_path = ROOT / "quality" / f"{method}_config.json"
        base.write(config_path, config)
        subprocess.run(
            [str(base.PYTHON), str(QUALITY_SCRIPT), "run", "--config", str(config_path)],
            check=True,
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": "2,3", "H3_NUM_GPUS": "2"},
        )
        summaries[method] = base.read(work / f"{method}_quality_results.json")["summary"]
    results = base.read(ROOT / "results.json")
    results.update(status="complete", quality=summaries)
    base.write(ROOT / "results.json", results)
    base.write(ROOT / "quality_results.json", dict(status="complete", reference="dense", cases=list(CASES), summaries=summaries))
    base.write(ROOT / "status.json", dict(status="complete", stage="complete", completed_records=30, total_records=30))


def run():
    os.environ.update(base.ENV)
    prepare()
    base.write(ROOT / "status.json", dict(status="running", stage="generating", completed_records=0, total_records=30))
    spawn("generate_worker")
    aggregate_runtime()
    base.write(ROOT / "status.json", dict(status="running", stage="decoding", completed_records=30, total_records=30))
    spawn("decode_worker")
    manifests()
    base.write(ROOT / "status.json", dict(status="running", stage="quality", completed_records=30, total_records=30))
    quality()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in {"generate_worker", "decode_worker"}:
        globals()[command](int(sys.argv[2]))
    elif command in globals():
        globals()[command]()
    else:
        raise SystemExit(f"unknown command: {command}")
