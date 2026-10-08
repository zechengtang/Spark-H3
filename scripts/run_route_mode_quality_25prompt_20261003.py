#!/usr/bin/env python3
"""25-prompt 5s/10s quality validation of the current Spark route modes."""
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


NAME = "route_mode_quality_25prompt_20261003"
EXPERIMENTS = Path("/autodl-fs/data/h3_experiments")
OUTPUTS = Path("/autodl-fs/data/h3_outputs")
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
OLD_10S = Path("/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920")
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
QUALITY_SCRIPT = base.SCRIPTS / "h3_quality_video_pair.py"
FV_EVAL = base.IMPL.parent / "FastVideo/examples/inference/eval"
CASES = tuple(range(1, 26))
GPUS = tuple(range(8))
ALL_METHODS = (
    "dense",
    "legacy_threshold",
    "fused_threshold",
    "fused_packed_external_no_route_qk",
)
GENERATED = {
    5: ALL_METHODS,
    10: ALL_METHODS[2:],
}


def roots(duration):
    suffix = f"{NAME}_{duration}s768p"
    return EXPERIMENTS / suffix, OUTPUTS / suffix


def cases():
    return base.pipeline().load_cases(SAMPLES, list(CASES), expected_indices=tuple(range(1, 51)))


def frames(duration):
    return duration * 24


def parse_args(duration, command="denoise"):
    _, out = roots(duration)
    return base.pipeline().build_parser().parse_args([
        command, "--samples", str(SAMPLES), "--output", str(out), "--method", "dense",
        "--case-indices", *map(str, CASES), "--steps", "20", "--frames", str(frames(duration)),
        "--height", "768", "--width", "1344", "--workers", "1",
    ])


def config(method, steps=20):
    base.pipeline()
    from h3_sparse_attention import H3SparseAttentionConfig

    if method == "dense":
        return None
    kwargs = dict(
        warmup_percent=20.0 if steps == 20 else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=0.1,
        sol_log_density=False,
    )
    if method != "legacy_threshold":
        kwargs["landmark_tree_v2_midpoint_direction_mode"] = "fused"
    kwargs["sol_route_topk_execution"] = {
        "legacy_threshold": "threshold",
        "fused_threshold": "threshold",
        "fused_packed_external_no_route_qk": "packed_external_no_route_qk",
    }[method]
    return H3SparseAttentionConfig.spark(steps, **kwargs)


def method_order(case_index, rank, methods):
    offset = (case_index - 1 + rank) % len(methods)
    return methods[offset:] + methods[:offset]


def prepare(duration):
    root, out = roots(duration)
    if root.exists() or out.exists():
        raise FileExistsError(f"refusing to overwrite {root} or {out}")
    root.mkdir(parents=True)
    out.mkdir(parents=True)
    selected = cases()
    for method in GENERATED[duration]:
        for path in (
            root / "records" / method,
            root / "warmup" / method,
            root / "decode_records" / method,
            root / "quality" / method,
            out / method / "latents",
            out / method / "videos",
        ):
            path.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        source = SOURCE / name
        (out / name).symlink_to(source, target_is_directory=source.is_dir())
    shutil.copy2(__file__, root / "runner_source.py")
    base.write(root / "protocol.json", dict(
        status="running", name=NAME, duration_seconds=duration,
        purpose="Orthogonal legacy-vs-fused and threshold-vs-packed route quality validation",
        cases=selected, case_indices=list(CASES), generated_methods=list(GENERATED[duration]),
        reused_10s_methods=list(ALL_METHODS[:2]) if duration == 10 else [],
        reused_10s_dense_manifest=str(SOURCE / "dense/generation_manifest.json") if duration == 10 else None,
        reused_10s_legacy_metrics=str(OLD_10S / "quality_work/quality") if duration == 10 else None,
        configs={method: dataclasses.asdict(config(method)) for method in GENERATED[duration] if method != "dense"},
        seed=42, requested_steps=20, transformer_evaluations=19,
        requested_frames=frames(duration), fps=24, height=768, width=1344,
        gpus=list(GPUS), torch_compile=True,
        timing="CUDA-synchronized denoise only; loading, warmup, transfer, save, decode and quality excluded",
        warmup="one excluded 3-step generation per generated method and GPU",
        pairing="all generated methods for a prompt run on the same GPU with rotating order",
        runner_sha256=base.sha(Path(__file__).resolve()), environment=base.ENV,
    ))
    base.pipeline().configure_denoise_workflow(parse_args(duration), selected)
    base.write(root / "status.json", dict(status="running", stage="prepared", completed_records=0))


def run_one(pipe, pipeline, original_state, case, duration, method, placement, *, warmup=False):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    root, out = roots(duration)
    steps = 3 if warmup else 20
    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
    cfg = config(method, steps)
    plugin_context = contextlib.nullcontext(None) if cfg is None else install_h3_sparse_attention(pipe.transformer, cfg)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with plugin_context as plugin, torch.inference_mode():
        if plugin is not None:
            plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=state, num_frames=frames(duration), height=768, width=1344,
            num_inference_steps=steps, generator=torch.Generator(device="cpu").manual_seed(42),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = None if plugin is None else plugin.summary()
    if summary is not None:
        assert summary["completed_evaluations"] == steps - 1, summary
        assert summary["dense_evaluations"] == (4 if steps == 20 else 1), summary
    if warmup:
        del result, state
        return dict(method=method, seconds=seconds, attention_summary=summary)
    payload = {key: result[key].detach().cpu().contiguous() for key in ("latents", "audio_latents")}
    assert all(torch.isfinite(value).all().item() for value in payload.values())
    latent = out / method / "latents" / f"{pipeline.case_stem(case)}.pt"
    pipeline.atomic_torch_save(payload, latent)
    # Keep the Benchmark's standard compile-provenance guard compatible with
    # this custom runner so an interrupted experiment can resume safely.
    base.write(
        latent.parent.parent / "denoise_records" / f"{latent.stem}.json",
        {
            "torch_compile": True,
            "compile_wrapped_blocks": pipeline.EXPECTED_TRANSFORMER_BLOCKS,
            "latent_sha256": base.sha(latent),
        },
    )
    record = dict(
        method=method, case=case["index"], sample_id=case["sample_id"],
        prompt_sha256=case["prompt_sha256"], config=None if cfg is None else dataclasses.asdict(cfg),
        attention_summary=summary, latent_path=str(latent), latent_sha256=base.sha(latent),
        seed=42, steps=20, frames=frames(duration), height=768, width=1344,
        denoise_seconds=seconds, peak_memory_bytes=torch.cuda.max_memory_allocated(), placement=placement,
    )
    base.write(root / "records" / method / f"case_{case['index']:02}.json", record)
    del result, payload, state
    return record


def generate_worker(duration, rank):
    import torch

    root, _ = roots(duration)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = base.pipeline()
    assigned = cases()[rank::len(GPUS)]
    workflow, states = pipeline.configure_denoise_workflow(parse_args(duration), assigned)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(parse_args(duration), workflow)
    methods = GENERATED[duration]
    try:
        for method in method_order(assigned[0]["index"], rank, methods):
            row = run_one(pipe, pipeline, states[0], assigned[0], duration, method, placement, warmup=True)
            base.write(root / "warmup" / method / f"gpu{GPUS[rank]}.json", row)
            print("WARMUP", duration, rank, method, f"{row['seconds']:.3f}s", flush=True)
        for case, state in zip(assigned, states, strict=True):
            for method in method_order(case["index"], rank, methods):
                record_path = root / "records" / method / f"case_{case['index']:02}.json"
                if record_path.exists():
                    previous = base.read(record_path)
                    latent_path = Path(previous["latent_path"])
                    if (
                        previous.get("method") == method
                        and previous.get("case") == case["index"]
                        and latent_path.exists()
                        and base.sha(latent_path) == previous.get("latent_sha256")
                    ):
                        print("REUSED", duration, rank, method, case["index"], flush=True)
                        continue
                row = run_one(pipe, pipeline, state, case, duration, method, placement)
                print("MEASURED", duration, rank, method, case["index"], f"{row['denoise_seconds']:.3f}s", flush=True)
        base.write(root / f"generate_gpu{GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


def decode_worker(duration, rank):
    import numpy as np
    import torch

    root, out = roots(duration)
    torch.set_num_threads(4)
    if str(FV_EVAL) not in sys.path:
        sys.path.insert(0, str(FV_EVAL))
    from fasth3_vbench_archive import archive, file_sha

    pipeline = base.pipeline()
    assigned = cases()[rank::len(GPUS)]
    pipe, manager, acceleration = pipeline.load_decoder(parse_args(duration, "decode"))
    try:
        for case in assigned:
            stem = pipeline.case_stem(case)
            for method in GENERATED[duration]:
                record_path = root / "decode_records" / method / f"case_{case['index']:02}.json"
                if record_path.exists():
                    previous = base.read(record_path)
                    target = Path(previous["output_path"])
                    if (
                        previous.get("method") == method
                        and previous.get("case") == case["index"]
                        and target.exists()
                        and file_sha(target) == previous.get("sha256")
                    ):
                        print("REUSED DECODE", duration, rank, method, case["index"], flush=True)
                        continue
                payload = torch.load(out / method / "latents" / f"{stem}.pt", map_location="cpu", weights_only=True)
                with torch.inference_mode():
                    result = pipe(
                        latents=payload["latents"].cuda(), audio_latents=payload["audio_latents"].cuda(),
                        output_type="np", output=["videos", "audio", "sampling_rate"],
                    )
                rgb = np.clip(np.round(result["videos"][0][:frames(duration)] * 255), 0, 255).astype(np.uint8)
                target = out / method / "videos" / f"{stem}.mkv"
                metadata = archive(rgb, result["audio"][0], int(result["sampling_rate"]), target, fps=24, threads=8)
                base.write(root / "decode_records" / method / f"case_{case['index']:02}.json", dict(
                    status="complete", method=method, case=case["index"], sample_id=case["sample_id"],
                    output_path=str(target), sha256=file_sha(target), archive=metadata,
                ))
                print("DECODED", duration, rank, method, case["index"], flush=True)
        base.write(root / f"decode_gpu{GPUS[rank]}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


def spawn(duration, stage):
    root, _ = roots(duration)
    load_slots = int(os.environ.get("H3_WEIGHT_LOAD_LOCK_SLOTS", "4"))
    if load_slots < 1:
        raise ValueError("H3_WEIGHT_LOAD_LOCK_SLOTS must be positive")
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (root / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(base.PYTHON), str(Path(__file__).resolve()), stage, str(duration), str(rank)],
            env={
                **os.environ,
                **base.ENV,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                # Four concurrent 135-GiB transient loads fit safely in this
                # 1-TiB host and avoid serializing all eight GPU transfers.
                "H3_WEIGHT_LOAD_LOCK_SLOT": str(rank % load_slots),
            },
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def manifests(duration):
    root, out = roots(duration)
    selected = cases()
    for method in GENERATED[duration]:
        rows = []
        for case in selected:
            rec = base.read(root / "decode_records" / method / f"case_{case['index']:02}.json")
            rows.append(dict(
                index=case["index"], sample_id=case["sample_id"], prompt_sha256=case["prompt_sha256"],
                evaluation_prompt=case["original_prompt"], vbench_dimensions=case["vbench_dimensions"],
                output_path=rec["output_path"], sha256=rec["sha256"],
                video=dict(frames=frames(duration), width=1344, height=768, fps=24.0),
            ))
        base.write(out / method / "generation_manifest.json", dict(
            schema_version=1, status="passed", method=method, sample_count=len(CASES),
            settings=dict(seed=42, steps=20, frames=frames(duration), height=768, width=1344, fps=24),
            records=rows,
        ))


def aggregate_runtime(duration):
    root, _ = roots(duration)
    result = {"status": "runtime_complete", "duration_seconds": duration, "runtime": {}}
    for method in GENERATED[duration]:
        rows = [base.read(root / "records" / method / f"case_{case:02}.json") for case in CASES]
        values = [row["denoise_seconds"] for row in rows]
        result["runtime"][method] = dict(
            mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
            stdev_seconds=statistics.stdev(values), min_seconds=min(values), max_seconds=max(values), values=values,
        )
    base.write(root / "results.json", result)


def quality(duration):
    root, out = roots(duration)
    reference_manifest = (
        out / "dense/generation_manifest.json" if duration == 5
        else SOURCE / "dense/generation_manifest.json"
    )
    summaries = {}
    for method in GENERATED[duration]:
        if method == "dense":
            continue
        work = root / "quality" / method
        cfg = dict(
            work_dir=str(work), reference_manifest=str(reference_manifest),
            candidate_manifest=str(out / method / "generation_manifest.json"),
            method=method, cases=list(CASES), workers=len(GPUS),
        )
        config_path = root / "quality" / f"{method}_config.json"
        base.write(config_path, cfg)
        subprocess.run(
            [str(base.PYTHON), str(QUALITY_SCRIPT), "run", "--config", str(config_path)], check=True,
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": ",".join(map(str, GPUS)), "H3_NUM_GPUS": str(len(GPUS))},
        )
        summaries[method] = base.read(work / f"{method}_quality_results.json")["summary"]
    if duration == 10:
        old = []
        for case in CASES:
            row = base.read(OLD_10S / "quality_work/quality" / f"topk10_reblock_global_reweight_{case:02}.json")
            old.append(row)
        summaries["legacy_threshold"] = {
            "reused_from": str(OLD_10S), "count": len(old),
            "psnr_db_mean": statistics.mean(row["psnr_db"] for row in old),
            "ssim_mean": statistics.mean(row["ssim"] for row in old),
            "lpips_mean": statistics.mean(row["lpips"] for row in old),
        }
    result = base.read(root / "results.json")
    result.update(status="complete", quality=summaries)
    base.write(root / "results.json", result)
    protocol = base.read(root / "protocol.json")
    protocol["status"] = "complete"
    base.write(root / "protocol.json", protocol)
    base.write(root / "status.json", dict(status="complete", stage="complete"))


def run(duration):
    os.environ.update(base.ENV)
    root, _ = roots(duration)
    prepare(duration)
    base.write(root / "status.json", dict(status="running", stage="generating"))
    spawn(duration, "generate_worker")
    aggregate_runtime(duration)
    base.write(root / "status.json", dict(status="running", stage="decoding"))
    spawn(duration, "decode_worker")
    manifests(duration)
    base.write(root / "status.json", dict(status="running", stage="quality"))
    quality(duration)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run(int(sys.argv[2]))
    elif command in {"generate_worker", "decode_worker"}:
        globals()[command](int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command: {command}")
