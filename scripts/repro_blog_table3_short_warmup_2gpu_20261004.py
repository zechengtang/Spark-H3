#!/usr/bin/env python3
"""Two-prompt short-warmup reproduction of the blog Table 3 Spark rows."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base


NAME = "blog_table3_short_warmup_2gpu_20261004"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
CASE_INDICES = (2, 13)
GPUS = (0, 1)
RATIOS = (0.10, 0.20, 0.30)
BLOG_SECONDS = {0.10: 341.7, 0.20: 378.2, 0.30: 408.6}
FRAMES, HEIGHT, WIDTH = 240, 768, 1344
FORMAL_STEPS, SHORT_STEPS = 20, 3


def cases():
    return base.pipeline().load_cases(
        SAMPLES, list(CASE_INDICES), expected_indices=tuple(range(1, 51))
    )


def args():
    return base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--case-indices", *map(str, CASE_INDICES),
        "--steps", str(FORMAL_STEPS), "--frames", str(FRAMES),
        "--height", str(HEIGHT), "--width", str(WIDTH), "--workers", "1",
    ])


def config(steps: int, ratio: float):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        steps,
        warmup_percent=20.0 if steps == FORMAL_STEPS else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=ratio,
        sol_log_density=False,
        landmark_tree_v2_midpoint_direction_mode="legacy",
        sol_route_topk_execution="threshold",
    )


def git(*items: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(base.IMPL), *items], text=True
    ).strip()


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(
            SOURCE / name, target_is_directory=name.endswith("cache")
        )
    selected = cases()
    base.pipeline().configure_denoise_workflow(args(), selected)
    diff = subprocess.check_output(
        ["git", "-C", str(base.IMPL), "diff", "--binary"]
    )
    protocol = {
        "status": "running",
        "name": NAME,
        "purpose": "reproduce the three Spark-H3 latency rows in blog Table 3",
        "cases": selected,
        "gpus": list(GPUS),
        "ratios": list(RATIOS),
        "blog_seconds": {f"{round(r * 100)}pct": v for r, v in BLOG_SECONDS.items()},
        "formal": {"steps": FORMAL_STEPS, "evaluations": 19, "samples_per_ratio": 2},
        "short_warmup": {
            "per_ratio_per_gpu": True,
            "steps": SHORT_STEPS,
            "evaluations": 2,
            "dense": 1,
            "sparse": 1,
        },
        "ratio_order": {"gpu0": [0.10, 0.20, 0.30], "gpu1": [0.30, 0.20, 0.10]},
        "frames": FRAMES,
        "resolution": [WIDTH, HEIGHT],
        "seed": 42,
        "configs": {
            f"{round(ratio * 100)}pct": dataclasses.asdict(config(FORMAL_STEPS, ratio))
            for ratio in RATIOS
        },
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": base.sha(Path(__file__).resolve()),
        "timing": "CUDA-synchronized denoise only; short warmup excluded",
    }
    base.write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def run_once(pipe, pipeline, original_state, *, steps: int, ratio: float, seed: int):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    with install_h3_sparse_attention(pipe.transformer, config(steps, ratio)) as plugin, torch.inference_mode():
        plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(
            state=state,
            num_frames=FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary()
    expected_evaluations = steps - 1
    assert summary["completed_evaluations"] == expected_evaluations, summary
    assert summary["dense_evaluations"] == (4 if steps == FORMAL_STEPS else 1), summary
    assert all(
        torch.isfinite(output[name]).all().item()
        for name in ("latents", "audio_latents")
    )
    del output, state
    return {
        "seconds": seconds,
        "ratio": ratio,
        "steps": steps,
        "evaluations": expected_evaluations,
        "attention_summary": summary,
    }


def worker(rank: int):
    import torch

    rank = int(rank)
    gpu = GPUS[rank]
    torch.set_num_threads(4)
    pipeline = base.pipeline()
    selected = cases()
    workflow, states = pipeline.configure_denoise_workflow(args(), selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args(), workflow)
    ratio_order = RATIOS if rank == 0 else tuple(reversed(RATIOS))
    try:
        for order, ratio in enumerate(ratio_order):
            label = f"{round(ratio * 100):02d}pct"
            warm = run_once(
                pipe, pipeline, states[rank], steps=SHORT_STEPS,
                ratio=ratio, seed=41,
            )
            base.write(ROOT / "records" / f"gpu{gpu}_{label}_warmup.json", {
                "phase": "discarded_short_warmup", "gpu": gpu, "order": order,
                "case": selected[rank]["index"], "placement": placement, **warm,
            })
            print("WARMUP", label, round(warm["seconds"], 3), flush=True)
            row = run_once(
                pipe, pipeline, states[rank], steps=FORMAL_STEPS,
                ratio=ratio, seed=42,
            )
            base.write(ROOT / "records" / f"gpu{gpu}_{label}_measured.json", {
                "phase": "measured", "gpu": gpu, "order": order,
                "case": selected[rank]["index"],
                "sample_id": selected[rank]["sample_id"],
                "prompt_sha256": selected[rank]["prompt_sha256"],
                "placement": placement, **row,
            })
            print("MEASURED", label, round(row["seconds"], 3), flush=True)
        base.write(ROOT / f"worker_gpu{gpu}.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


def summarize():
    rows = [
        json.loads(path.read_text())
        for path in (ROOT / "records").glob("*_measured.json")
    ]
    result = {"status": "complete", "ratios": {}}
    for ratio in RATIOS:
        label = f"{round(ratio * 100):02d}pct"
        chosen = sorted(
            (row for row in rows if row["ratio"] == ratio), key=lambda row: row["gpu"]
        )
        if len(chosen) != len(GPUS):
            raise RuntimeError(f"missing {label}: {len(chosen)}")
        values = [row["seconds"] for row in chosen]
        mean = statistics.mean(values)
        target = BLOG_SECONDS[ratio]
        result["ratios"][label] = {
            "seconds": values,
            "mean_seconds": mean,
            "blog_seconds": target,
            "delta_seconds": mean - target,
            "delta_percent": (mean / target - 1.0) * 100.0,
            "within_2_percent": abs(mean / target - 1.0) <= 0.02,
        }
    result["all_within_2_percent"] = all(
        row["within_2_percent"] for row in result["ratios"].values()
    )
    base.write(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete"
    base.write(ROOT / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


def launch():
    prepare()
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen(
            [str(base.PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, proc, log))
    codes = [(gpu, proc.wait()) for gpu, proc, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"workers failed: {codes}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(int(sys.argv[2]))
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
