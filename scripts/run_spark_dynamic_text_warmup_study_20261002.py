#!/usr/bin/env python3
"""Four-GPU warmup study for Spark's dynamic text-length compiled cache."""
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


NAME = "spark_dynamic_text_warmup_study_20261002"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
CASE_INDICES = (2, 13)
CONDITIONS = ("full_same", "full_cross", "short_same", "short_cross")
GPUS = (0, 1, 2, 3)
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


def config(steps):
    from h3_sparse_attention import H3SparseAttentionConfig
    return H3SparseAttentionConfig.spark(
        steps,
        warmup_percent=20.0 if steps == FORMAL_STEPS else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=0.10,
        sol_log_density=False,
        landmark_tree_v2_midpoint_direction_mode="legacy",
        sol_route_topk_execution="threshold",
    )


def git(*items):
    return subprocess.check_output(["git", "-C", str(base.IMPL), *items], text=True).strip()


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(SOURCE / name, target_is_directory=name.endswith("cache"))
    selected = cases()
    base.pipeline().configure_denoise_workflow(args(), selected)
    diff = subprocess.check_output(["git", "-C", str(base.IMPL), "diff", "--binary"])
    protocol = dict(
        status="running", name=NAME, cases=selected,
        conditions=list(CONDITIONS), gpus=list(GPUS),
        formal=dict(steps=FORMAL_STEPS, evaluations=19, repeats=2),
        short_warmup=dict(steps=SHORT_STEPS, evaluations=2, dense=1, sparse=1),
        full_warmup=dict(steps=FORMAL_STEPS, evaluations=19),
        frames=FRAMES, resolution=[WIDTH, HEIGHT], seed=42,
        configs={str(n): dataclasses.asdict(config(n)) for n in (SHORT_STEPS, FORMAL_STEPS)},
        git_revision=git("rev-parse", "HEAD"),
        git_diff_sha256=hashlib.sha256(diff).hexdigest(),
        runner_sha256=base.sha(Path(__file__).resolve()),
        timing="CUDA-synchronized denoise only; discarded warmup excluded",
        pass_rule="short mean differs from corresponding full mean by <=2.0 seconds",
    )
    base.write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def run_once(pipe, p, original_state, *, steps, seed):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention
    from h3_sparse_attention.sol_numerator_virtual_q import fused_compile_cache_stats

    state = p.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    cfg = config(steps)
    before = fused_compile_cache_stats()
    with install_h3_sparse_attention(pipe.transformer, cfg) as plugin, torch.inference_mode():
        plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(
            state=state, num_frames=FRAMES, height=HEIGHT, width=WIDTH,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary()
    expected_evals = steps - 1
    assert summary["completed_evaluations"] == expected_evals, summary
    assert summary["dense_evaluations"] == (4 if steps == FORMAL_STEPS else 1), summary
    assert all(torch.isfinite(output[name]).all().item() for name in ("latents", "audio_latents"))
    after = fused_compile_cache_stats()
    del output, state
    return dict(
        seconds=seconds, steps=steps, evaluations=expected_evals,
        attention_summary=summary, compile_before=before, compile_after=after,
        new_compile_calls=after["compile_calls"] - before["compile_calls"],
        new_compile_seconds=after["compile_seconds"] - before["compile_seconds"],
    )


def worker(rank):
    import torch
    import h3_sparse_attention.sol_numerator_virtual_q as fused

    rank = int(rank)
    torch.set_num_threads(4)
    p = base.pipeline()
    selected = cases()
    workflow, states = p.configure_denoise_workflow(args(), selected)
    pipe, manager, acceleration, placement = p.load_denoiser(args(), workflow)
    condition = CONDITIONS[rank]
    target_index = 0 if condition.endswith("same") else 1
    warm_steps = SHORT_STEPS if condition.startswith("short") else FORMAL_STEPS
    try:
        fused._FUSED_COMPILED.clear()
        fused._FUSED_COMPILE_CALLS = 0
        fused._FUSED_COMPILE_SECONDS = 0.0
        warm = run_once(pipe, p, states[0], steps=warm_steps, seed=41)
        base.write(ROOT / "records" / f"gpu{GPUS[rank]}_{condition}_warmup.json", {
            "phase": "discarded_warmup", "condition": condition,
            "gpu": GPUS[rank], "case": selected[0]["index"], "placement": placement, **warm,
        })
        print("WARMUP", condition, round(warm["seconds"], 3), flush=True)
        for repeat, seed in ((1, 42), (2, 43)):
            row = run_once(pipe, p, states[target_index], steps=FORMAL_STEPS, seed=seed)
            base.write(ROOT / "records" / f"gpu{GPUS[rank]}_{condition}_measured{repeat}.json", {
                "phase": "measured", "repeat": repeat, "condition": condition,
                "gpu": GPUS[rank], "case": selected[target_index]["index"],
                "prompt_sha256": selected[target_index]["prompt_sha256"],
                "placement": placement, **row,
            })
            print("MEASURED", condition, repeat, round(row["seconds"], 3),
                  "new_compiles", row["new_compile_calls"], flush=True)
        base.write(ROOT / f"worker_gpu{GPUS[rank]}.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def summarize():
    rows = [json.loads(path.read_text()) for path in (ROOT / "records").glob("*measured*.json")]
    summary = {}
    for condition in CONDITIONS:
        values = [row["seconds"] for row in rows if row["condition"] == condition]
        if len(values) != 2:
            raise RuntimeError(f"missing {condition}: {values}")
        compile_calls = [row["new_compile_calls"] for row in rows if row["condition"] == condition]
        summary[condition] = dict(values=values, mean_seconds=statistics.mean(values),
                                  new_compile_calls=compile_calls)
    deltas = {
        "same": abs(summary["short_same"]["mean_seconds"] - summary["full_same"]["mean_seconds"]),
        "cross": abs(summary["short_cross"]["mean_seconds"] - summary["full_cross"]["mean_seconds"]),
    }
    passed = max(deltas.values()) <= 2.0 and all(
        calls == [0, 0] for calls in (summary["short_same"]["new_compile_calls"],
                                     summary["short_cross"]["new_compile_calls"])
    )
    result = dict(status="complete", summary=summary, deltas_seconds=deltas,
                  short_warmup_accepted=passed,
                  selected_protocol="one_dense_one_sparse" if passed else "kernel_performance_matrix")
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
            stdout=log, stderr=subprocess.STDOUT,
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
        worker(sys.argv[2])
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
