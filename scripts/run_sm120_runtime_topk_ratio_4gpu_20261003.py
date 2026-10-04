#!/usr/bin/env python3
"""Four-GPU A/B for making the SM120 fused Top-K ratio runtime-dynamic."""
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


NAME = "sm120_runtime_topk_ratio_4gpu_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
CASE_INDICES = (2, 13, 31, 33)
GPUS = (0, 1, 2, 3)
RATIOS = (0.10, 0.20, 0.30)
MODES = ("static", "runtime")
FRAMES, HEIGHT, WIDTH, STEPS = 120, 768, 1344, 20


def cases():
    return base.pipeline().load_cases(
        SAMPLES, list(CASE_INDICES), expected_indices=tuple(range(1, 51))
    )


def args():
    return base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--case-indices", *map(str, CASE_INDICES),
        "--steps", str(STEPS), "--frames", str(FRAMES),
        "--height", str(HEIGHT), "--width", str(WIDTH), "--workers", "1",
    ])


def config(steps, ratio):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        steps,
        warmup_percent=20.0 if steps == STEPS else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=ratio,
        sol_log_density=False,
        landmark_tree_v2_midpoint_direction_mode="fused",
        sol_route_topk_execution="fused",
    )


def git(*items):
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
    diff = subprocess.check_output(["git", "-C", str(base.IMPL), "diff", "--binary"])
    protocol = dict(
        status="running",
        name=NAME,
        hypothesis=(
            "one SM120 fused callable can safely reuse 10%, 20%, and 30% Top-K "
            "ratios without changing steady-state denoise performance"
        ),
        cases=selected,
        gpus=list(GPUS),
        modes=list(MODES),
        ratios=list(RATIOS),
        frames=FRAMES,
        resolution=[WIDTH, HEIGHT],
        steps=STEPS,
        evaluations=STEPS - 1,
        seed=42,
        design=(
            "every GPU runs both modes; mode and ratio orders are reversed across GPUs; "
            "a discarded 3-step 20% warmup precedes each mode; a final 20% repeat is "
            "compile-free in both modes"
        ),
        expected_compile_calls={"static": 3, "runtime": 1},
        configs={str(ratio): dataclasses.asdict(config(STEPS, ratio)) for ratio in RATIOS},
        git_revision=git("rev-parse", "HEAD"),
        git_diff_sha256=hashlib.sha256(diff).hexdigest(),
        runner_sha256=base.sha(Path(__file__).resolve()),
        timing="CUDA-synchronized denoise only; signatures and cleanup excluded",
    )
    base.write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def output_signature(output):
    """Small deterministic signature, computed outside the timed region."""
    import torch

    result = {}
    for name in ("latents", "audio_latents"):
        value = output[name]
        flat = value.reshape(-1)
        count = min(257, flat.numel())
        indices = torch.linspace(
            0, flat.numel() - 1, count, device=value.device, dtype=torch.float64
        ).to(torch.int64)
        sample = flat.index_select(0, indices).float()
        result[name] = {
            "shape": list(value.shape),
            "sample_sum": float(sample.sum(dtype=torch.float64).item()),
            "sample_abs_sum": float(sample.abs().sum(dtype=torch.float64).item()),
            "sample_values": sample[:16].cpu().tolist(),
        }
    return result


def run_once(pipe, p, original_state, *, steps, ratio, seed):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention
    from h3_sparse_attention.sol_numerator_virtual_q import fused_compile_cache_stats

    state = p.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    before = fused_compile_cache_stats()
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
    assert summary["dense_evaluations"] == (4 if steps == STEPS else 1), summary
    assert all(
        torch.isfinite(output[name]).all().item()
        for name in ("latents", "audio_latents")
    )
    signature = output_signature(output)
    after = fused_compile_cache_stats()
    del output, state
    return dict(
        seconds=seconds,
        ratio=ratio,
        steps=steps,
        attention_summary=summary,
        output_signature=signature,
        compile_before=before,
        compile_after=after,
        new_compile_calls=after["compile_calls"] - before["compile_calls"],
        new_compile_seconds=after["compile_seconds"] - before["compile_seconds"],
    )


def worker(rank):
    import torch
    import h3_sparse_attention.sol_numerator_virtual_q as fused

    rank = int(rank)
    gpu = GPUS[rank]
    torch.set_num_threads(4)
    p = base.pipeline()
    selected = cases()
    workflow, states = p.configure_denoise_workflow(args(), selected)
    pipe, manager, acceleration, placement = p.load_denoiser(args(), workflow)
    mode_order = MODES if rank < 2 else tuple(reversed(MODES))
    ratio_order = RATIOS if rank % 2 == 0 else tuple(reversed(RATIOS))
    try:
        fused._FUSED_COMPILED.clear()
        fused._FUSED_COMPILE_CALLS = 0
        fused._FUSED_COMPILE_SECONDS = 0.0
        for mode_order_index, mode in enumerate(mode_order):
            os.environ["H3_SM120_RUNTIME_TOPK_RATIO"] = "1" if mode == "runtime" else "0"
            warm = run_once(
                pipe, p, states[rank], steps=3, ratio=0.20,
                seed=40 + mode_order_index,
            )
            base.write(ROOT / "records" / f"gpu{gpu}_{mode}_warmup.json", {
                "phase": "discarded_warmup", "mode": mode, "gpu": gpu,
                "case": selected[rank]["index"], "placement": placement, **warm,
            })
            print("WARMUP", mode, round(warm["seconds"], 3),
                  "new_compiles", warm["new_compile_calls"], flush=True)
            for ratio_order_index, ratio in enumerate(ratio_order):
                row = run_once(pipe, p, states[rank], steps=STEPS, ratio=ratio, seed=42)
                label = f"{round(ratio * 100):02d}pct"
                base.write(ROOT / "records" / f"gpu{gpu}_{mode}_{label}.json", {
                    "phase": "ratio_sweep", "mode": mode, "gpu": gpu,
                    "mode_order": mode_order_index, "ratio_order": ratio_order_index,
                    "case": selected[rank]["index"], "placement": placement, **row,
                })
                print("MEASURED", mode, label, round(row["seconds"], 3),
                      "new_compiles", row["new_compile_calls"], flush=True)
            repeat = run_once(pipe, p, states[rank], steps=STEPS, ratio=0.20, seed=42)
            base.write(ROOT / "records" / f"gpu{gpu}_{mode}_20pct_repeat.json", {
                "phase": "steady_repeat", "mode": mode, "gpu": gpu,
                "mode_order": mode_order_index, "case": selected[rank]["index"],
                "placement": placement, **repeat,
            })
            print("REPEAT", mode, round(repeat["seconds"], 3),
                  "new_compiles", repeat["new_compile_calls"], flush=True)
        base.write(ROOT / f"worker_gpu{gpu}.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def _mean(rows, field="seconds"):
    return statistics.mean(row[field] for row in rows)


def summarize():
    rows = [json.loads(path.read_text()) for path in (ROOT / "records").glob("*.json")]
    result = {"status": "complete", "ratios": {}, "steady_20pct": {}, "compile": {}}
    for ratio in RATIOS:
        label = f"{round(ratio * 100):02d}pct"
        result["ratios"][label] = {}
        for mode in MODES:
            selected = [
                row for row in rows
                if row.get("phase") == "ratio_sweep"
                and row["mode"] == mode and row["ratio"] == ratio
            ]
            if len(selected) != len(GPUS):
                raise RuntimeError(f"missing {mode} {label}: {len(selected)}")
            result["ratios"][label][mode] = {
                "seconds": [row["seconds"] for row in selected],
                "mean_seconds": _mean(selected),
                "new_compile_calls": [row["new_compile_calls"] for row in selected],
                "new_compile_seconds": [row["new_compile_seconds"] for row in selected],
            }
        result["ratios"][label]["runtime_minus_static_seconds"] = (
            result["ratios"][label]["runtime"]["mean_seconds"]
            - result["ratios"][label]["static"]["mean_seconds"]
        )
    for mode in MODES:
        selected = [
            row for row in rows
            if row.get("phase") == "steady_repeat" and row["mode"] == mode
        ]
        if len(selected) != len(GPUS):
            raise RuntimeError(f"missing steady {mode}: {len(selected)}")
        result["steady_20pct"][mode] = {
            "seconds": [row["seconds"] for row in selected],
            "mean_seconds": _mean(selected),
            "new_compile_calls": [row["new_compile_calls"] for row in selected],
        }
    result["steady_20pct"]["runtime_minus_static_seconds"] = (
        result["steady_20pct"]["runtime"]["mean_seconds"]
        - result["steady_20pct"]["static"]["mean_seconds"]
    )
    for mode in MODES:
        selected = [row for row in rows if row.get("mode") == mode]
        by_gpu = {}
        for gpu in GPUS:
            gpu_rows = [row for row in selected if row["gpu"] == gpu]
            by_gpu[str(gpu)] = {
                "calls": sum(row["new_compile_calls"] for row in gpu_rows),
                "seconds": sum(row["new_compile_seconds"] for row in gpu_rows),
            }
        result["compile"][mode] = by_gpu
    signatures_match = True
    signature_mismatches = []
    for gpu in GPUS:
        for phase, ratio in (("ratio_sweep", 0.10), ("ratio_sweep", 0.20),
                             ("ratio_sweep", 0.30), ("steady_repeat", 0.20)):
            pair = [
                row for row in rows
                if row.get("phase") == phase and row.get("gpu") == gpu
                and row.get("ratio") == ratio
            ]
            if len(pair) != 2 or pair[0]["output_signature"] != pair[1]["output_signature"]:
                signatures_match = False
                signature_mismatches.append({"gpu": gpu, "phase": phase, "ratio": ratio})
    expected_compiles = all(
        result["compile"]["static"][str(gpu)]["calls"] == 3
        and result["compile"]["runtime"][str(gpu)]["calls"] == 1
        for gpu in GPUS
    )
    steady_delta = result["steady_20pct"]["runtime_minus_static_seconds"]
    result.update(
        signatures_match=signatures_match,
        signature_mismatches=signature_mismatches,
        expected_compile_reuse=expected_compiles,
        steady_overhead_percent=(
            steady_delta / result["steady_20pct"]["static"]["mean_seconds"] * 100.0
        ),
        accepted=(signatures_match and expected_compiles and abs(steady_delta) <= 2.0),
        acceptance_rule="matching signatures, 3-to-1 compile reduction, and <=2s steady 20% delta",
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
