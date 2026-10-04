#!/usr/bin/env python3
"""Remeasure the 8-evaluation VDN-table Dense/Spark rows in BF16 and FP8."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base

NAME = "vdn8_warmup_correction_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
CASES = (14, 36)
ARMS = ("dense_bf16", "spark_bf16", "dense_fp8", "spark_fp8")
GPUS = (0, 1, 2, 3)
STEPS, EVALUATIONS, FRAMES = 9, 8, 345


def write(path, value):
    base.write(path, value)


def pipeline_args():
    return base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--case-indices", *map(str, CASES),
        "--steps", str(STEPS), "--frames", str(FRAMES),
        "--height", "768", "--width", "1344", "--workers", "1",
    ])


def spark_config(steps=STEPS):
    from h3_sparse_attention import H3SparseAttentionConfig
    # ceil((8 + 1) * 20%) == two schedule-dense evaluations.
    return H3SparseAttentionConfig.spark(
        steps, warmup_percent=20.0, sol_dense_layers=1,
        sol_route_topk_ratio=.1, sol_log_density=False,
    )


class CudaTimer:
    def __init__(self, torch):
        self.torch, self.pairs, self.calls = torch, [], 0

    def wrap(self, fn):
        def timed(*args, **kwargs):
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return fn(*args, **kwargs)
            finally:
                end.record(); self.pairs.append((start, end)); self.calls += 1
        return self.torch.compiler.disable(timed)

    def reset(self):
        self.pairs.clear(); self.calls = 0

    def collect(self):
        self.torch.cuda.synchronize()
        result = {"seconds": sum(s.elapsed_time(e) for s, e in self.pairs) / 1e3,
                  "calls": self.calls}
        self.reset()
        return result


def install_attn_timers(torch, pipe):
    import torch.nn.functional as functional
    import h3_sparse_attention.processor as processor
    timers = {name: CudaTimer(torch) for name in ("attn_module", "sdpa", "sol_core")}
    forwards = [(block.attn, block.attn.forward) for block in pipe.transformer.transformer_blocks]
    original_sdpa, original_sol = functional.scaled_dot_product_attention, processor._sol_attention
    for attn, forward in forwards:
        attn.forward = timers["attn_module"].wrap(forward)
    functional.scaled_dot_product_attention = timers["sdpa"].wrap(original_sdpa)
    processor._sol_attention = timers["sol_core"].wrap(original_sol)

    def uninstall():
        for attn, forward in forwards:
            attn.forward = forward
        functional.scaled_dot_product_attention = original_sdpa
        processor._sol_attention = original_sol
    return timers, uninstall


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True); OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(SOURCE / name, target_is_directory=name.endswith("cache"))
    selected = base.pipeline().load_cases(SAMPLES, list(CASES), expected_indices=tuple(range(1, 51)))
    base.pipeline().configure_denoise_workflow(pipeline_args(), selected)
    write(ROOT / "protocol.json", {
        "status": "running", "arms": ARMS, "gpus": GPUS, "cases": CASES,
        "steps": STEPS, "actual_evaluations": EVALUATIONS, "frames": FRAMES,
        "resolution": [1344, 768], "seed": 42,
        "spark_schedule": "two schedule-dense evaluations and dense transformer layer 0",
        "runtime_warmup": "excluded 3-requested-step pass before each arm",
        "fp8": "install_fp8 on both FP8 arms before timers/plugin",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.IMPL, text=True).strip(),
        "runner_sha256": base.sha(Path(__file__).resolve()),
    })


def load_fp8_denoiser(torch, p, args, workflow):
    """Quantize on CPU before resident placement to avoid a ~95 GiB peak."""
    from diffusers import ComponentsManager
    from h3_sparse_attention.fp8_linear import install_fp8
    lock = p.acquire_weight_load_lock()
    manager = ComponentsManager()
    acceleration = None
    try:
        pipe = workflow.init_pipeline(str(args.model.resolve()), components_manager=manager)
        pipe.load_components(
            dtype=torch.bfloat16,
            pretrained_model_name_or_path={"default": str(args.model.resolve())},
        )
        acceleration = p.install_h3_acceleration(
            pipe, p.H3AccelerationConfig(torch_compile=True, vae_fp16=False))
        acceleration.__enter__()
        wrapped = p._impl_bootstrap.verify_transformer_compile(
            acceleration, pipe.transformer, requested=True)
        assert wrapped == 50, wrapped
        swapped = install_fp8(pipe.transformer)
        pipe.transformer.to("cuda")
        torch.cuda.synchronize()
    except Exception:
        if acceleration is not None:
            acceleration.remove()
        raise
    finally:
        p.release_weight_load_lock(lock)
    p.release_cpu_arenas()
    return pipe, manager, acceleration, {"mode": "resident", "resident_blocks": 50}, swapped


def run_one(torch, pipe, p, original_state, sparse, steps, timers):
    from h3_sparse_attention import install_h3_sparse_attention
    state = p.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
    cfg = spark_config(steps) if sparse else None
    ctx = install_h3_sparse_attention(pipe.transformer, cfg) if cfg else contextlib.nullcontext(None)
    for timer in timers.values(): timer.reset()
    with ctx as plugin, torch.inference_mode():
        if plugin: plugin.reset()
        torch.cuda.synchronize(); started = time.perf_counter()
        result = pipe(
            state=state, num_frames=FRAMES, height=768, width=1344,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize(); seconds = time.perf_counter() - started
        timing = {name: timer.collect() for name, timer in timers.items()}
        summary = plugin.summary() if plugin else {"method": "dense"}
    assert all(torch.isfinite(result[key]).all().item() for key in ("latents", "audio_latents"))
    del result, state
    return seconds, timing, summary


def worker(rank):
    import torch
    rank = int(rank); arm = ARMS[rank]; sparse = arm.startswith("spark"); fp8 = arm.endswith("fp8")
    torch.set_num_threads(4); p = base.pipeline()
    selected = p.load_cases(SAMPLES, list(CASES), expected_indices=tuple(range(1, 51)))
    workflow, states = p.configure_denoise_workflow(pipeline_args(), selected)
    if fp8:
        pipe, manager, acceleration, placement, swapped = load_fp8_denoiser(
            torch, p, pipeline_args(), workflow)
    else:
        pipe, manager, acceleration, placement = p.load_denoiser(pipeline_args(), workflow)
        swapped = 0
    timers, uninstall = install_attn_timers(torch, pipe)
    try:
        warm_seconds, warm_timing, warm_summary = run_one(torch, pipe, p, states[0], sparse, 3, timers)
        write(ROOT / f"warmup_{arm}.json", {"seconds": warm_seconds, "timing": warm_timing,
              "attention_summary": warm_summary, "fp8_linears": swapped})
        for case, state in zip(selected, states):
            seconds, timing, summary = run_one(torch, pipe, p, state, sparse, STEPS, timers)
            if sparse:
                assert summary["completed_evaluations"] == EVALUATIONS, summary
                assert summary["dense_evaluations"] == 2, summary
            expected = len(pipe.transformer.transformer_blocks) * EVALUATIONS
            assert timing["attn_module"]["calls"] == expected, timing
            if sparse:
                assert timing["sol_core"]["calls"] > 0, timing
            else:
                assert timing["sol_core"]["calls"] == 0, timing
            row = {"arm": arm, "case": case["index"], "sample_id": case["sample_id"],
                   "gpu": rank, "precision": "fp8" if fp8 else "bf16", "sparse": sparse,
                   "denoise_seconds": seconds, "attn_timing": timing,
                   "attention_summary": summary, "fp8_linears": swapped,
                   "steps": STEPS, "evaluations": EVALUATIONS, "placement": placement,
                   "config": dataclasses.asdict(spark_config()) if sparse else {"method": "dense"}}
            write(ROOT / "records" / f"{arm}_{case['index']:02}.json", row)
            print(arm, case["index"], round(seconds, 3),
                  {k: round(v["seconds"], 3) for k, v in timing.items()}, flush=True)
        write(ROOT / f"worker_gpu{rank}.json", {"status": "complete", "arm": arm})
    finally:
        uninstall(); acceleration.remove(); del pipe, manager; p.release_cpu_arenas()


def summarize():
    rows = [json.loads(path.read_text()) for path in sorted((ROOT / "records").glob("*.json"))]
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        assert len(selected) == len(CASES), (arm, len(selected))
        summary[arm] = {
            "n": len(selected),
            "mean_denoise_seconds": statistics.mean(r["denoise_seconds"] for r in selected),
            "per_eval_dit_seconds": statistics.mean(r["denoise_seconds"] for r in selected) / EVALUATIONS,
            "mean_attn_module_seconds": statistics.mean(r["attn_timing"]["attn_module"]["seconds"] for r in selected),
            "per_eval_attn_module_seconds": statistics.mean(r["attn_timing"]["attn_module"]["seconds"] for r in selected) / EVALUATIONS,
            "per_case": {str(r["case"]): r for r in selected},
        }
    for precision in ("bf16", "fp8"):
        dense, spark = summary[f"dense_{precision}"], summary[f"spark_{precision}"]
        spark["dit_speedup_vs_dense"] = dense["per_eval_dit_seconds"] / spark["per_eval_dit_seconds"]
        spark["attn_module_speedup_vs_dense"] = dense["per_eval_attn_module_seconds"] / spark["per_eval_attn_module_seconds"]
    result = {"status": "complete", "summary": summary, "rows": rows}
    write(ROOT / "results.json", result); print(json.dumps(result["summary"], indent=2), flush=True)


def launch():
    prepare(); jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen([str(base.PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    codes = [proc.wait() for proc, _ in jobs]
    for _, log in jobs: log.close()
    write(ROOT / "exit_codes.json", codes)
    if any(codes): raise RuntimeError(codes)
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run": launch()
    elif command == "worker": worker(sys.argv[2])
    elif command == "summarize": summarize()
    else: raise ValueError(command)
