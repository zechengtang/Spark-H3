#!/usr/bin/env python3
"""Paired 5s Diffusers Sol/Spark timing on the four ComfyUI audit prompts."""
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


NAME = "diffusers_sol_spark_4prompt_5s768p_20260926"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
COMFY_PROTOCOL = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
INDICES = (2, 13, 31, 33)
GPUS = (0, 1, 2, 3)
METHODS = ("sol_tau1", "spark_10pct")


def cases():
    p = base.pipeline()
    selected = p.load_cases(SAMPLES, list(INDICES), expected_indices=tuple(range(1, 51)))
    comfy = json.loads(COMFY_PROTOCOL.read_text())["cases"]
    assert [(c["sample_id"], c["prompt_sha256"]) for c in selected] == [
        (c["sample_id"], c["prompt_sha256"]) for c in comfy
    ]
    return selected


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig
    common = dict(warmup_percent=20, sol_dense_layers=1, sol_tau=1.0,
                  sol_log_density=False, sol_video_tail_mode="dense")
    if method == "sol_tau1":
        return H3SparseAttentionConfig.sol(20, sol_force_local_blocks=False, **common)
    return H3SparseAttentionConfig.spark(20, **common)


def args():
    return base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--steps", "20", "--frames", "120",
        "--height", "768", "--width", "1344", "--workers", "1",
        "--no-torch-compile",
    ])


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for item in ("conditioning_cache", "conditioning_manifest.json"):
        link = OUT / item
        if not link.exists():
            link.symlink_to(SOURCE / item, target_is_directory=item == "conditioning_cache")
    selected = cases()
    base.pipeline().configure_denoise_workflow(args(), selected)
    from h3_sparse_attention import __file__ as implementation
    base.write(ROOT / "protocol.json", dict(
        status="running", sample_indices=INDICES, cases=selected,
        methods=METHODS, configs={m: dataclasses.asdict(config(m)) for m in METHODS},
        source_file=implementation, source_sha256=base.sha(implementation),
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=base.IMPL, text=True).strip(),
        seed=42, steps=20, actual_evaluations=19, warmup_evaluations=4,
        frames=120, resolution="1344x768", gpus=GPUS,
        warmup="excluded full 20-step denoise per arm on each GPU",
        measurement="paired synchronized full denoise; no VAE/conditioning/load/save",
        note="Current Diffusers implementation; historical 10s reference used a frozen older source.",
    ))


def worker(rank):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    torch.set_num_threads(4)
    p = base.pipeline()
    case = cases()[rank]
    workflow, states = p.configure_denoise_workflow(args(), [case])
    pipe, manager, acceleration, placement = p.load_denoiser(args(), workflow)
    try:
        # Opposite order on half the cards controls warm-cache/order effects.
        order = METHODS if rank % 2 == 0 else METHODS[::-1]
        for repeat in ("warmup", "measured", "measured2"):
            for method in order:
                target = ROOT / "records" / f"{repeat}_{method}_{case['index']:02}.json"
                if target.exists():
                    continue
                state = p.clone_state(states[0])
                state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
                with install_h3_sparse_attention(pipe.transformer, config(method)) as plugin, torch.inference_mode():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = pipe(
                        state=state, num_frames=120, height=768, width=1344,
                        num_inference_steps=20,
                        generator=torch.Generator(device="cpu").manual_seed(42),
                        output=["latents", "audio_latents"],
                    )
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    summary = plugin.summary()
                assert summary["completed_evaluations"] == 19, summary
                assert summary["dense_evaluations"] == 4, summary
                assert summary["processor_calls"].get("sparse:sol") == 735, summary
                assert all(torch.isfinite(output[k]).all() for k in ("latents", "audio_latents"))
                base.write(target, dict(case=case["index"], sample_id=case["sample_id"],
                    method=method, repeat=repeat, gpu=GPUS[rank], seconds=seconds,
                    attention_summary=summary, placement=placement))
                print("DONE", repeat, case["index"], method, round(seconds, 3), flush=True)
                del output, state
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def summarize():
    rows = [json.loads(p.read_text()) for p in (ROOT / "records").glob("measured*json")]
    out = {}
    for method in METHODS:
        values = [r["seconds"] for r in rows if r["method"] == method]
        assert len(values) == 8, (method, len(values))
        out[method] = dict(mean_seconds=statistics.mean(values), values=values)
    out["speedup_sol_over_spark"] = out["sol_tau1"]["mean_seconds"] / out["spark_10pct"]["mean_seconds"]
    base.write(ROOT / "results.json", out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "worker":
        worker(int(sys.argv[2]))
    elif command == "run":
        setup()
        jobs = []
        for rank, gpu in enumerate(GPUS):
            log = (ROOT / f"gpu{gpu}.log").open("a")
            env = {**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "H3_IMPL_REPO": str(base.IMPL)}
            proc = subprocess.Popen([str(base.PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((proc, log))
        for proc, log in jobs:
            assert proc.wait() == 0, f"worker failed; inspect {log.name}"
            log.close()
        summarize()
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
