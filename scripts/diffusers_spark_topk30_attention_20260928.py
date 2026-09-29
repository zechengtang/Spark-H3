#!/usr/bin/env python3
"""Paired full-attention-call timing for current Dense vs Spark-H3-30pct.

GPU pairs (0,1), (2,3), (4,5) use prompts 2, 13, 31 respectively.
Each worker excludes one full 20-grid-point denoise as warmup, then times
all 50 attention modules across the 19 model evaluations with CUDA events.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base


NAME = "diffusers_spark_topk30_attention_20260928"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
COND = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
ASSIGNMENTS = ((2, "dense"), (2, "spark30"), (13, "dense"),
               (13, "spark30"), (31, "dense"), (31, "spark30"))


def args():
    return base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--steps", "20", "--frames", "240",
        "--height", "768", "--width", "1344", "--workers", "1",
    ])


def spark_config():
    base.pipeline()
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        20, sol_route_topk_ratio=0.3, sol_log_density=False,
        sol_video_tail_mode="dense", sol_tail_granularity="query",
        sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore",
        sol_reweight_logmass_key="stored", sol_reweight_components="full",
    )


def run_worker(rank):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    torch.set_num_threads(4)
    index, arm = ASSIGNMENTS[rank]
    p = base.pipeline()
    case = p.load_cases(SAMPLES, [index], expected_indices=tuple(range(1, 51)))[0]
    workflow, states = p.configure_denoise_workflow(args(), [case])
    base.write(ROOT / f"status_gpu{rank}.json", dict(stage="loading", index=index, arm=arm))
    pipe, manager, acceleration, placement = p.load_denoiser(args(), workflow)
    records = []
    originals = []
    try:
        for block in pipe.transformer.transformer_blocks:
            attn = block.attn
            original = attn.forward

            def measured_forward(*a, _forward=original, **kw):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    return _forward(*a, **kw)
                finally:
                    end.record()
                    records.append((start, end))

            originals.append((attn, original))
            attn.forward = torch.compiler.disable(measured_forward)

        for phase in ("warmup", "measured"):
            state = p.clone_state(states[0])
            state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
            records.clear()
            base.write(ROOT / f"status_gpu{rank}.json",
                       dict(stage=phase, index=index, arm=arm))
            config = spark_config() if arm == "spark30" else None
            if config is None:
                import contextlib
                ctx = contextlib.nullcontext()
            else:
                ctx = install_h3_sparse_attention(pipe.transformer, config)
            with ctx as plugin, torch.inference_mode():
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = pipe(
                    state=state, num_frames=240, height=768, width=1344,
                    num_inference_steps=20,
                    generator=torch.Generator(device="cpu").manual_seed(42),
                    output=["latents", "audio_latents"],
                )
                torch.cuda.synchronize()
                denoise_seconds = time.perf_counter() - started
                summary = plugin.summary() if plugin is not None else None
            expected = 50 * 19
            if len(records) != expected:
                raise AssertionError(f"attention calls: {len(records)} != {expected}")
            attn_seconds = sum(s.elapsed_time(e) for s, e in records) / 1000
            if attn_seconds > denoise_seconds * 1.05:
                raise AssertionError((attn_seconds, denoise_seconds))
            if summary is not None and summary["completed_evaluations"] != 19:
                raise AssertionError(summary)
            row = dict(phase=phase, gpu=rank, case=index, sample_id=case["sample_id"],
                       arm=arm, denoise_seconds=denoise_seconds,
                       attention_module_seconds=attn_seconds,
                       attention_calls=len(records), torch_compile=True,
                       placement=placement)
            base.write(ROOT / f"{phase}_gpu{rank}.json", row)
            print(json.dumps(row), flush=True)
            del result, state
        base.write(ROOT / f"status_gpu{rank}.json",
                   dict(stage="complete", index=index, arm=arm))
    finally:
        for attn, original in originals:
            attn.forward = original
        acceleration.remove()


def run():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("refusing to overwrite existing timing output")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for item in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / item).symlink_to(COND / item,
                                target_is_directory=item == "conditioning_cache")
    base.write(ROOT / "protocol.json", dict(
        assignments=ASSIGNMENTS, seed=42, steps=20, actual_evaluations=19,
        frames=240, height=768, width=1344, torch_compile=True,
        config=dataclasses.asdict(spark_config()),
        timing="CUDA-event inclusive block.attn.forward time, all 50 blocks, 19 evaluations",
        warmup="one excluded complete denoise per GPU before timed complete denoise",
        source=str(Path(__file__).resolve()),
        source_sha256=base.sha(Path(__file__).resolve()),
        code_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
    ))
    jobs = []
    for gpu in range(len(ASSIGNMENTS)):
        log = (ROOT / f"gpu{gpu}.log").open("w")
        proc = subprocess.Popen([sys.executable, __file__, "worker", str(gpu)],
                                env={**os.environ, **base.ENV,
                                     "CUDA_VISIBLE_DEVICES": str(gpu)},
                                stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    codes = [proc.wait() for proc, _ in jobs]
    for _, log in jobs:
        log.close()
    base.write(ROOT / "exit_codes.json", codes)
    if any(codes):
        raise RuntimeError(f"worker failures: {codes}")
    pairs = []
    for a, b in ((0, 1), (2, 3), (4, 5)):
        dense = json.loads((ROOT / f"measured_gpu{a}.json").read_text())
        spark = json.loads((ROOT / f"measured_gpu{b}.json").read_text())
        if dense["sample_id"] != spark["sample_id"]:
            raise AssertionError("unpaired prompt")
        pairs.append(dict(case=dense["case"], dense_attention_seconds=dense["attention_module_seconds"],
                          spark_attention_seconds=spark["attention_module_seconds"],
                          speedup=dense["attention_module_seconds"] / spark["attention_module_seconds"]))
    ratio_of_means = statistics.mean(p["dense_attention_seconds"] for p in pairs) / statistics.mean(
        p["spark_attention_seconds"] for p in pairs)
    base.write(ROOT / "results.json", dict(pairs=pairs, attention_speedup_ratio_of_means=ratio_of_means))
    print(json.dumps(dict(pairs=pairs, attention_speedup_ratio_of_means=ratio_of_means), indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        run_worker(int(sys.argv[2]))
    else:
        run()
