#!/usr/bin/env python3
"""Paired full-denoise attention-operator timing for Dense and Spark-30.

Dense times the post-QKV attention dispatch. Spark times the complete sparse
operator, including route/reblock/reweight and its internal sink-query work.
Both include every attention call in the 19-evaluation denoise, but exclude
shared QKV and output projections. One complete denoise is discarded as warmup.
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

import diffusers_spark_topk30_attention_20260928 as prior


NAME = "diffusers_spark_topk30_attention_core_v2_20260928"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME


def worker(rank: int) -> None:
    import torch
    import diffusers.models.transformers.transformer_minimax_h3 as minimax_module
    import h3_sparse_attention.processor as sparse_module
    from h3_sparse_attention import install_h3_sparse_attention

    torch.set_num_threads(4)
    index, arm = prior.ASSIGNMENTS[rank]
    pipeline = prior.base.pipeline()
    arguments = prior.args()
    arguments.output = OUT
    case = pipeline.load_cases(prior.SAMPLES, [index], expected_indices=tuple(range(1, 51)))[0]
    workflow, states = pipeline.configure_denoise_workflow(arguments, [case])
    prior.base.write(ROOT / f"status_gpu{rank}.json", dict(stage="loading", case=index, arm=arm))
    pipe, manager, acceleration, placement = pipeline.load_denoiser(arguments, workflow)

    original_dispatch = minimax_module.dispatch_attention_fn
    original_sparse = sparse_module._sol_attention
    original_forwards = []
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    inside_transformer_attention = [0]

    def time_call(kind, func, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return func(*args, **kwargs)
        finally:
            end.record()
            events.append((kind, start, end))

    def timed_dispatch(*args, **kwargs):
        if not inside_transformer_attention[0]:
            return original_dispatch(*args, **kwargs)
        return time_call("dense_core", original_dispatch, *args, **kwargs)

    def timed_sparse(*args, **kwargs):
        return time_call("sparse_operator", original_sparse, *args, **kwargs)

    minimax_module.dispatch_attention_fn = torch.compiler.disable(timed_dispatch)
    sparse_module._sol_attention = torch.compiler.disable(timed_sparse)
    try:
        # Keep the same compiled-block graph-break boundary as the previous
        # paired attention-module experiment, but do not time projections.
        for block in pipe.transformer.transformer_blocks:
            attn = block.attn
            original = attn.forward
            original_forwards.append((attn, original))

            def forward_passthrough(*args, _original=original, **kwargs):
                inside_transformer_attention[0] += 1
                try:
                    return _original(*args, **kwargs)
                finally:
                    inside_transformer_attention[0] -= 1

            attn.forward = torch.compiler.disable(forward_passthrough)

        for phase in ("warmup", "measured"):
            state = pipeline.clone_state(states[0])
            state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
            events.clear()
            prior.base.write(ROOT / f"status_gpu{rank}.json",
                             dict(stage=phase, case=index, arm=arm))
            config = prior.spark_config() if arm == "spark30" else None
            if config is None:
                import contextlib
                context = contextlib.nullcontext()
            else:
                context = install_h3_sparse_attention(pipe.transformer, config)
            with context as plugin, torch.inference_mode():
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

            counts = {kind: sum(label == kind for label, _, _ in events)
                      for kind in ("dense_core", "sparse_operator")}
            if arm == "dense":
                expected = {"dense_core": 950, "sparse_operator": 0}
            else:
                expected = {"dense_core": 215, "sparse_operator": 735}
            if counts != expected:
                raise AssertionError(f"missing or double-counted attention calls: {counts} != {expected}")
            seconds_by_kind = {
                kind: sum(start.elapsed_time(end) for label, start, end in events if label == kind) / 1000
                for kind in counts
            }
            attention_seconds = sum(seconds_by_kind.values())
            if attention_seconds > denoise_seconds * 1.05:
                raise AssertionError((attention_seconds, denoise_seconds))
            if summary is not None and summary["completed_evaluations"] != 19:
                raise AssertionError(summary)
            row = dict(phase=phase, gpu=rank, case=index, sample_id=case["sample_id"],
                       arm=arm, denoise_seconds=denoise_seconds,
                       attention_core_seconds=attention_seconds,
                       seconds_by_kind=seconds_by_kind, calls_by_kind=counts,
                       torch_compile=True, placement=placement)
            prior.base.write(ROOT / f"{phase}_gpu{rank}.json", row)
            print(json.dumps(row), flush=True)
            del result, state
        prior.base.write(ROOT / f"status_gpu{rank}.json", dict(stage="complete", case=index, arm=arm))
    finally:
        for attn, original in original_forwards:
            attn.forward = original
        minimax_module.dispatch_attention_fn = original_dispatch
        sparse_module._sol_attention = original_sparse
        acceleration.remove()


def run() -> None:
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("refusing to overwrite existing timing output")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for item in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / item).symlink_to(prior.COND / item,
                                target_is_directory=item == "conditioning_cache")
    prior.base.write(ROOT / "protocol.json", dict(
        assignments=prior.ASSIGNMENTS, seed=42, steps=20, actual_evaluations=19,
        frames=240, height=768, width=1344, torch_compile=True,
        config=dataclasses.asdict(prior.spark_config()),
        timing="CUDA-event post-QKV attention dispatch for Dense; full sparse attention operator for Spark, including route/reblock/reweight/sink work; all 950 layer calls",
        warmup="one excluded complete denoise per GPU before timed complete denoise",
        instrumentation="same outer attn.forward graph break as 20260928 module experiment",
        source=str(Path(__file__).resolve()),
        source_sha256=prior.base.sha(Path(__file__).resolve()),
        code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    ))
    jobs = []
    for gpu in range(len(prior.ASSIGNMENTS)):
        log = (ROOT / f"gpu{gpu}.log").open("w")
        process = subprocess.Popen([sys.executable, __file__, "worker", str(gpu)],
                                   env={**os.environ, **prior.base.ENV,
                                        "CUDA_VISIBLE_DEVICES": str(gpu)},
                                   stdout=log, stderr=subprocess.STDOUT)
        jobs.append((process, log))
    codes = [process.wait() for process, _ in jobs]
    for _, log in jobs:
        log.close()
    prior.base.write(ROOT / "exit_codes.json", codes)
    if any(codes):
        raise RuntimeError(f"worker failures: {codes}")
    pairs = []
    for dense_gpu, sparse_gpu in ((0, 1), (2, 3), (4, 5)):
        dense = json.loads((ROOT / f"measured_gpu{dense_gpu}.json").read_text())
        sparse = json.loads((ROOT / f"measured_gpu{sparse_gpu}.json").read_text())
        if dense["sample_id"] != sparse["sample_id"]:
            raise AssertionError("unpaired prompt")
        pairs.append(dict(case=dense["case"], dense_core_seconds=dense["attention_core_seconds"],
                          spark_core_seconds=sparse["attention_core_seconds"],
                          speedup=dense["attention_core_seconds"] / sparse["attention_core_seconds"]))
    speedup = statistics.mean(x["dense_core_seconds"] for x in pairs) / statistics.mean(
        x["spark_core_seconds"] for x in pairs)
    result = dict(pairs=pairs, attention_core_speedup_ratio_of_means=speedup)
    prior.base.write(ROOT / "results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run()
    elif len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]))
    else:
        raise SystemExit("usage: diffusers_spark_topk30_attention_core_20260928.py [worker GPU]")
