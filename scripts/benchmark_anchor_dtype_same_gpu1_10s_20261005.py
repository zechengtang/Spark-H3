#!/usr/bin/env python3
"""Same-process GPU1 timing check for BF16 versus FP32 global anchors at 10s.

This intentionally measures full denoising while changing only
``sol_global_anchor_dtype``.  Each dtype receives an excluded short warmup and
the measured order alternates by prompt to reduce order/clock bias.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import torch

import run_route_mode_quality_25prompt_20261003 as study


ROOT = Path("/autodl-fs/data/h3_experiments/anchor_dtype_same_gpu1_10s_20261005")
RESULTS = ROOT / "results.json"
METHODS = ("bf16_anchor", "fp32_anchor")
CASE_INDICES = (1, 2, 3, 4)
CONDITIONING_SOURCE = Path(
    "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913"
)


def config(method: str, steps: int):
    baseline = study.config("legacy_threshold", steps)
    dtype = {"bf16_anchor": "bfloat16", "fp32_anchor": "float32"}[method]
    return dataclasses.replace(baseline, sol_global_anchor_dtype=dtype)


def write(payload):
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, RESULTS)


def run(pipe, pipeline, original_state, case, method: str, steps: int):
    from h3_sparse_attention import install_h3_sparse_attention

    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
    cfg = config(method, steps)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with install_h3_sparse_attention(pipe.transformer, cfg) as plugin, torch.inference_mode():
        plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipe(
            state=state,
            num_frames=240,
            height=768,
            width=1344,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary()
    assert summary["completed_evaluations"] == steps - 1
    row = {
        "case": case["index"],
        "sample_id": case["sample_id"],
        "method": method,
        "anchor_dtype": cfg.sol_global_anchor_dtype,
        "steps": steps,
        "seconds": seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "attention_summary": summary,
    }
    del result, state
    torch.cuda.empty_cache()
    return row


def summarize(rows):
    measured = [r for r in rows if r["phase"] == "measure"]
    means = {
        method: statistics.fmean(r["seconds"] for r in measured if r["method"] == method)
        for method in METHODS
        if any(r["method"] == method for r in measured)
    }
    by_case = {r["case"]: {} for r in measured}
    for row in measured:
        by_case[row["case"]][row["method"]] = row["seconds"]
    deltas = [
        v["bf16_anchor"] - v["fp32_anchor"]
        for v in by_case.values()
        if all(method in v for method in METHODS)
    ]
    return {
        "means_seconds": means,
        "bf16_minus_fp32_seconds": deltas,
        "mean_bf16_minus_fp32_seconds": statistics.fmean(deltas) if deltas else None,
        "bf16_slower_cases": sum(x > 0 for x in deltas),
        "count": len(deltas),
    }


def main():
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES=1")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    study.NAME = "anchor_dtype_same_gpu1_10s_20261005"
    study.CASES = CASE_INDICES
    pipeline = study.base.pipeline()
    selected = study.cases()
    args = study.parse_args(10)
    # Reuse the exact conditioning cache used by the historical 50-prompt
    # baseline. Conditioning is outside the timed region and independent of
    # anchor dtype.
    _, output_root = study.roots(10)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        destination = output_root / name
        if destination.exists() and not destination.is_symlink():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if not destination.exists():
            destination.symlink_to(
                CONDITIONING_SOURCE / name,
                target_is_directory=(CONDITIONING_SOURCE / name).is_dir(),
            )
    workflow, states = pipeline.configure_denoise_workflow(args, selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)

    payload = {
        "status": "running",
        "physical_gpu": 1,
        "device_name": torch.cuda.get_device_name(0),
        "cases": list(CASE_INDICES),
        "protocol": {
            "duration_seconds": 10,
            "frames": 240,
            "requested_steps": 20,
            "warmup_steps": 3,
            "timing": "CUDA-synchronized full denoise only",
            "pairing": "same process/GPU; alternating BF16-first and FP32-first order",
            "only_changed_field": "sol_global_anchor_dtype",
            "configs": {m: dataclasses.asdict(config(m, 20)) for m in METHODS},
            "placement": placement,
        },
        "rows": [],
    }
    write(payload)
    try:
        for method in METHODS:
            row = run(pipe, pipeline, states[0], selected[0], method, 3)
            row["phase"] = "warmup"
            payload["rows"].append(row)
            write(payload)
            print("WARMUP", method, f"{row['seconds']:.3f}s", flush=True)

        for position, (case, state) in enumerate(zip(selected, states, strict=True)):
            order = METHODS if position % 2 == 0 else tuple(reversed(METHODS))
            for method in order:
                row = run(pipe, pipeline, state, case, method, 20)
                row.update(phase="measure", order_in_pair=order.index(method) + 1)
                payload["rows"].append(row)
                payload["summary"] = summarize(payload["rows"])
                write(payload)
                print("MEASURED", case["index"], method, f"{row['seconds']:.3f}s", flush=True)
        payload["status"] = "complete"
        payload["summary"] = summarize(payload["rows"])
        write(payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
    finally:
        with contextlib.suppress(Exception):
            manager.cleanup()


if __name__ == "__main__":
    main()
