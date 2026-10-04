#!/usr/bin/env python3
"""One-prompt, same-session denoise probe for the external-route prototype.

This is an experiment-only runner.  It imports the packed-mask prototype and
temporarily patches the route builder only for the compatibility arm.
Production source and defaults remain untouched.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

import run_route_mode_quality_25prompt_20261003 as study
from prototype_threshold_compatible_external_20261004 import (
    threshold_compatible_packed_route,
)


ROOT = Path("/autodl-fs/data/h3_experiments/threshold_compatible_external_full_probe_20261004")
OUT = Path("/autodl-fs/data/h3_outputs/threshold_compatible_external_full_probe_20261004")
CASES = {5: 8, 10: 12}
METHODS = ("legacy_threshold", "legacy_external", "legacy_compatible_external")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(16 << 20):
            digest.update(data)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def config(method: str, steps: int):
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        steps,
        warmup_percent=20.0 if steps == 20 else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=0.1,
        sol_log_density=False,
        landmark_tree_v2_midpoint_direction_mode="legacy",
        sol_route_topk_execution=(
            "threshold" if method == "legacy_threshold"
            else "packed_external_no_route_qk"
        ),
    )


def args(duration: int, case_index: int):
    return study.base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(study.SAMPLES), "--output", str(OUT / f"{duration}s"),
        "--method", "dense", "--case-indices", str(case_index),
        "--steps", "20", "--frames", str(duration * 24),
        "--height", "768", "--width", "1344", "--workers", "1",
    ])


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float(), b.float()
    diff = af - bf
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "mismatched_elements": int((a != b).sum().item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(af)).item()
        ),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }


def run_method(pipe, pipeline, original_state, duration: int, method: str, steps: int):
    from h3_sparse_attention import install_h3_sparse_attention, sol_topk_cutoff

    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].cuda()
    cfg = config(method, steps)
    original_route = sol_topk_cutoff.gemm_topk_packed_route
    if method == "legacy_compatible_external":
        sol_topk_cutoff.gemm_topk_packed_route = threshold_compatible_packed_route
    try:
        with install_h3_sparse_attention(pipe.transformer, cfg) as plugin, torch.inference_mode():
            plugin.reset()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipe(
                state=state,
                num_frames=duration * 24,
                height=768,
                width=1344,
                num_inference_steps=steps,
                generator=torch.Generator(device="cpu").manual_seed(42),
                output=["latents", "audio_latents"],
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            summary = plugin.summary()
    finally:
        sol_topk_cutoff.gemm_topk_packed_route = original_route
    payload = {
        name: result[name].detach().cpu().contiguous()
        for name in ("latents", "audio_latents")
    }
    return payload, {
        "seconds": seconds,
        "config": dataclasses.asdict(cfg),
        "attention_summary": summary,
        "finite": all(torch.isfinite(value).all().item() for value in payload.values()),
    }


def worker(duration: int) -> None:
    case_index = CASES[duration]
    result_path = ROOT / f"{duration}s_case{case_index:02}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    output = OUT / f"{duration}s"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        target = output / name
        if not target.exists():
            source = study.SOURCE / name
            target.symlink_to(source, target_is_directory=source.is_dir())

    pipeline = study.base.pipeline()
    selected = pipeline.load_cases(
        study.SAMPLES, [case_index], expected_indices=tuple(range(1, 51))
    )
    run_args = args(duration, case_index)
    workflow, states = pipeline.configure_denoise_workflow(run_args, selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(run_args, workflow)
    records = {}
    payloads = {}
    try:
        # Compile/warm each arm outside the measured 20-step calls.
        for method in METHODS:
            warm, warm_record = run_method(
                pipe, pipeline, states[0], duration, method, 3
            )
            del warm
            records[method] = {"warmup_seconds": warm_record["seconds"]}
        for method in METHODS:
            payloads[method], record = run_method(
                pipe, pipeline, states[0], duration, method, 20
            )
            records[method].update(record)
            print(duration, method, record["seconds"], flush=True)
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()

    comparisons = {}
    for candidate in ("legacy_external", "legacy_compatible_external"):
        comparisons[f"{candidate}_vs_legacy_threshold"] = {
            key: metrics(payloads[candidate][key], payloads["legacy_threshold"][key])
            for key in ("latents", "audio_latents")
        }
    base_l2 = comparisons["legacy_external_vs_legacy_threshold"]["latents"]["relative_l2"]
    compat_l2 = comparisons[
        "legacy_compatible_external_vs_legacy_threshold"
    ]["latents"]["relative_l2"]
    write(result_path, {
        "status": "complete",
        "duration_seconds": duration,
        "case": selected[0],
        "seed": 42,
        "requested_steps": 20,
        "transformer_evaluations": 19,
        "placement": placement,
        "gpu": torch.cuda.get_device_name(),
        "production_source_modified": False,
        "prototype_source": str(Path(__file__).with_name(
            "prototype_threshold_compatible_external_20261004.py"
        )),
        "records": records,
        "comparisons": comparisons,
        "latent_relative_l2_reduction_percent": (
            (1.0 - compat_l2 / base_l2) * 100.0 if base_l2 else None
        ),
    })


def aggregate() -> None:
    rows = [json.loads((ROOT / f"{duration}s_case{CASES[duration]:02}.json").read_text())
            for duration in (5, 10)]
    write(ROOT / "results.json", {
        "status": "complete",
        "rows": rows,
        "runner_sha256": sha256(Path(__file__)),
        "prototype_sha256": sha256(Path(__file__).with_name(
            "prototype_threshold_compatible_external_20261004.py"
        )),
    })


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "worker":
        worker(int(sys.argv[2]))
    elif command == "aggregate":
        aggregate()
    else:
        raise SystemExit("use: worker <5|10> | aggregate")
