#!/usr/bin/env python3
"""One-prompt warmed dense versus Spark-10 benchmark on an SM80 GPU."""

from __future__ import annotations

import contextlib
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import (
    MiniMaxH3ModularPipeline,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3CoreDenoiseStep,
)

from h3_sparse_attention import H3SparseAttentionConfig, install_h3_sparse_attention
from h3_sparse_attention.acceleration import install_h3_acceleration


MODEL = Path("/mnt/CFS/tangzecheng/models/MiniMax-H3")
CONDITIONING = Path(
    "/mnt/CFS/tangzecheng/calibration_f16_tau_20260919/conditioning_cache/01.pt"
)
STEPS = 20
HEIGHT = 768
WIDTH = 1344
SEED = 42


def result_path(seconds: int, tag: str, route_execution: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    route_suffix = "_cta_topk" if route_execution == "fused" else ""
    return (
        Path(__file__).resolve().parents[1]
        / "reports"
        / f"sm80_fused{route_suffix}_dense_vs_spark10_{seconds}s768p_seed42{suffix}.json"
    )


def load_pipeline() -> MiniMaxH3ModularPipeline:
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        MODEL / "transformer",
        dtype=torch.bfloat16,
        local_files_only=True,
        device_map=0,
        low_cpu_mem_usage=True,
    )
    scheduler = MiniMaxH3Scheduler.from_pretrained(
        MODEL / "scheduler", local_files_only=True
    )
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(
        MODEL / "audio_scheduler", local_files_only=True
    )
    pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3CoreDenoiseStep())
    pipe.register_components(
        transformer=transformer,
        scheduler=scheduler,
        audio_scheduler=audio_scheduler,
    )
    return pipe


def fresh_state():
    state = torch.load(CONDITIONING, map_location="cpu", weights_only=False)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    return state


def run(
    pipe,
    method: str,
    *,
    measured: bool,
    inference_steps: int = STEPS,
    installed_plugin=None,
    frames: int,
    route_execution: str,
):
    state = fresh_state()
    if method == "dense":
        attention = contextlib.nullcontext(None)
    elif method == "spark10":
        if installed_plugin is None:
            config = H3SparseAttentionConfig.spark(
                STEPS,
                warmup_percent=20.0,
                sol_dense_layers=1,
                sol_route_topk_ratio=0.10,
                sol_route_topk_execution=route_execution,
                sol_log_density=False,
            )
            assert config.dense_evaluations == 4
            attention = install_h3_sparse_attention(pipe.transformer, config)
        else:
            # Keep the exact same processors/controller alive across discarded
            # warmup and measurement so object guards cannot recompile the
            # first sparse step in the measured pass.
            attention = contextlib.nullcontext(installed_plugin)
    else:
        raise ValueError(method)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with attention as plugin, torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(
            state=state,
            num_frames=frames,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = None if plugin is None else plugin.summary()
        if plugin is None:
            profile_summary = None
        else:
            from h3_sparse_attention.spark_integration import (
                spark_reblock_profile_summary,
            )
            profile_summary = spark_reblock_profile_summary(plugin.controller)

    finite = all(torch.isfinite(output[name]).all().item() for name in output)
    record = {
        "method": method,
        "phase": "measured" if measured else "warmup_discarded",
        "seconds": seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "finite": finite,
        "attention_summary": summary,
        "cuda_profile": profile_summary,
    }
    if summary is not None:
        assert summary["sol_backend"] == "sm80_fused_virtual_query", summary
        calls = summary["processor_calls"]
        expected_evaluations = inference_steps - 1
        assert summary["completed_evaluations"] == expected_evaluations, summary
        assert summary["dense_evaluations"] == 4, summary
        assert calls["dense:warmup"] == 200, calls
        sparse_evaluations = expected_evaluations - 4
        assert calls["dense:dense_layer"] == sparse_evaluations, calls
        assert calls["sparse:sol"] == 49 * sparse_evaluations, calls
    del output, state
    print(json.dumps(record, default=str), flush=True)
    return record


def reset_measurement_statistics(plugin):
    """Restart the same prompt without discarding its warmed static plan."""
    controller = plugin.controller
    controller.evaluation_index = -1
    controller.layout = None
    controller.counts.clear()
    controller.sol_backend = None
    controller.sol_route_density = None
    controller.head_topk_budget = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-dense",
        action="store_true",
        help="reuse the measured dense row already in RESULT",
    )
    parser.add_argument(
        "--reuse-dense-report",
        type=Path,
        help="reuse the measured dense row from another completed report",
    )
    parser.add_argument("--seconds", type=int, choices=(5, 10, 15), default=10)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--route-execution", choices=("threshold", "fused"), default="threshold"
    )
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("this benchmark requires SM80")
    previous_dense = None
    result_file = result_path(args.seconds, args.tag, args.route_execution)
    # 15.0 * 24 = 360 rounds to 362, beyond the pipeline's 360-frame cap.
    # MiniMax-H3's established "15s" benchmark therefore uses the longest
    # supported 17*n+5 sequence: 345 frames (14.375 seconds at 24 fps).
    frames = {5: 120, 10: 240, 15: 345}[args.seconds]
    if args.reuse_dense or args.reuse_dense_report is not None:
        reuse_file = args.reuse_dense_report or result_file
        if not reuse_file.exists() and args.route_execution == "fused":
            reuse_file = result_path(args.seconds, args.tag, "threshold")
        if reuse_file.exists():
            previous = json.loads(reuse_file.read_text())
            if previous.get("duration_seconds") != args.seconds:
                raise ValueError(
                    f"dense reuse duration mismatch in {reuse_file}"
                )
            previous_dense = next(
                row for row in previous["runs"]
                if row["method"] == "dense" and row["phase"] == "measured"
            )
    loaded = time.perf_counter()
    pipe = load_pipeline()
    load_seconds = time.perf_counter() - loaded
    rows = []
    with install_h3_acceleration(pipe):
        config = H3SparseAttentionConfig.spark(
            STEPS,
            warmup_percent=20.0,
            sol_dense_layers=1,
            sol_route_topk_ratio=0.10,
            sol_route_topk_execution=args.route_execution,
            sol_log_density=False,
        )
        assert config.dense_evaluations == 4
        attention = install_h3_sparse_attention(pipe.transformer, config)
        with attention as plugin:
            # Discard one five-evaluation pass: four configured dense warmups
            # plus the first sparse evaluation.  Reuse this exact plugin for
            # measurement; reinstalling it invalidates torch object guards and
            # recompiles the first measured sparse step.
            rows.append(run(
                pipe, "spark10", measured=False, inference_steps=6,
                installed_plugin=plugin, frames=frames,
                route_execution=args.route_execution,
            ))
            reset_measurement_statistics(plugin)
            rows.append(run(
                pipe, "spark10", measured=True, installed_plugin=plugin,
                frames=frames, route_execution=args.route_execution,
            ))
        # The discarded Spark pass also executed four dense evaluations, so
        # the separate dense measurement starts from an already-warm graph.
        rows.append(
            previous_dense
            if previous_dense is not None
            else run(
                pipe, "dense", measured=True, frames=frames,
                route_execution=args.route_execution,
            )
        )

    measured = {row["method"]: row for row in rows if row["phase"] == "measured"}
    dense = measured["dense"]["seconds"]
    spark = measured["spark10"]["seconds"]
    result = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "model": str(MODEL),
        "conditioning": str(CONDITIONING),
        "seed": SEED,
        "requested_steps": STEPS,
        "transformer_evaluations": STEPS - 1,
        "duration_seconds": args.seconds,
        "requested_frames": frames,
        "resolution": [WIDTH, HEIGHT],
        "spark": {
            "topk_ratio": 0.10,
            "route_execution": args.route_execution,
            "dense_warmup_evaluations": 4,
            "dense_layers": 1,
        },
        "torch_compile": True,
        "dense_reused_from_previous_report": previous_dense is not None,
        "load_seconds": load_seconds,
        "runs": rows,
        "dense_seconds": dense,
        "spark_seconds": spark,
        "speedup_x": dense / spark,
        "latency_reduction_percent": (dense - spark) / dense * 100.0,
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
