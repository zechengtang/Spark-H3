#!/usr/bin/env python3
"""Matched CUDA profiling of the Diffusers and ComfyUI Spark implementations.

The benchmark replays one captured real 10 s / 768p H3 attention input.  Run
the two modes in separate processes so the frozen Diffusers package and the
current ComfyUI package cannot contaminate each other's imports.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path("/autodl-fs/data/h3_experiments/spark_cross_pipeline_profile_20260925")
REPOSITORY = Path(__file__).resolve().parents[1]
CAPTURE = Path(
    "/autodl-fs/data/h3_experiments/attention_path_optimization_20260917/"
    "attention_input_gpu1.pt"
)
SNAPSHOT = Path(
    "/autodl-fs/data/h3_experiments/attn_kernel_speedup_20260921/snapshot"
)


def configure_environment() -> None:
    os.environ.update(
        H3_METRIC_FACTOR="cholesky",
        H3_LMV2_COS_PRECISION="fp16",
        H3_LMV2_SMALL_PROXY_FAST="1",
        H3_LMV2_FP8_FEATURES="0",
        H3_LMV2_FUSED_NODE="1",
        H3_LMV2_COS_FAST="1",
        H3_LMV2_GROUP1_FAST="1",
        H3_TEMPORAL_MIN_FRAMES="0",
        H3_TEMPORAL_CHUNK_FRAMES="0",
        H3_LMV2_TERMINAL_LEAVES="16",
    )


def cuda_benchmark(torch, fn, *, warmup: int = 4, iterations: int = 12):
    for _ in range(warmup):
        output = fn()
        del output
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
        del output
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def profiler_rows(torch, fn, label: str):
    # One warmed call is enough for kernel attribution; CUDA events above are
    # the authoritative latency measurement.
    output = fn()
    del output
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(label):
            output = fn()
        torch.cuda.synchronize()
        del output
    rows = []
    for event in prof.key_averages():
        device_us = getattr(event, "self_device_time_total", 0.0)
        if not device_us:
            device_us = getattr(event, "self_cuda_time_total", 0.0)
        if device_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "self_device_ms": float(device_us) / 1000.0,
            }
        )
    rows.sort(key=lambda row: row["self_device_ms"], reverse=True)
    return rows[:40]


def load_capture(torch):
    data = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    q, k, v = (
        data[name].to("cuda", non_blocking=False).permute(0, 2, 1, 3).contiguous()
        for name in ("q", "k", "v")
    )
    return data, q, k, v


def layout_kwargs(torch, data):
    names = {
        "permutation",
        "inverse_permutation",
        "grid",
        "video_tokens",
        "sequence_length",
        "video_positions",
    }
    return {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in data["layout"].items()
        if key in names
    }


def run_diffusers(*, current=False):
    # Import the exact source snapshot used by the established 343.83 s versus
    # 355.48 s Diffusers result.
    sys.path.insert(0, str(REPOSITORY if current else SNAPSHOT))
    import torch
    import h3_sparse_attention.processor as proc
    import h3_sparse_attention.spark_integration as spark

    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    # The processor-facing representation is BHTD.
    qh, kh, vh = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    layout = proc.PackedLayout(**layout_kwargs(torch, data))
    common = dict(
        warmup_percent=20,
        sol_dense_layers=1,
        sol_tau=1.0,
        sol_log_density=False,
    )
    configs = {
        # Ordinary Sol retains its default local three-block exact rule.
        # TopK/Spark retain their no-forced-local routing policy.
        "sol": proc.H3SparseAttentionConfig.sol(20, **common),
        "topk10": proc.H3SparseAttentionConfig.sol(
            20, **common, sol_route_topk_ratio=0.1, sol_force_local_blocks=False,
        ),
        "topk10_reblock": proc.H3SparseAttentionConfig.sol(
            20,
            **common,
            sol_route_topk_ratio=0.1,
            sol_force_local_blocks=False,
            sol_landmark_preprocess=True,
            sol_landmark_preprocess_version="v2",
            landmark_tree_v2_children=16,
            landmark_tree_v2_fanout_mode="power_of_two_fanout",
        ),
        "spark_global": proc.H3SparseAttentionConfig.sol(
            20,
            **common,
            sol_route_topk_ratio=0.1,
            sol_force_local_blocks=False,
            sol_landmark_preprocess=True,
            sol_landmark_preprocess_version="v2",
            landmark_tree_v2_children=16,
            landmark_tree_v2_fanout_mode="power_of_two_fanout",
            sol_virtual_query_levels_up=99,
            sol_virtual_query_target_blocks=None,
        ),
    }
    controllers = {}
    calls = {}
    for name, config in configs.items():
        controller = proc._Controller(config)
        controller.evaluation_index = 5
        controllers[name] = controller
        calls[name] = lambda c=controller: proc._sol_attention(
            c, qh, kh, vh, layout, int(data["layer"]), return_bthd=True
        )

    # Isolate the dynamic planning and explicit tensor-transport stages used by
    # the frozen Diffusers implementation.
    plan_controller = proc._Controller(configs["spark_global"])
    plan_controller.evaluation_index = 5

    def plan_call():
        return spark._landmark_tree_v2_qk_block_permutations(
            plan_controller, q, k, layout
        )

    plan_result = plan_call()
    query_perm, query_inverse, key_perm = plan_result[0], plan_result[1], plan_result[2]

    def transport_call():
        qr = spark._headwise_permute_video_tokens(q, query_perm, video_tokens=layout.video_tokens)
        kr = spark._headwise_permute_video_tokens(k, key_perm, video_tokens=layout.video_tokens)
        vr = spark._headwise_permute_video_tokens(v, key_perm, video_tokens=layout.video_tokens)
        restored = spark._headwise_permute_video_tokens(
            qr, query_inverse, video_tokens=layout.video_tokens
        )
        return qr, kr, vr, restored

    calls["reblock_plan_only"] = plan_call
    calls["explicit_reblock_transport"] = transport_call
    # Matched reweight increment with identical precomputed permutations.
    # This is a measurement boundary, not a production reuse policy change.
    from unittest.mock import patch
    def cached_plan(c, *args, **kwargs):
        c.landmark_reblock_hierarchy = plan_controller.landmark_reblock_hierarchy
        return plan_result
    def cached_call(name):
        with patch.object(spark, "_landmark_tree_v2_qk_block_permutations", cached_plan):
            return calls[name]()
    for name in ("topk10_reblock", "spark_global"):
        calls[name + "_cached_plan"] = lambda n=name: cached_call(n)
    timings = {name: cuda_benchmark(torch, fn) for name, fn in calls.items()}
    profiles = {
        name: profiler_rows(torch, calls[name], f"diffusers/{name}")
        for name in ("sol", "topk10", "topk10_reblock", "spark_global", "reblock_plan_only")
    }
    result = {
        "side": "diffusers_current" if current else "diffusers_snapshot_20260921",
        "capture": str(CAPTURE),
        "shape_bthd": list(q.shape),
        "video_tokens": layout.video_tokens,
        "settings": {
            "force_local": {
                name: config.sol_local_blocks_enabled
                for name, config in configs.items()
            },
            "topk_ratio": 0.1,
            "fanout": 16,
            "reweight": "global",
            "reuse": 1,
        },
        "timings": timings,
        "profiles": profiles,
        "configs": {name: dataclasses.asdict(config) for name, config in configs.items()},
    }
    return result


def run_comfy(*, fingerprint=False):
    sys.path.insert(0, str(REPOSITORY))
    import torch
    from comfy_kitchen.backends import cuda as ck
    from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
    from comfyui_reblock_plan import build_comfy_reblock_permutations

    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    config = ComfySparkConfig(
        topk_ratio=0.1,
        tau=1.0,
        global_anchor_dtype="float32",
        landmark_tree_v2_children=16,
    )
    controller = ComfySparkController(config)
    controller.evaluation_index = 5
    blocks = (q.shape[1] + 63) // 64
    sink_first = layout.video_tokens // 64
    sinks = [sink_first, blocks]

    def plan_call():
        return build_comfy_reblock_permutations(controller, q, k, layout)

    plan_result = plan_call()
    query_perm, query_inverse, key_perm = plan_result[:3]

    def native_topk_exact_only():
        # Match Spark's native fixed-budget selector, without pooled or global
        # reweight. The public Sol tail=False wrapper uses an external route
        # oracle; call the native entry explicitly to avoid that confounder.
        batch, tokens, heads, dim = q.shape
        plan = ck._C.sol_attn_plan(batch, tokens, heads, token_aug=0)
        workspace = torch.empty(plan["total"], dtype=torch.uint8, device=q.device)
        output = torch.empty_like(q)
        wrap = ck._wrap_for_dlpack
        count = ck._topk_count(sink_first, 0.1)
        ck._C.sol_attn(
            wrap(q), wrap(k), wrap(v), wrap(output), wrap(workspace),
            batch, tokens, heads, dim, 1.0, dim ** -0.5,
            sinks[0], sinks[1], sinks[0], sinks[1],
            torch.cuda.current_stream(q.device).cuda_stream,
            key_bias=None, threshold=None, block_len=None,
            tail=False, token_aug=0, topk_count=count,
        )
        return output

    calls = {
        "native_topk10_exact_only": native_topk_exact_only,
        "sol_tau1": lambda: ck.sol_attn(
            q, k, v, tau=1.0, sink_blocks=sinks, sink_q=sinks
        ),
        "official_topk10_pooled": lambda: ck.sol_attn(
            q, k, v, topk_ratio=0.1, sink_blocks=sinks, sink_q=sinks
        ),
        "spark_global_no_reblock": lambda: ck.spark_attn(
            q,
            k,
            v,
            video_tokens=layout.video_tokens,
            anchor_dtype=torch.float32,
            topk_ratio=0.1,
            sink_blocks=sinks,
            sink_q=sinks,
        ),
        "spark_global_fused_reblock": lambda: ck.spark_attn(
            q,
            k,
            v,
            video_tokens=layout.video_tokens,
            anchor_dtype=torch.float32,
            topk_ratio=0.1,
            sink_blocks=sinks,
            sink_q=sinks,
            query_permutation=query_perm,
            key_permutation=key_perm,
            query_inverse=query_inverse,
        ),
        "reblock_plan_only": plan_call,
    }
    timings = {name: cuda_benchmark(torch, fn) for name, fn in calls.items()}
    profiles = {
        name: profiler_rows(torch, calls[name], f"comfy/{name}")
        for name in calls
    }
    fingerprints = {}
    if fingerprint:
        import hashlib
        for name in ("sol_tau1", "spark_global_no_reblock", "spark_global_fused_reblock"):
            output = calls[name]()
            raw = output.contiguous().view(torch.uint8).cpu().numpy()
            fingerprints[name] = hashlib.sha256(raw).hexdigest()
            del raw, output
    return {
        "side": "comfy_kitchen_current",
        "output_sha256": fingerprints,
        "comfy_spark_prune_env": os.environ.get("COMFY_SPARK_PRUNE", "default"),
        "comfy_spark_warp_mode_env": os.environ.get("COMFY_SPARK_WARP_MODE", "default"),
        "comfy_spark_cache_summary_k_env": os.environ.get("COMFY_SPARK_CACHE_SUMMARY_K", "default"),
        "capture": str(CAPTURE),
        "shape_bthd": list(q.shape),
        "video_tokens": layout.video_tokens,
        "settings": {
            "topk_ratio": 0.1,
            "fanout": 16,
            "reweight": "global",
            "anchor_dtype": "float32",
            "reuse": 1,
        },
        "timings": timings,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=("diffusers", "comfy"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--current-diffusers", action="store_true")
    parser.add_argument("--fingerprint", action="store_true",
                        help="Hash Comfy outputs after timing; useful for binary A/B checks")
    args = parser.parse_args()
    configure_environment()
    ROOT.mkdir(parents=True, exist_ok=True)
    result = run_diffusers(current=args.current_diffusers) if args.side == "diffusers" else run_comfy(fingerprint=args.fingerprint)
    output = args.output or ROOT / f"{args.side}.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "timings": result["timings"]}, indent=2))


if __name__ == "__main__":
    main()
