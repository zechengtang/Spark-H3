#!/usr/bin/env python3
"""Compare SM80 runtime Top-K reuse with static-ratio specialization.

Run one process per GPU so compilation and timings remain device-local:
    python scripts/benchmark_sm80_runtime_topk_reuse.py --gpus 0 1 2 3 4 5 6 7 \
        --output /tmp/sm80_runtime_topk_reuse.json
"""

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


DEFAULT_VIDEO_TOKENS = 16384
DEFAULT_HEADS = 4
SINK_LENGTHS = (64, 81)
RATIOS = (0.1, 0.2)


def worker(gpu: int, repeats: int, output: Path, video_tokens: int, heads: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import h3_sparse_attention.sol_numerator_virtual_q as fused

    if torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError(f"GPU {gpu} is not SM80")
    if video_tokens < 64 or video_tokens % 64 or heads < 1:
        raise ValueError("video_tokens must be a positive multiple of 64; heads must be positive")
    torch.manual_seed(1000 + gpu)

    def inputs(sink_tokens: int):
        total = video_tokens + sink_tokens
        blocks = (total + 63) // 64
        q, k, v = [
            torch.randn(1, total, heads, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        ranges = torch.tensor(
            [[0, video_tokens], [video_tokens, total]],
            device="cuda", dtype=torch.int64,
        )
        mapping = torch.tensor(
            [0] * (video_tokens // 64) + [1] * (blocks - video_tokens // 64),
            device="cuda", dtype=torch.int64,
        )
        return (
            q, k, v, fused.build_virtual_anchors(q, ranges), ranges, mapping,
            fused.reduce_virtual_key_centroids(k),
        )

    data = {sink: inputs(sink) for sink in SINK_LENGTHS}

    def call(sink: int, ratio: float):
        return fused._fused_virtual(
            *data[sink], None, None, video_tokens, sink,
            force_local_blocks=False, _query_tokens=video_tokens,
            fused_topk_ratio=ratio,
        )[:, :video_tokens]

    fused._FUSED_COMPILED.clear()
    fused._FUSED_COMPILE_CALLS = 0
    fused._FUSED_COMPILE_SECONDS = 0.0
    os.environ["H3_SM80_RUNTIME_TOPK_RATIO"] = "1"
    dynamic_outputs = {}
    for sink in SINK_LENGTHS:
        for ratio in RATIOS:
            dynamic_outputs[(sink, ratio)] = call(sink, ratio).clone()
    torch.cuda.synchronize()
    dynamic_cache = fused.fused_compile_cache_stats().copy()
    if (dynamic_cache["entries"], dynamic_cache["compile_calls"]) != (1, 1):
        raise AssertionError(f"dynamic cache reuse failed: {dynamic_cache}")

    os.environ["H3_SM80_RUNTIME_TOPK_RATIO"] = "0"
    for sink in SINK_LENGTHS:
        for ratio in RATIOS:
            static_output = call(sink, ratio)
            if not torch.equal(dynamic_outputs[(sink, ratio)], static_output):
                raise AssertionError(f"static/dynamic output mismatch: {sink}, {ratio}")
    torch.cuda.synchronize()
    total_cache = fused.fused_compile_cache_stats().copy()
    if (total_cache["entries"], total_cache["compile_calls"]) != (3, 3):
        raise AssertionError(f"static-ratio cache count unexpected: {total_cache}")
    del dynamic_outputs

    cases = [(sink, ratio) for sink in SINK_LENGTHS for ratio in RATIOS]
    samples = {
        (sink, ratio, mode): []
        for sink, ratio in cases for mode in ("dynamic", "static")
    }

    def timed_call(sink: int, ratio: float, mode: str) -> float:
        os.environ["H3_SM80_RUNTIME_TOPK_RATIO"] = "1" if mode == "dynamic" else "0"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call(sink, ratio)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for sink, ratio in cases:
        for mode in ("dynamic", "static"):
            for _ in range(4):
                timed_call(sink, ratio, mode)

    rng = random.Random(2000 + gpu)
    for _ in range(repeats):
        rng.shuffle(cases)
        for sink, ratio in cases:
            modes = ("dynamic", "static") if rng.randrange(2) else ("static", "dynamic")
            for mode in modes:
                samples[(sink, ratio, mode)].append(timed_call(sink, ratio, mode))

    rows = []
    for sink, ratio in sorted(cases):
        dynamic = samples[(sink, ratio, "dynamic")]
        static = samples[(sink, ratio, "static")]
        rows.append({
            "sink_tokens": sink,
            "topk_ratio": ratio,
            "dynamic_median_ms": statistics.median(dynamic),
            "static_median_ms": statistics.median(static),
            "paired_median_ratio": statistics.median(
                a / b for a, b in zip(dynamic, static)
            ),
            "dynamic_ms": dynamic,
            "static_ms": static,
        })
    result = {
        "gpu": gpu,
        "name": torch.cuda.get_device_name(0),
        "video_tokens": video_tokens,
        "heads": heads,
        "repeats": repeats,
        "dynamic_cache": dynamic_cache,
        "total_cache": total_cache,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--video-tokens", type=int, default=DEFAULT_VIDEO_TOKENS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.gpu, args.repeats, args.output, args.video_tokens, args.heads)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu in args.gpus:
        path = args.output.with_name(f"{args.output.stem}.gpu{gpu}.json")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--gpu", str(gpu), "--repeats", str(args.repeats),
            "--video-tokens", str(args.video_tokens), "--heads", str(args.heads),
            "--output", str(path),
        ]
        processes.append((gpu, path, subprocess.Popen(command, env=env,
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       text=True)))
    results = []
    for gpu, path, process in processes:
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"GPU {gpu} failed:\n{stdout}{stderr}")
        results.append(json.loads(path.read_text()))
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")
    for result in results:
        print(f"GPU {result['gpu']}: " + ", ".join(
            f"{row['sink_tokens']}t/{row['topk_ratio']:.0%}="
            f"{100 * (row['paired_median_ratio'] - 1):+.2f}%"
            for row in result["rows"]
        ))


if __name__ == "__main__":
    main()
