#!/usr/bin/env python3
"""Paired ComfyUI Spark output-layout microbenchmark and safe decode-stage runner.

The existing attention output is [video, before, after]. Restore ComfyUI's
[before, video, after] order either with the current inverse index_select or
with contiguous segment concatenation. This does not alter Q/K/V or attention.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


SHAPES = {"5s": (37897, 37296), "10s": (73573, 72576)}
EXPERIMENT = Path(
    "/autodl-fs/data/h3_experiments/"
    "diffusers_spark_reweight_components_10prompt_10s768p_20260926"
)
OUTPUT = Path(
    "/autodl-fs/data/h3_experiments/"
    "comfyui_segmented_output_profile_20260926"
)


def benchmark(label: str, output_path: Path) -> None:
    import torch

    tokens, video_tokens = SHAPES[label]
    before = tokens - video_tokens
    after = 0  # The measured T2VA layout has target video last.
    device = torch.device("cuda")
    source = torch.randn(1, tokens, 56, 128, device=device, dtype=torch.bfloat16)
    permutation = torch.cat((
        torch.arange(before, before + video_tokens, device=device),
        torch.arange(before, device=device),
        torch.arange(before + video_tokens, tokens, device=device),
    ))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(tokens, device=device)

    def gather():
        return source.index_select(1, inverse)

    def segmented():
        if after == 0:
            return torch.cat((source[:, video_tokens:], source[:, :video_tokens]), dim=1)
        return torch.cat((source[:, video_tokens:video_tokens + before],
                          source[:, :video_tokens],
                          source[:, video_tokens + before:]), dim=1)

    with torch.inference_mode():
        if not torch.equal(gather(), segmented()):
            raise AssertionError("segmented restore differs from inverse gather")
        for _ in range(10):
            gather()
            segmented()
        torch.cuda.synchronize()
        samples = {"index_select": [], "segmented_cat": []}
        operations = {"index_select": gather, "segmented_cat": segmented}
        for pair in range(72):
            order = ("index_select", "segmented_cat") if pair % 2 else (
                "segmented_cat", "index_select")
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = operations[name]()
                end.record()
                end.synchronize()
                if pair >= 8:
                    samples[name].append(start.elapsed_time(end))
                del result
    baseline = statistics.median(samples["index_select"])
    candidate = statistics.median(samples["segmented_cat"])
    result = {
        "status": "complete", "label": label, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tokens": tokens, "video_tokens": video_tokens, "before_tokens": before,
        "after_tokens": after, "bitwise_equal": True, "pairs": 64,
        "index_select_median_ms": baseline, "segmented_cat_median_ms": candidate,
        "median_delta_ms_per_sparse_call": candidate - baseline,
        "estimated_delta_seconds_784_sparse_calls": (candidate - baseline) * 784 / 1000,
        "samples_ms": samples,
        "scope": "isolated output restore only; projection and full denoise excluded",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples_ms"}), flush=True)


def wait_for_window(timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    markers = (EXPERIMENT / "decode_gpu4.json", EXPERIMENT / "decode_gpu5.json")
    while time.monotonic() < deadline:
        if all(path.is_file() for path in markers):
            # Other decoders must still be running. Otherwise the parent may
            # start the six-GPU scorer immediately; defer until it is done.
            remaining = [EXPERIMENT / f"decode_gpu{gpu}.json" for gpu in range(4)]
            if any(not path.is_file() for path in remaining):
                print("GPU4/5 decode complete; running during remaining decode", flush=True)
                break
        results = EXPERIMENT / "results.json"
        if results.is_file():
            state = json.loads(results.read_text()).get("status")
            if state == "quality_complete":
                print("No safe decode window; running after scoring", flush=True)
                break
        time.sleep(10)
    else:
        raise TimeoutError("no safe GPU4/5 window before timeout")

    jobs = []
    for gpu, label in ((4, "5s"), (5, "10s")):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        destination = OUTPUT / f"{label}_gpu{gpu}.json"
        jobs.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "bench", label,
             "--output", str(destination)], env=env))
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(f"segmented output benchmark failed: {codes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("bench", "wait"))
    parser.add_argument("label", nargs="?", choices=tuple(SHAPES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    if args.command == "bench":
        if args.label is None or args.output is None:
            parser.error("bench requires label and --output")
        benchmark(args.label, args.output)
    else:
        wait_for_window(args.timeout_seconds)
