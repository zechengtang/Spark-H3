#!/usr/bin/env python3
"""Isolated ComfyUI Spark global-layout conversion cost, without kernel edits.

The paired control uses the *same* current QKV producer, projection, RMS/RoPE,
chunk size, token count and output buffers, but receives natural-order tokens.
The measured delta is the current video-last -> video-first producer penalty;
the full producer time is NOT attributed to layout. The inverse gather is timed
separately. Reblock's internal video ordering is outside both measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

COMFY = Path("/autodl-fs/data/h3_repos/ComfyUI")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(REPO))

import torch
from comfyui_nodes import _native_spark_qkv


SHAPES = {"5s": (37897, 37296), "10s": (73573, 72576)}
SPARSE_CALLS = 16 * 49  # ComfyUI's 20 evaluations, first 4 dense, layer 0 dense.


def measure(fn, repeat=16, discard=4):
    samples = []
    for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        if i >= discard:
            samples.append(start.elapsed_time(stop))
        del result
    return samples


def run_one(label, denoise_seconds):
    tokens, video_tokens = SHAPES[label]
    condition_tokens = tokens - video_tokens
    device = torch.device("cuda")
    torch.manual_seed(42)
    x = torch.randn(tokens, 5376, device=device, dtype=torch.bfloat16)
    rope = torch.randn(1, tokens, 1, 64, 2, 2, device=device, dtype=torch.float32)
    attn = SimpleNamespace(
        heads=56, head_dim=128,
        qkv_proj=torch.nn.Linear(5376, 56 * 128 * 3, bias=False,
                                 dtype=torch.bfloat16, device=device),
        q_norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16, device=device), eps=1e-5),
        k_norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16, device=device), eps=1e-5),
    )
    actual = torch.cat((torch.arange(condition_tokens, tokens, device=device),
                        torch.arange(condition_tokens, device=device)))
    identity = torch.arange(tokens, device=device)
    inverse = torch.empty_like(actual)
    inverse[actual] = identity

    def producer(reordered):
        permutation = actual if reordered else identity
        return _native_spark_qkv(
            attn, x, rope, permutation,
            video_start=condition_tokens if reordered else 0,
            video_tokens=video_tokens if reordered else tokens)

    with torch.inference_mode():
        original = producer(False)
        reordered = producer(True)
        bitwise = [torch.equal(a.index_select(1, actual), b)
                   for a, b in zip(original, reordered, strict=True)]
        if not all(bitwise):
            raise AssertionError(f"producer reorder mismatch: {label}: {bitwise}")
        del original, reordered
        torch.cuda.synchronize()
        for _ in range(3):
            producer(False)
            producer(True)
        torch.cuda.synchronize()
        samples = {False: [], True: []}
        for i in range(64):
            order = (False, True) if i % 2 else (True, False)
            for reordered in order:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                result = producer(reordered)
                stop.record()
                stop.synchronize()
                if i >= 8:
                    samples[reordered].append(start.elapsed_time(stop))
                del result
        output = torch.randn(1, tokens, 56, 128, device=device, dtype=torch.bfloat16)
        inverse_samples = measure(lambda: output.index_select(1, inverse))
    natural_ms = statistics.median(samples[False])
    reordered_ms = statistics.median(samples[True])
    paired_differences = [candidate - control for control, candidate in
                          zip(samples[False], samples[True], strict=True)]
    producer_increment_ms = statistics.median(paired_differences)
    inverse_ms = statistics.median(inverse_samples)
    total_ms = producer_increment_ms + inverse_ms
    return dict(
        label=label, tokens=tokens, video_tokens=video_tokens,
        condition_tokens=condition_tokens, producer_chunk=16384,
        numerical_equivalence=bitwise,
        producer_natural_ms=natural_ms, producer_reordered_ms=reordered_ms,
        producer_layout_increment_ms=producer_increment_ms,
        inverse_permutation_ms=inverse_ms,
        combined_layout_ms_per_sparse_call=total_ms,
        sparse_calls=SPARSE_CALLS, sparse_evaluations=16, model_evaluations=20,
        estimated_layout_seconds_per_denoise=total_ms * SPARSE_CALLS / 1000,
        reference_denoise_seconds=denoise_seconds,
        estimated_fraction_of_full_denoise=total_ms * SPARSE_CALLS / 1000 / denoise_seconds,
        estimated_layout_ms_per_sparse_evaluation=total_ms * 49,
        reference_ms_per_model_evaluation=denoise_seconds * 1000 / 20,
        samples=dict(natural=samples[False], reordered=samples[True],
                     paired_producer_increment=paired_differences,
                     inverse=inverse_samples),
        scope=("Paired same-shape isolated producer delta plus independent inverse gather; "
               "excludes reblock internal video ordering, attention, output projection. "
               "Full-denoise fractions are estimates using separately measured same-code runs."),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", choices=tuple(SHAPES), default=list(SHAPES))
    parser.add_argument("--denoise-5s", type=float, required=True,
                        help="Fresh same-code 5s full denoise seconds; do not reuse removed hybrid ablation")
    parser.add_argument("--denoise-10s", type=float, required=True,
                        help="Fresh same-code 10s full denoise seconds")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = [run_one(label, {"5s": args.denoise_5s, "10s": args.denoise_10s}[label])
               for label in args.labels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="complete", results=results), indent=2) + "\n")
    print(json.dumps({"results": [{key: value for key, value in result.items()
                                    if key != "samples"} for result in results]}, indent=2))
