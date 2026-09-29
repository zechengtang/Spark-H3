"""Check deterministic all-exact routing against a permissive tau threshold."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck

from profile_diffusers_vs_comfy_spark_20260925 import load_capture


def time_call(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(6):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=37897)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, q, k, v = load_capture(torch)
    q, k, v = (x[:, :args.tokens].contiguous() for x in (q, k, v))
    nblocks = (args.tokens + 63) // 64
    calls = {
        "threshold_neg1e9": lambda: ck.sol_attn(q, k, v, tau=-1e9, tail=False),
        "all_key_blocks_sink": lambda: ck.sol_attn(
            q, k, v, sink_blocks=[0, nblocks], tail=False),
        "tau_zero": lambda: ck.sol_attn(q, k, v, tau=0.0, tail=False),
    }
    with torch.inference_mode():
        outputs = {name: fn() for name, fn in calls.items()}
        ref = outputs["all_key_blocks_sink"]
        comparisons = {}
        for name, out in outputs.items():
            delta = (out.float() - ref.float()).abs()
            comparisons[name] = {
                "bitwise_equal_to_all_sink": bool(torch.equal(out, ref)),
                "mean_abs_difference": delta.mean().item(),
                "max_abs_difference": delta.max().item(),
            }
            del delta
        del outputs, ref
        timings = {name: time_call(fn) for name, fn in calls.items()}
    result = {
        "tokens": args.tokens,
        "blocks": nblocks,
        "comparisons": comparisons,
        "timings_ms": timings,
        "median_ms": {name: statistics.median(v) for name, v in timings.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("tokens", "blocks", "comparisons", "median_ms")}, indent=2))


if __name__ == "__main__":
    main()
