"""Matched real-QKV kernel-only A/B: official Sol TopK vs Spark route without extras."""

import argparse
import json
from pathlib import Path
import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import (
    configure_environment, cuda_benchmark, load_capture, layout_kwargs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    configure_environment()
    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    layout = layout_kwargs(torch, data)
    first = layout["video_tokens"] // 64
    blocks = (q.shape[1] + 63) // 64
    sinks = [first, blocks]

    calls = {
        "official_sol_topk10_block": lambda: ck.sol_attn(
            q, k, v, topk_ratio=0.1, sink_blocks=sinks, sink_q=sinks,
            tail_granularity="block", token_aug=0,
        ),
        "spark_topk10_block_no_reblock_no_reweight": lambda: ck.spark_attn(
            q, k, v, video_tokens=layout["video_tokens"], topk_ratio=0.1,
            sink_blocks=sinks, sink_q=sinks, tail_granularity="block",
            force_local_blocks=True, reweight=False,
        ),
    }
    names = list(calls)
    if args.reverse:
        names.reverse()
    result = {"shape": list(q.shape), "sinks": sinks, "gpu": torch.cuda.get_device_name(),
              "scope": "single real QKV, same sink and forced-local policy; no reblock or reweight",
              "timings": {}}
    for name in names:
        result["timings"][name] = cuda_benchmark(torch, calls[name], warmup=5, iterations=20)
        print(name, result["timings"][name]["median_ms"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
