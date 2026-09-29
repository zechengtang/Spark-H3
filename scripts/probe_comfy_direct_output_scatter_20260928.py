"""Bitwise and speed A/B for Spark's direct video-first output scatter."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import (
    cuda_benchmark, load_capture, layout_kwargs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_C", args.library)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    ck._C = extension
    data, q, k, v = load_capture(torch)
    layout = layout_kwargs(torch, data)
    video_tokens = int(layout["video_tokens"])
    video_start = int(layout["permutation"][0].item())
    blocks = (q.shape[1] + 63) // 64
    sinks = [video_tokens // 64, blocks]
    common = dict(video_tokens=video_tokens, topk_ratio=0.1,
                  sink_blocks=sinks, sink_q=sinks, tail_granularity="block",
                  reweight=False, force_local_blocks=True)

    def old():
        return ck.spark_attn(q, k, v, **common).index_select(
            1, layout["inverse_permutation"])

    def direct():
        return ck.spark_attn(q, k, v, **common, video_start=video_start)

    old_out = old()
    new_out = direct()
    torch.cuda.synchronize()
    equal = torch.equal(old_out, new_out)
    calls = {"old_with_restore": old, "direct_scatter": direct}
    names = list(calls)
    if args.reverse:
        names.reverse()
    timings = {name: cuda_benchmark(torch, calls[name]) for name in names}
    result = {"shape": list(q.shape), "video_start": video_start,
              "video_tokens": video_tokens, "bitwise_equal": equal,
              "max_abs": (old_out.float() - new_out.float()).abs().max().item(),
              "timings": timings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"bitwise_equal": equal,
                      "median_ms": {key: value["median_ms"]
                                    for key, value in timings.items()}}, indent=2))


if __name__ == "__main__":
    main()
