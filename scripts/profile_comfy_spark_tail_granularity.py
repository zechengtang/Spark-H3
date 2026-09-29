"""Single captured attention layer, shared KV reweight with two consumers.

No reblock or video generation: timings and errors must not be called video
latency, video PSNR, or VBench. The two calls have identical route/sink settings.
"""
import argparse
import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, cuda_benchmark
from probe_comfy_summary_gemm_20260925 import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    video = int(data['layout']['video_tokens'])
    sinks = [video // 64, (q.shape[1] + 63) // 64]
    calls = {grain: (lambda g=grain: ck.spark_attn(
        q, k, v, video_tokens=video, topk_ratio=.1, anchor_dtype=torch.float32,
        sink_blocks=sinks, sink_q=sinks, tail_granularity=g))
        for grain in ('block', 'query')}
    result = dict(shape=list(q.shape), video_tokens=video, reblock=False,
                  gpu=torch.cuda.get_device_name(), timings={}, attention_vs_dense={})
    order = list(calls)
    if args.reverse: order.reverse()
    for name in order:
        result['timings'][name] = cuda_benchmark(torch, calls[name])
        print(name, result['timings'][name], flush=True)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        dense = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2).contiguous()
    for name, fn in calls.items():
        result['attention_vs_dense'][name] = metrics(dense, fn())
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
