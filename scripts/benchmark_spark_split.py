"""Replay captured GPU-resident QKV through both supported Spark schedules.

Includes current-input planning, producer and BSA/reweight; excludes linear
projection, CPU transfers and VAE. Set H3_SPARK_SPLIT_LIBRARY before running.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemes", nargs="+", choices=["keep_k", "drop_k"],
                        default=["keep_k", "drop_k"])
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.chunk < 64 or args.chunk % 64:
        parser.error("repetitions must be positive; chunk must be a positive multiple of 64")

    from comfy_kitchen.backends import cuda as ck
    from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
    from comfyui_spark_split import spark_split_attention

    torch.set_grad_enabled(False)
    saved = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    qkv = saved["qkv"].cuda()
    t, h = qkv.shape[0], qkv.shape[1] // 384
    ids = torch.arange(t, device="cuda")
    video = saved["video_tokens"]
    layout = ComfyPackedLayout(ids, ids, tuple(saved["grid"]), video, t,
                              torch.zeros(video, 3, device="cuda"), saved["video_start"])
    rope = saved["rope_freqs"].cuda()
    weights = tuple(saved[key].cuda() for key in ("q_norm_weight", "k_norm_weight"))
    stale = None if args.bootstrap else tuple(saved[key].cuda() for key in ("kmean", "vscale"))
    controllers = {name: ComfySparkController(ComfySparkConfig(topk_mode="topk_ratio", topk_ratio=.2))
                   for name in args.schemes}
    for controller in controllers.values():
        controller.direct_output = True

    def provider(kind, *, indices=None, start=None, stop=None):
        raw = qkv.index_select(0, indices) if indices is not None else qkv[start:stop]
        q, k, v = raw.split(h * 128, dim=-1)
        return {"q": q, "k": k, "kv": raw[:, h * 128:], "qkv": raw, "qv": (q, v)}[kind]

    def run(name):
        return spark_split_attention(controllers[name], provider, t, h, rope, weights,
            saved["rope_eps"], layout, stale, scheme=name, chunk=args.chunk)

    timings = {name: [] for name in args.schemes}
    peaks = {name: [] for name in args.schemes}
    for name in args.schemes:
        for _ in range(3):
            result = run(name)
            del result
    torch.cuda.synchronize()
    for iteration in range(args.repetitions):
        order = args.schemes if iteration % 2 == 0 else list(reversed(args.schemes))
        for name in order:
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            result = run(name)
            b.record()
            b.synchronize()
            timings[name].append(a.elapsed_time(b))
            peaks[name].append((torch.cuda.max_memory_allocated() - base) / 2**20)
            del result

    hashes = {}
    for name in args.schemes:
        out, stats = run(name)
        hashes[name] = [hashlib.sha256(x.cpu().view(torch.uint8).numpy()).hexdigest()
                        for x in (out, *stats)]
        del out, stats
    report = dict(capture=str(args.capture), shape=[t, h, 128], bootstrap=args.bootstrap,
                  chunk=args.chunk, extension=ck._C.__file__, measurements={})
    for name in args.schemes:
        report["measurements"][name] = dict(samples_ms=timings[name], peak_delta_mib=peaks[name],
            median_ms=statistics.median(timings[name]),
            median_peak_delta_mib=statistics.median(peaks[name]), output_stats_sha256=hashes[name])
    report["schemes_bitwise_equal"] = all(value == next(iter(hashes.values())) for value in hashes.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
