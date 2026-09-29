"""Real-QKV Spark kernel attribution for the official BSA producer graft audit.

CUDA_VISIBLE_DEVICES=2 python scripts/probe_bsa_spark_reuse_ceiling_20260928.py --output OUT
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from comfy_kitchen.backends import cuda as ck

from profile_diffusers_vs_comfy_spark_20260925 import (
    CAPTURE, configure_environment, cuda_benchmark, load_capture, layout_kwargs,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata():
    extension = Path(ck._C.__file__).resolve()
    root = extension.parents[3]
    sources = [
        root / "comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu",
        root / "comfy_kitchen/backends/cuda/sage_attention/sol_attn_producer.cu",
    ]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return {
        "capture": {"path": str(CAPTURE), "sha256": sha256(CAPTURE)},
        "extension": {"path": str(extension), "sha256": sha256(extension)},
        "comfy_kitchen": {
            "root": str(root), "git_head": commit,
            "source_sha256": {str(path): sha256(path) for path in sources},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true",
                        help="append source hashes to an existing result without rerunning CUDA")
    args = parser.parse_args()
    if args.metadata_only:
        result = json.loads(args.output.read_text())
        capture = torch.load(CAPTURE, map_location="cpu", weights_only=False, mmap=True)
        video = int(capture["layout"]["video_tokens"])
        blocks = (result["shape"][1] + 63) // 64
        result["configuration"] = {
            "video_tokens": video, "sinks": [video // 64, blocks],
            "topk_ratio": 0.1, "tail_granularity": "block",
            "force_local_blocks": True, "reweight": True,
            "reblock_permutations": False, "direct_output_scatter": False,
        }
        del capture
        result["sources"] = source_metadata()
        args.output.write_text(json.dumps(result, indent=2))
        return
    configure_environment()
    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    layout = layout_kwargs(torch, data)
    video = int(layout["video_tokens"])
    blocks = (q.shape[1] + 63) // 64
    sinks = [video // 64, blocks]
    common = dict(
        video_tokens=video, topk_ratio=0.1, sink_blocks=sinks,
        sink_q=sinks, tail_granularity="block", force_local_blocks=True,
    )
    calls = {
        "spark_full": lambda: ck.spark_attn(q, k, v, reweight=True, **common),
        "spark_topk_only": lambda: ck.spark_attn(q, k, v, reweight=False, **common),
    }
    result = {
        "scope": "captured real post-RoPE QKV; route and exact have no reblock permutation",
        "shape": list(q.shape), "gpu": torch.cuda.get_device_name(),
        "configuration": {
            "video_tokens": video, "sinks": sinks, "topk_ratio": 0.1,
            "tail_granularity": "block", "force_local_blocks": True,
            "reweight": True, "reblock_permutations": False,
            "direct_output_scatter": False,
        },
        "sources": source_metadata(),
        "timings": {}, "kernels": {},
    }
    for name, call in calls.items():
        result["timings"][name] = cuda_benchmark(torch, call, warmup=5, iterations=12)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        result["kernels"][name] = sorted([
            {"name": row.key, "count": int(row.count),
             "cuda_ms": float(row.self_device_time_total) / 1000}
            for row in prof.key_averages() if row.self_device_time_total > 0
        ], key=lambda row: row["cuda_ms"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["median_ms"] for k, v in result["timings"].items()}))


if __name__ == "__main__":
    main()
