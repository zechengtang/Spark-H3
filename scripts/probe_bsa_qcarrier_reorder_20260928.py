"""Lower-bound timing of the required post-reblock Q carrier reorder."""

import argparse
import json
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl

from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations
from profile_diffusers_vs_comfy_spark_20260925 import (
    configure_environment, load_capture, layout_kwargs,
)


@triton.jit
def gather_qi(src, perm, dst, video: tl.constexpr, heads: tl.constexpr,
              tokens: tl.constexpr, block: tl.constexpr):
    ix = tl.program_id(0) * block + tl.arange(0, block)
    h = tl.program_id(1)
    t = ix // 128
    d = ix % 128
    valid = t < video
    old_t = tl.load(perm + h * video + t, mask=valid, other=0)
    value = tl.load(src + (old_t * heads + h) * 128 + d, mask=valid, other=0)
    tl.store(dst + (t * heads + h) * 128 + d, value, mask=valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_environment()
    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    controller = ComfySparkController(ComfySparkConfig())
    perm, _, _, _ = build_comfy_reblock_permutations(controller, q, k, layout)
    del q, k, v, data
    torch.cuda.empty_cache()
    tokens = layout.sequence_length
    video = layout.video_tokens
    heads = perm.shape[1]
    src = torch.randint(-127, 128, (tokens, heads, 128), dtype=torch.int8, device="cuda")
    dst = torch.empty_like(src)
    perm = perm.contiguous()
    call = lambda: gather_qi[(triton.cdiv(video * 128, 8192), heads)](
        src, perm, dst, video, heads, tokens, 8192, num_warps=8)
    for _ in range(5): call()
    torch.cuda.synchronize()
    values = []
    for _ in range(30):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); call(); b.record(); b.synchronize(); values.append(a.elapsed_time(b))
    # Spot-check the index contract for three real heads and rows.
    checks = []
    for h in (0, heads // 2, heads - 1):
        for t in (0, video // 2, video - 1):
            old = int(perm[0, h, t])
            checks.append(bool(torch.equal(dst[t, h], src[old, h])))
    result = {"shape": [tokens, heads, 128], "video_tokens": video,
              "carrier_bytes": int(src.numel()), "moves_bytes": int(2 * video * heads * 128),
              "gpu": torch.cuda.get_device_name(), "correct_sample_checks": all(checks),
              "median_ms": statistics.median(values), "samples_ms": values,
              "scope": "INT8 Q carrier gather only; excludes qs scale, centroid/qmean, threshold, and producer quantization"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"median_ms": result["median_ms"], "correct": result["correct_sample_checks"]}))


if __name__ == "__main__":
    main()
