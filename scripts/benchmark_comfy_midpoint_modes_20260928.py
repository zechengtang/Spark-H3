"""Quick ComfyUI Spark midpoint-mode A/B on a captured 10s 768p layer."""

import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, layout_kwargs
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations


OUTPUT = Path("/autodl-fs/data/h3_experiments/comfyui_midpoint_quick_ab_20260928/results.json")


def main():
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    started = time.monotonic()
    data, q, k, _ = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    work = {}
    outputs = {}
    for mode in ("legacy", "fused"):
        controller = ComfySparkController(ComfySparkConfig(
            topk_ratio=0.1,
            global_anchor_dtype="float32",
            landmark_tree_v2_children=16,
            landmark_tree_v2_midpoint_direction_mode=mode,
        ))
        controller.evaluation_index = 5
        work[mode] = lambda controller=controller: build_comfy_reblock_permutations(
            controller, q, k, layout
        )
        for _ in range(5):
            result = work[mode]()
        torch.cuda.synchronize()
        outputs[mode] = (result[0].clone(), result[2].clone())

    samples = {mode: [] for mode in work}
    # ABBA order cancels most monotonic GPU clock and thermal drift.
    for _ in range(12):
        for mode in ("legacy", "fused", "fused", "legacy"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            work[mode]()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end))

    comparison = {}
    for label, index in (("query", 0), ("key", 1)):
        a, b = outputs["legacy"][index], outputs["fused"][index]
        comparison[label] = {
            "equal": bool(torch.equal(a, b)),
            "different_elements": int(torch.count_nonzero(a != b).item()),
            "total_elements": a.numel(),
        }
    timings = {
        mode: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for mode, values in samples.items()
    }
    payload = {
        "scope": "one ComfyUI Spark reblock plan, captured real 10s 768p Q/K",
        "shape": list(q.shape),
        "video_tokens": layout.video_tokens,
        "gpu": torch.cuda.get_device_name(0),
        "warmup_per_mode": 5,
        "samples_per_mode": 24,
        "timings": timings,
        "fused_over_legacy_median_ratio": timings["fused"]["median_ms"] / timings["legacy"]["median_ms"],
        "output_comparison": comparison,
        "wall_seconds": time.monotonic() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
