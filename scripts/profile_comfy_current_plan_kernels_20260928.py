"""One warmed 10s ComfyUI reblock-plan kernel trace on captured real Q/K."""

import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, layout_kwargs
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations


OUTPUT = Path("/autodl-fs/data/h3_experiments/comfyui_current_plan_trace_20260928")


def main():
    torch.set_num_threads(4)
    data, q, k, _ = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    controller = ComfySparkController(ComfySparkConfig(
        topk_ratio=.1, global_anchor_dtype="float32",
        landmark_tree_v2_children=16,
        landmark_tree_v2_midpoint_direction_mode="legacy"))
    controller.evaluation_index = 5
    def work():
        return build_comfy_reblock_permutations(controller, q, k, layout)
    for _ in range(5):
        work()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                             torch.profiler.ProfilerActivity.CUDA]) as prof:
        work()
        torch.cuda.synchronize()
    rows=[]
    for event in prof.key_averages():
        gpu_us=getattr(event,"self_device_time_total",0) or getattr(event,"self_cuda_time_total",0)
        if gpu_us>0:
            rows.append(dict(name=event.key,calls=event.count,gpu_ms=float(gpu_us)/1000))
    rows.sort(key=lambda row: row["gpu_ms"], reverse=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(OUTPUT/"trace.json"))
    (OUTPUT/"summary.json").write_text(json.dumps(dict(
        shape=list(q.shape),video_tokens=layout.video_tokens,rows=rows),indent=2)+"\n")
    print(json.dumps(rows[:25],indent=2),flush=True)


if __name__ == "__main__":
    main()
