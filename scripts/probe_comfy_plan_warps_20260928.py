"""Test fused-node warp count without changing production defaults."""

import json
from pathlib import Path
import statistics
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, layout_kwargs
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations
from h3_sparse_attention import landmark_v2_fused_node


OUTPUT = Path("/autodl-fs/data/h3_experiments/comfyui_plan_warps_20260928/results.json")


def run():
    torch.set_num_threads(4)
    data, q, k, _ = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    modes = (2, 4, 8)
    timings, outputs = {}, {}
    original = landmark_v2_fused_node.FUSED_NODE_WARPS
    try:
        for warps in modes:
            landmark_v2_fused_node.FUSED_NODE_WARPS = str(warps)
            controller = ComfySparkController(ComfySparkConfig(
                topk_ratio=.1, landmark_tree_v2_children=16,
                landmark_tree_v2_midpoint_direction_mode="legacy"))
            controller.evaluation_index = 5
            def work():
                return build_comfy_reblock_permutations(controller, q, k, layout)
            for _ in range(5):
                result = work()
            torch.cuda.synchronize()
            samples=[]
            for _ in range(16):
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record()
                result=work()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            outputs[warps] = (result[0].clone(), result[2].clone())
            timings[warps] = dict(median_ms=statistics.median(samples),
                                  mean_ms=statistics.fmean(samples), samples_ms=samples)
        matches={warps: dict(query=torch.equal(outputs[warps][0], outputs[2][0]),
                             key=torch.equal(outputs[warps][1], outputs[2][1]))
                 for warps in modes}
    finally:
        landmark_v2_fused_node.FUSED_NODE_WARPS = original
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(shape=list(q.shape),video_tokens=layout.video_tokens,
                 timings=timings,matches_baseline=matches)
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__":
    run()
