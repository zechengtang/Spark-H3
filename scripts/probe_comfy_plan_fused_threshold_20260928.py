"""A/B/A full-plan test for fused-node maximum size on 10s capture."""

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


OUTPUT = Path("/autodl-fs/data/h3_experiments/comfyui_plan_fused_threshold_20260928/results.json")


def run():
    torch.set_num_threads(4)
    data, q, k, _ = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    records = {}
    original = landmark_v2_fused_node.FUSED_NODE_MAX_TOKENS
    try:
        for name, threshold in (("baseline_a", 1024), ("candidate_512", 512),
                                ("candidate_256", 256), ("baseline_b", 1024)):
            landmark_v2_fused_node.FUSED_NODE_MAX_TOKENS = threshold
            controller = ComfySparkController(ComfySparkConfig(
                topk_ratio=.1, landmark_tree_v2_children=16,
                landmark_tree_v2_midpoint_direction_mode="legacy"))
            controller.evaluation_index = 5

            def work():
                return build_comfy_reblock_permutations(controller, q, k, layout)

            for _ in range(5):
                result = work()
            torch.cuda.synchronize()
            samples = []
            for _ in range(12):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = work()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            outputs = (result[0].clone(), result[2].clone())
            if name == "baseline_a":
                baseline = outputs
            records[name] = dict(
                median_ms=statistics.median(samples), samples_ms=samples,
                same_query=torch.equal(outputs[0], baseline[0]),
                same_key=torch.equal(outputs[1], baseline[1]),
                differing_query=int((outputs[0] != baseline[0]).sum().item()),
                differing_key=int((outputs[1] != baseline[1]).sum().item()))
    finally:
        landmark_v2_fused_node.FUSED_NODE_MAX_TOKENS = original
    payload = dict(shape=list(q.shape), video_tokens=layout.video_tokens, records=records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    run()
