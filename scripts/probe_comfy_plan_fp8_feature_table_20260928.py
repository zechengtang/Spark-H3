"""Screen optional FP8 feature-table traffic in one captured 10s reblock plan."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("H3_METRIC_FACTOR", "cholesky")
os.environ.setdefault("H3_LMV2_COS_PRECISION", "fp16")
os.environ.setdefault("H3_LMV2_SMALL_PROXY_FAST", "1")
os.environ.setdefault("H3_LMV2_FUSED_NODE", "1")
os.environ.setdefault("H3_LMV2_COS_FAST", "1")
os.environ.setdefault("H3_LMV2_GROUP1_FAST", "1")
os.environ.setdefault("H3_TEMPORAL_MIN_FRAMES", "0")
os.environ.setdefault("H3_TEMPORAL_CHUNK_FRAMES", "0")
os.environ.setdefault("H3_LMV2_TERMINAL_LEAVES", "16")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, layout_kwargs
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_plan_fp8_feature_table_20260928")
MODE = os.environ.get("H3_LMV2_FP8_FEATURES", "0")


def main():
    if MODE not in ("0", "1"):
        raise ValueError("H3_LMV2_FP8_FEATURES must be 0 or 1")
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

    for _ in range(4):
        result = work()
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = work()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
    qperm, _, kperm, _ = result
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"q": qperm.cpu(), "k": kperm.cpu()}, ROOT / f"permutations_fp8_{MODE}.pt")
    summary = dict(mode=MODE, shape=list(q.shape), video_tokens=layout.video_tokens,
                   median_ms=statistics.median(times), mean_ms=statistics.fmean(times),
                   times_ms=times, q_sha256=hashlib.sha256(qperm.cpu().numpy().tobytes()).hexdigest(),
                   k_sha256=hashlib.sha256(kperm.cpu().numpy().tobytes()).hexdigest())
    (ROOT / f"summary_fp8_{MODE}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
