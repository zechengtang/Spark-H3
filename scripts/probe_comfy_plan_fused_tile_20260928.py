"""Fused reblock node tile/row scheduling A/B on a real 10s capture."""

import json
from pathlib import Path
import statistics
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, layout_kwargs
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations
from h3_sparse_attention import landmark_v2_fused_node as fused


OUTPUT = Path("/autodl-fs/data/h3_experiments/comfyui_plan_fused_tile_20260928/results.json")


def run():
    torch.set_num_threads(4)
    data,q,k,_ = load_capture(torch)
    layout=ComfyPackedLayout(**layout_kwargs(torch,data))
    modes=((64,1),(32,1),(128,1),(64,2),(64,4))
    originals=(fused._default_block_t,fused._default_rows_per_step)
    records={};baseline=None
    try:
        for block_t,rows_per_step in modes:
            fused._default_block_t=lambda n,tile=block_t:tile
            fused._default_rows_per_step=lambda n,rows=rows_per_step:rows
            controller=ComfySparkController(ComfySparkConfig(
                topk_ratio=.1,landmark_tree_v2_children=16,
                landmark_tree_v2_midpoint_direction_mode="legacy"))
            controller.evaluation_index=5
            def work():
                return build_comfy_reblock_permutations(controller,q,k,layout)
            for _ in range(5):result=work()
            torch.cuda.synchronize()
            samples=[]
            for _ in range(12):
                begin=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                begin.record();result=work();end.record();end.synchronize()
                samples.append(begin.elapsed_time(end))
            if baseline is None:
                baseline=(result[0].clone(),result[2].clone())
            records[f"tile{block_t}_rows{rows_per_step}"]=dict(
                median_ms=statistics.median(samples),samples_ms=samples,
                same_query=torch.equal(result[0],baseline[0]),
                same_key=torch.equal(result[2],baseline[1]))
    finally:
        fused._default_block_t,fused._default_rows_per_step=originals
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(shape=list(q.shape),video_tokens=layout.video_tokens,records=records)
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":run()
