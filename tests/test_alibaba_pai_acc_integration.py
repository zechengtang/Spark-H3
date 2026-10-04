import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "spark_h3_alibaba_pai_acc_fl2va_8step_lora_5p2s_t2va.json"


def _load_pdd_module():
    path = ROOT / "scripts" / "minimax_h3_pdd.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_pdd", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pdd_plan_covers_one_trained_block():
    pdd = _load_pdd_module()
    grid = pdd.pdd_time_grid(12.0, 32)
    steps = grid.diff()
    plan = pdd.pdd_sampling_plan(steps, start=8, block_size=4)

    assert grid.shape == (33,)
    assert torch.all(grid[1:] > grid[:-1])
    assert plan.shape == (1, 32)
    assert plan.sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(plan).item() == 4
    assert torch.count_nonzero(plan[:, :8]).item() == 0
    assert torch.count_nonzero(plan[:, 12:]).item() == 0


def test_workflow_uses_pdd_node_sigmas_and_plain_euler():
    graph = json.loads(WORKFLOW.read_text())
    nodes = {node["id"]: node for node in graph["nodes"]}
    by_type = {node["type"]: node for node in graph["nodes"]}

    apply = by_type["MiniMaxH3PDDAccApply"]
    sampler = by_type["KSamplerSelect"]
    advanced = by_type["SamplerCustomAdvanced"]
    image_to_video = by_type["MiniMaxH3ImageToVideo"]

    assert "BasicScheduler" not in by_type
    assert apply["widgets_values"][:5] == [
        "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        "8",
        1.0,
        1.0,
        "raise",
    ]
    assert sampler["widgets_values"] == ["euler"]
    assert image_to_video["widgets_values"][1:4] == [1280, 704, 124]

    sigma_link = advanced["inputs"][3]["link"]
    link = next(link for link in graph["links"] if link[0] == sigma_link)
    assert link[1:6] == [apply["id"], 1, advanced["id"], 3, "SIGMAS"]

    model_link = by_type["MiniMaxH3SparkAttentionSM120"]["inputs"][0]["link"]
    link = next(link for link in graph["links"] if link[0] == model_link)
    assert link[1] == apply["id"]
    assert nodes[link[3]]["type"] == "MiniMaxH3SparkAttentionSM120"
