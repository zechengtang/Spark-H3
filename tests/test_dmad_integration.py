from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_dmad_lora_to_comfyui", ROOT / "scripts" / "convert_dmad_lora_to_comfyui.py"
)
CONVERTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONVERTER)


def pair(a, b):
    return {"A": torch.tensor(a, dtype=torch.float32), "B": torch.tensor(b, dtype=torch.float32)}


def test_destination_block_names():
    assert CONVERTER.destination_block("transformer_blocks.3") == "blocks.3"
    assert (
        CONVERTER.destination_block("token_refiner.refiner_blocks.1")
        == "token_refiner.blocks.1"
    )


def test_convert_fuses_qkv_and_swaps_swiglu_halves():
    block = "transformer_blocks.0"
    pairs = {
        f"{block}.attn.to_q": pair([[1, 2]], [[1], [2]]),
        f"{block}.attn.to_k": pair([[3, 4]], [[3], [4]]),
        f"{block}.attn.to_v": pair([[5, 6]], [[5], [6]]),
        f"{block}.attn.to_out.0": pair([[7, 8]], [[7], [8]]),
        f"{block}.ff.net.0.proj": pair([[9, 10]], [[1], [2], [3], [4]]),
        f"{block}.ff.net.2": pair([[11, 12]], [[9], [10]]),
    }
    output = CONVERTER.convert(pairs, alpha=8.0)
    prefix = "diffusion_model.blocks.0"
    assert torch.equal(
        output[f"{prefix}.attn.qkv_proj.lora_A.weight"],
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
    )
    assert torch.equal(
        output[f"{prefix}.attn.qkv_proj.lora_B.weight"],
        torch.block_diag(
            torch.tensor([[1], [2]], dtype=torch.float32),
            torch.tensor([[3], [4]], dtype=torch.float32),
            torch.tensor([[5], [6]], dtype=torch.float32),
        ),
    )
    assert output[f"{prefix}.attn.qkv_proj.alpha"].item() == 24.0
    assert torch.equal(
        output[f"{prefix}.mlp.fc1.lora_B.weight"],
        torch.tensor([[3], [4], [1], [2]], dtype=torch.float32),
    )
