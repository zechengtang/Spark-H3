#!/usr/bin/env python3
"""Convert a Diffusers MiniMax-H3 LoRA to ComfyUI's fused H3 layout.

MiniMax-H3's ComfyUI implementation fuses Q/K/V and uses the opposite SwiGLU
half order from Diffusers.  This converter preserves the represented low-rank
updates by building a block-diagonal QKV B matrix, concatenating the three A
matrices, and swapping the two FFN-up halves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LORA_SUFFIXES = (
    (".lora.down.weight", "A"),
    (".lora.up.weight", "B"),
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
)


def read_pairs(path: Path) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            for suffix, part in LORA_SUFFIXES:
                if key.endswith(suffix):
                    pairs.setdefault(key[: -len(suffix)], {})[part] = handle.get_tensor(key)
                    break
            else:
                raise ValueError(f"unsupported LoRA tensor key: {key}")
    incomplete = sorted(name for name, tensors in pairs.items() if set(tensors) != {"A", "B"})
    if incomplete:
        raise ValueError(f"LoRA pairs missing A or B: {incomplete[:5]}")
    return pairs, metadata


def source_blocks(pairs: dict[str, dict[str, torch.Tensor]]) -> list[str]:
    suffix = ".attn.to_q"
    blocks = sorted(name[: -len(suffix)] for name in pairs if name.endswith(suffix))
    if not blocks:
        raise ValueError("no MiniMax-H3 attention blocks found")
    return blocks


def destination_block(source: str) -> str:
    if source.startswith("transformer_blocks."):
        return "blocks." + source.removeprefix("transformer_blocks.")
    if source.startswith("token_refiner.refiner_blocks."):
        return "token_refiner.blocks." + source.removeprefix("token_refiner.refiner_blocks.")
    raise ValueError(f"unsupported MiniMax-H3 block: {source}")


def convert(pairs: dict[str, dict[str, torch.Tensor]], alpha: float) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for source in source_blocks(pairs):
        destination = f"diffusion_model.{destination_block(source)}"
        qkv = [pairs[f"{source}.attn.to_{name}"] for name in ("q", "k", "v")]
        rank = qkv[0]["A"].shape[0]
        if any(item["A"].shape[0] != rank for item in qkv):
            raise ValueError(f"mixed QKV ranks in {source}")
        output[f"{destination}.attn.qkv_proj.lora_A.weight"] = torch.cat(
            [item["A"] for item in qkv], dim=0
        )
        output[f"{destination}.attn.qkv_proj.lora_B.weight"] = torch.block_diag(
            *[item["B"] for item in qkv]
        )
        output[f"{destination}.attn.qkv_proj.alpha"] = torch.tensor(alpha * 3.0)

        direct = {
            "attn.to_out.0": "attn.out_proj",
            "ff.net.0.proj": "mlp.fc1",
            "ff.net.2": "mlp.fc2",
        }
        for source_tail, destination_tail in direct.items():
            item = pairs[f"{source}.{source_tail}"]
            a, b = item["A"], item["B"]
            if source_tail == "ff.net.0.proj":
                if b.shape[0] % 2:
                    raise ValueError(f"odd SwiGLU output width in {source}.{source_tail}")
                value, gate = b.chunk(2, dim=0)
                b = torch.cat((gate, value), dim=0)
            output[f"{destination}.{destination_tail}.lora_A.weight"] = a
            output[f"{destination}.{destination_tail}.lora_B.weight"] = b
            output[f"{destination}.{destination_tail}.alpha"] = torch.tensor(alpha)
    # Six Diffusers modules become four ComfyUI modules per block; each output
    # module has A, B and alpha tensors, so both sides contain 12 tensors/block.
    expected = len(pairs) * 2
    if len(output) != expected:
        raise ValueError(f"expected {expected} output tensors, produced {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pairs, source_metadata = read_pairs(args.source)
    ranks = {item["A"].shape[0] for item in pairs.values()}
    if len(ranks) != 1:
        raise ValueError(f"mixed LoRA ranks: {sorted(ranks)}")
    rank = ranks.pop()
    alpha = float(source_metadata.get("lora_alpha", source_metadata.get("alpha", rank)))
    tensors = convert(pairs, alpha)
    metadata = {
        **source_metadata,
        "source_format": "Diffusers MiniMax-H3 LoRA",
        "target_format": "ComfyUI generic LoRA",
        "qkv_fusion": "block diagonal B; concat A; alpha multiplied by 3",
        "swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]",
        "conversion_source": str(args.source),
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_name(f".{args.destination.name}.tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    temporary.replace(args.destination)
    print(f"converted {len(pairs) // 6} blocks / {len(tensors)} tensors -> {args.destination}")


if __name__ == "__main__":
    main()
