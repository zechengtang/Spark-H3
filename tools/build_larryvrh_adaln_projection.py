#!/usr/bin/env python3
"""Maintainer tool: build the small LarryVRH-to-pruned-AdaLN projection asset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruned-model", type=Path, required=True)
    parser.add_argument("--silu-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with safe_open(args.pruned_model, framework="pt", device="cpu") as model:
        table = model.get_tensor("adaln_t_table").float()
    egrid = load_file(str(args.silu_grid), device="cpu")["silu_t_emb_grid"].float()
    if table.shape != (1025, 8) or egrid.shape != (1025, 2688):
        raise ValueError(f"unexpected inputs: table={table.shape}, egrid={egrid.shape}")
    design = torch.cat((table, torch.ones(table.shape[0], 1)), dim=1)
    projection = torch.linalg.pinv(design.double()).float() @ egrid
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"projection": projection.contiguous()}, str(args.output), metadata={
        "description": "least-squares map from full H3 silu(t_emb) to pruned 8-D AdaLN curve+bias",
        "base": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    })
    print(args.output)


if __name__ == "__main__":
    main()
