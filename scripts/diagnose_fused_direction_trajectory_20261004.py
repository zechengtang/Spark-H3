#!/usr/bin/env python3
"""Scan existing 5s captures for legacy/fused permutation divergence."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from diagnose_fused_external_20261004 import (
    build_permutations,
    layout_for,
    perm_metrics,
)


CAPTURES = Path(
    "/autodl-fs/data/h3_experiments/reblock_approx_10prompt_5s768p_20261002/captures"
)
OUTPUT = Path(
    "/autodl-fs/data/h3_experiments/check_fused_extern_20261004/"
    "direction_trajectory_5s_case01.json"
)
EVALUATIONS = (4, 11, 18)
LAYERS = (1, 25, 49)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    rows = []
    for evaluation in EVALUATIONS:
        for layer in LAYERS:
            path = CAPTURES / f"case01_eval{evaluation:02}_layer{layer:02}.pt"
            data = torch.load(path, map_location="cpu", weights_only=True)
            q = data["q"].permute(1, 0, 2).unsqueeze(0).contiguous().cuda()
            k = data["k"].permute(1, 0, 2).unsqueeze(0).contiguous().cuda()
            video_tokens = q.shape[1]
            # Permutation construction only reads the video prefix.
            layout = layout_for(q, video_tokens, tuple(data["grid"]))
            legacy, _ = build_permutations(q, k, layout, "legacy")
            fused, _ = build_permutations(q, k, layout, "fused")
            rows.append({
                "case": 1,
                "evaluation": evaluation,
                "layer": layer,
                "capture": str(path),
                "metrics": perm_metrics(legacy, fused),
            })
            del data, q, k, legacy, fused
            torch.cuda.empty_cache()
            print(f"complete eval={evaluation} layer={layer}", flush=True)
    OUTPUT.write_text(json.dumps({"status": "complete", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
