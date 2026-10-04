#!/usr/bin/env python3
"""SM120 execution parity after constructing an identical exact mask."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from h3_sparse_attention import H3SparseAttentionConfig
from h3_sparse_attention.processor import _Controller
from h3_sparse_attention.sol_numerator_virtual_q import reduce_virtual_key_centroids
from h3_sparse_attention.sol_topk_cutoff import (
    _gemm_score_map_prefix,
    gemm_radix_topk_cutoff,
    gemm_topk_packed_route,
)
from h3_sparse_attention.spark_integration import _spark_topk_attention


OUTPUT = Path(
    "/autodl-fs/data/h3_experiments/check_fused_extern_20261004/"
    "same_mask_fixture.json"
)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "mismatched_elements": int((a != b).sum().item()),
        "max_abs": float(diff.max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(a.float() - b.float()) /
             torch.linalg.vector_norm(a.float())).item()
        ),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    generator = torch.Generator(device="cuda").manual_seed(20261004)
    q = torch.zeros(1, 8256, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[..., 0] = 1
    # Exactly ordered BF16 centroids avoid a Top-K boundary tie.  This fixture
    # is deliberately synthetic: it isolates execution after mask equality.
    for block in range(129):
        k[:, block * 64:min((block + 1) * 64, 8256), :, 0] = block / 128
    v = torch.randn(
        1, 8256, 1, 128, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    kc = reduce_virtual_key_centroids(k)
    threshold, threshold_stats = gemm_radix_topk_cutoff(
        q, kc, video_tokens=8192, sink_tokens=64, topk_ratio=0.25,
        _return_first_excluded=True,
    )
    packed, packed_stats = gemm_topk_packed_route(
        q, kc, video_tokens=8192, topk_ratio=0.25, query_tokens=8192,
    )
    scores = _gemm_score_map_prefix(
        q, kc, query_blocks=128, candidate_blocks=128
    )
    threshold_mask = scores > threshold[:, :128, :, None]
    bits = torch.arange(32, device="cuda", dtype=torch.int32)
    packed_mask = ((packed[..., None] >> bits) & 1).bool().flatten(-2)[..., :128]

    ranges = torch.tensor([[0, 8192], [8192, 8256]], device="cuda", dtype=torch.int64)
    mapping = torch.tensor([0] * 128 + [1], device="cuda", dtype=torch.int64)
    layout = SimpleNamespace(video_tokens=8192, sequence_length=8256)
    outputs = {}
    for execution in ("threshold", "packed_external", "packed_external_no_route_qk"):
        controller = _Controller(H3SparseAttentionConfig.spark(
            20, sol_route_topk_execution=execution, sol_route_topk_ratio=0.25,
            sol_log_density=False, sol_force_local_blocks=False,
        ))
        outputs[execution] = _spark_topk_attention(
            controller, q, k, v, layout,
            virtual_query_data=(ranges, mapping, None), _query_tokens=8192,
        )
    result = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(),
        "shape": list(q.shape),
        "mask_bitwise_equal": bool(torch.equal(threshold_mask, packed_mask)),
        "mask_different_bits": int((threshold_mask ^ packed_mask).sum().item()),
        "threshold_tie_rows": threshold_stats["cutoff_tie_video_query_rows"],
        "threshold_budget_min_max": [
            int(threshold_mask.sum(-1).min()), int(threshold_mask.sum(-1).max())
        ],
        "packed_budget": packed_stats["target_topk_blocks_per_query"],
        "threshold_vs_packed_external": compare(
            outputs["threshold"], outputs["packed_external"]
        ),
        "packed_external_vs_no_route_qk": compare(
            outputs["packed_external"], outputs["packed_external_no_route_qk"]
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
