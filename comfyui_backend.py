"""ComfyUI-only Spark backend built on the official comfy-kitchen Sol stack.

This module deliberately owns its configuration, runtime state, backend
selection, and attention call.  It never enters the Diffusers/PyTorch Spark
dispatcher.  It shares only low-level landmark mathematics and tensor kernels,
not configuration, planning state, backend selection, or attention dispatch.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass

import torch


BLOCK_SIZE = 64
_PROFILE_REBLOCK = os.environ.get("SPARK_PROFILE_REBLOCK") == "1"


@dataclass(frozen=True)
class ComfyPackedLayout:
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    grid: tuple[int, int, int]
    video_tokens: int
    sequence_length: int
    video_positions: torch.Tensor


@dataclass(frozen=True)
class ComfySparkConfig:
    topk_ratio: float = 0.1
    tau: float = 1.0
    global_anchor_dtype: str = "float32"
    video_tail_mode: str = "dense"
    landmark_tree_v2_initial_order: str = "flat"
    landmark_tree_v2_children: int | tuple[int, ...] = 16
    landmark_tree_v2_fanout_mode: str = "power_of_two_fanout"
    landmark_tree_v2_final_fanout: int | tuple[int, ...] | None = None
    landmark_tree_v2_root_fanout: int | None = None
    landmark_tree_v2_landmark_mode: str = "midpoint"
    landmark_tree_v2_landmark_count: int = 32
    landmark_tree_v2_aggregation: str = "linear"
    landmark_tree_v2_distance: str = "cosine"
    landmark_tree_v2_mean_mode: str = "raw"
    landmark_tree_v2_moment_mode: str = "raw"
    landmark_tree_v2_order_mode: str = "parent_order"
    landmark_tree_v2_group_size: int | tuple[int, ...] = 1
    rope_sol_key_ridge_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.global_anchor_dtype not in ("float32", "bfloat16"):
            raise ValueError("global_anchor_dtype must be 'float32' or 'bfloat16'")


class ComfySparkController:
    def __init__(self, config: ComfySparkConfig):
        self.config = config
        self.reblock_reuse_layers = 1
        self.reblock_reuse_start_layer = 1
        self.reset()

    def reset(self) -> None:
        self.evaluation_index = -1
        self.counts = Counter()
        self.backend = None
        self.rope_sol_key_clustering_static = {}
        self.landmark_reblock_hierarchy = None
        self.last_reblock = None
        self.reblock_profile = []


def _profile_begin(controller: ComfySparkController, name: str):
    if not _PROFILE_REBLOCK:
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return name, start, end


def _profile_end(controller: ComfySparkController, sample) -> None:
    if sample is None:
        return
    name, start, end = sample
    end.record()
    controller.reblock_profile.append((name, start, end))


def reblock_profile_summary(controller: ComfySparkController, *, clear: bool = True):
    samples = controller.reblock_profile
    if not samples:
        return None
    torch.cuda.synchronize()
    durations = {}
    for name, start, end in samples:
        durations.setdefault(name, []).append(start.elapsed_time(end))
    if clear:
        controller.reblock_profile = []
    return {
        name: {
            "total_ms": sum(values),
            "calls": len(values),
            "mean_ms": sum(values) / len(values),
            "median_ms": sorted(values)[len(values) // 2],
            "max_ms": max(values),
        }
        for name, values in durations.items()
    }


@torch.no_grad()
def comfy_kitchen_spark_attention(
    controller: ComfySparkController,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: ComfyPackedLayout,
    layer: int,
) -> torch.Tensor:
    """Execute ComfyUI Spark directly through comfy-kitchen, without fallback."""

    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise RuntimeError("ComfyUI Spark requires contiguous BF16 BTHD tensors with head_dim=128")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("ComfyUI Spark requires matching Q/K/V shapes")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise RuntimeError("ComfyUI Spark requires contiguous Q/K/V")

    try:
        from .comfyui_reblock_plan import build_comfy_reblock_permutations
    except ImportError:  # standalone source-tree tests
        from comfyui_reblock_plan import build_comfy_reblock_permutations

    reuse = int(controller.reblock_reuse_layers)
    start = int(controller.reblock_reuse_start_layer)
    group = (layer - start) // reuse if reuse > 1 and layer >= start else None
    key = (controller.evaluation_index, group)
    group_start = start + group * reuse if group is not None else layer
    cached = controller.last_reblock
    if group is not None and layer != group_start and cached is not None and cached[0] == key:
        query_permutation, query_inverse, key_permutation, hierarchy = cached[1]
        controller.landmark_reblock_hierarchy = hierarchy
        controller.counts["reblock_reuse_calls"] += 1
    else:
        sample = _profile_begin(controller, "reblock_permutation_build_total")
        query_permutation, query_inverse, key_permutation, _ = (
            build_comfy_reblock_permutations(controller, q, k, layout)
        )
        _profile_end(controller, sample)
        if group is not None:
            controller.last_reblock = (
                key,
                (
                    query_permutation,
                    query_inverse,
                    key_permutation,
                    controller.landmark_reblock_hierarchy,
                ),
            )

    try:
        from comfy_kitchen.backends.cuda import spark_attn
    except (ImportError, OSError) as error:
        raise RuntimeError("ComfyUI Spark requires the comfy-kitchen Spark extension") from error

    blocks = math.ceil(q.shape[1] / BLOCK_SIZE)
    sink_first = layout.video_tokens // BLOCK_SIZE
    sample = _profile_begin(controller, "sparse_attention")
    output = spark_attn(
        q,
        k,
        v,
        video_tokens=layout.video_tokens,
        anchor_dtype=(
            torch.bfloat16
            if controller.config.global_anchor_dtype == "bfloat16"
            else torch.float32
        ),
        topk_ratio=controller.config.topk_ratio,
        tau=controller.config.tau,
        sink_blocks=[sink_first, blocks],
        sink_q=[sink_first, blocks],
        query_permutation=query_permutation,
        key_permutation=key_permutation,
        query_inverse=query_inverse,
    )
    _profile_end(controller, sample)
    controller.backend = "comfy-kitchen-spark-global"
    controller.counts["comfy_kitchen_calls"] += 1
    return output
