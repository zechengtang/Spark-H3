"""Reversible Sol-Attn integration for packed MiniMax-H3 inference.

Adapted from MiniMax-H3-Sparse. Target video uses Sol-Attn by default; Ref2VA
may explicitly opt conditioning-video rows into the same sparse video domain.
Text and audio keys remain exact, and their context queries run densely.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from .spark_defaults import (SPARK_REWEIGHT_TARGET_BLOCKS, SPARK_REWEIGHT_MIN_BLOCKS, SPARK_REWEIGHT_MAX_BLOCKS)

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class H3SparseAttentionConfig:
    method: str = "sol"
    total_evaluations: int = 49
    warmup_percent: float = 20.0
    sol_tau: float = 1.0
    sol_thresh_type: str = "diag"
    sol_kv_splits: int = 1
    sol_dense_layers: int = 1
    sol_extra_dense_evaluations: tuple[int, ...] = ()
    sol_extra_dense_layers: tuple[int, ...] = ()
    sol_force_local_blocks: bool | None = None
    # Optional symmetric exact-block radius outside the Top-K budget.  Zero
    # retains only the self block; one is the legacy three-block policy.
    sol_exact_block_radius: int | None = None
    sol_tail_granularity: Literal["query", "block", "block8x8"] = "query"
    sol_global_anchor_dtype: Literal["bfloat16", "float32"] = "float32"
    # Numeric ablations for virtual-query K/V summaries. The default preserves
    # the existing tensor-core path and its stored-key log-mass convention.
    sol_reweight_summary_math: Literal["tensorcore", "comfy_fp32"] = "tensorcore"
    sol_reweight_logmass_key: Literal["stored", "pre_round"] = "stored"
    # Orthogonal ablation of anchor-weighted K/V and the per-block log-mass bias.
    sol_reweight_components: Literal["full", "weights_only", "bias_only", "none"] = "full"

    sol_route_topk_ratio: float | None = None
    sol_route_topk_cutoff_mode: Literal[
        "gemm_radix", "gaussian_moments"
    ] = "gemm_radix"
    sol_route_topk_execution: Literal[
        "threshold", "packed_external", "packed_external_no_route_qk", "fused"
    ] = "threshold"
    sol_sparse_video_scope: Literal["target", "target_and_condition"] = "target"
    sol_video_tail_mode: Literal["dense", "pad"] = "dense"
    # Benchmark-only compatibility switch for the pre-tail-fix behavior.  It
    # intentionally evaluates the packed context query rows sparsely whenever
    # the video prefix is not block aligned, before the dense overwrite below.
    sol_legacy_full_query: bool = False
    sol_log_density: bool = True
    sol_landmark_preprocess: bool = False
    sol_landmark_preprocess_version: Literal["v1", "v2"] = "v1"
    sol_virtual_query_levels_up: int | None = None
    sol_virtual_query_target_blocks: int | None = None
    sol_virtual_query_min_blocks: int = 4
    sol_virtual_query_max_blocks: int = 12
    sol_virtual_query_fused_permute: bool = True
    sol_virtual_query_route_score: Literal["native_mean", "mean", "weighted_k", "weighted_mass"] = "native_mean"
    landmark_tree_v2_initial_order: Literal[
        "flat", "tile_t4h4w4", "hilbert_thw", "hilbert_twh", "hilbert_htw",
        "hilbert_hwt", "hilbert_wth", "hilbert_wht",
    ] = "flat"
    landmark_tree_v2_distance: Literal["euclidean", "cosine"] = "cosine"
    landmark_tree_v2_order_mode: Literal["parent_order", "scalar_order"] = "parent_order"
    landmark_tree_v2_mean_mode: Literal["raw", "input_unit", "metric_unit"] = "raw"
    landmark_tree_v2_moment_mode: Literal["raw", "unit"] = "raw"
    landmark_tree_v2_group_size: int | list[int] | tuple[int, ...] = 1
    landmark_tree_v2_children: int | list[int] | tuple[int, ...] | None = None
    landmark_tree_v2_fanout: int | list[int] | tuple[int, ...] | None = None
    landmark_tree_v2_final_fanout: int | list[int] | tuple[int, ...] | None = None
    landmark_tree_v2_root_fanout: int | None = None
    landmark_tree_v2_landmark_mode: Literal["mean", "midpoint"] = "midpoint"
    landmark_tree_v2_landmark_count: int = 32
    # Preserve the frozen Spark routing arithmetic by default. In the paired
    # 25-prompt 768p study, fused directions lowered mean PSNR by 0.114 dB at
    # 5s and 0.191 dB at 10s versus legacy (not statistically significant:
    # p=0.61 and p=0.26). Keep this choice independent of packed/external
    # execution so kernel optimizations can be evaluated without that confound.
    landmark_tree_v2_midpoint_direction_mode: Literal["legacy", "fused"] = "legacy"
    landmark_tree_v2_aggregation: Literal["linear", "max"] = "linear"
    rope_sol_key_ridge_epsilon: float = 1e-3

    # Reblock-only experiment controls.  Defaults preserve the production
    # Spark-H3 path; consumers may opt into the alternatives explicitly.
    landmark_tree_v2_m2_side: Literal["both", "query", "key", "none"] = "both"
    landmark_tree_v2_m2_estimator: Literal[
        "hilbert_midpoint", "flat64_block_mean", "flat64_midpoint_1",
        "flat64_midpoint_2", "flat64_midpoint_4", "flat64_mean_diag", "full",
    ] = "hilbert_midpoint"
    landmark_tree_v2_proxy_iterations: Literal[0, 1, 2, 4] = 2
    landmark_tree_v2_proxy_seed_rule: Literal[
        "farthest_pair", "endpoint_order"
    ] = "farthest_pair"
    landmark_tree_v2_proxy_update_rule: Literal["mean", "medoid"] = "mean"
    landmark_tree_v2_layout_reuse: Literal[
        "independent", "q_from_k", "k_from_q"
    ] = "independent"

    sol_route_global_weighted_mean: bool = False
    sol_route_global_weighted_side: Literal["both", "query", "key"] = "both"
    landmark_tree_v2_fanout_mode: Literal[
        "power_of_two_fanout", "arbitrary_fanout", "power_of_two_arbitrary_final"
    ] = "power_of_two_fanout"

    def __post_init__(self):
        if self.method != "sol":
            raise ValueError("this package supports method='sol' only")
        if type(self.total_evaluations) is not int or self.total_evaluations < 1:
            raise ValueError("total_evaluations must be a positive integer")
        if not 0 <= self.warmup_percent <= 100:
            raise ValueError("warmup_percent must lie in [0, 100]")
        if not math.isfinite(self.sol_tau) or self.sol_tau < 0:
            raise ValueError("sol_tau must be finite and nonnegative")
        if self.sol_thresh_type not in ("diag", "exact"):
            raise ValueError("sol_thresh_type must be 'diag' or 'exact'")
        if type(self.sol_kv_splits) is not int or self.sol_kv_splits < 1:
            raise ValueError("sol_kv_splits must be a positive integer")
        if type(self.sol_dense_layers) is not int or self.sol_dense_layers < 0:
            raise ValueError("sol_dense_layers must be a nonnegative integer")
        if type(self.sol_extra_dense_evaluations) is not tuple or not all(
                type(e) is int and 0 <= e < self.total_evaluations
                for e in self.sol_extra_dense_evaluations):
            raise ValueError("sol_extra_dense_evaluations must be a tuple of evaluation "
                             "indices in [0, total_evaluations)")
        object.__setattr__(self, "sol_extra_dense_evaluations",
                           tuple(sorted(set(self.sol_extra_dense_evaluations))))
        if type(self.sol_extra_dense_layers) is not tuple or not all(
                type(layer) is int and layer >= 0 for layer in self.sol_extra_dense_layers):
            raise ValueError("sol_extra_dense_layers must be a tuple of nonnegative layer indices")
        object.__setattr__(self, "sol_extra_dense_layers",
                           tuple(sorted(set(self.sol_extra_dense_layers))))
        if self.sol_force_local_blocks is not None and type(self.sol_force_local_blocks) is not bool:
            raise ValueError("sol_force_local_blocks must be bool")
        if (self.sol_exact_block_radius is not None
                and (type(self.sol_exact_block_radius) is not int
                     or self.sol_exact_block_radius < 0)):
            raise ValueError(
                "sol_exact_block_radius must be None or a nonnegative integer"
            )
        if (self.sol_exact_block_radius is not None
                and self.sol_force_local_blocks is not None):
            raise ValueError(
                "set sol_exact_block_radius or sol_force_local_blocks, not both"
            )
        if (self.sol_exact_block_radius is not None
                and self.sol_route_topk_ratio is None):
            raise ValueError("sol_exact_block_radius requires Top-K routing")
        if self.sol_tail_granularity not in ("query", "block", "block8x8"):
            raise ValueError("sol_tail_granularity must be 'query', 'block', or 'block8x8'")
        if self.sol_global_anchor_dtype not in ("bfloat16", "float32"):
            raise ValueError("sol_global_anchor_dtype must be 'bfloat16' or 'float32'")
        if self.sol_reweight_summary_math not in ("tensorcore", "comfy_fp32"):
            raise ValueError("invalid sol_reweight_summary_math")
        if self.sol_reweight_logmass_key not in ("stored", "pre_round"):
            raise ValueError("invalid sol_reweight_logmass_key")
        if self.sol_reweight_components not in ("full", "weights_only", "bias_only", "none"):
            raise ValueError("invalid sol_reweight_components")

        if self.sol_route_topk_ratio is not None and not 0 < self.sol_route_topk_ratio <= 1:
            raise ValueError("sol_route_topk_ratio must lie in (0, 1]")
        if self.sol_route_topk_cutoff_mode not in ("gemm_radix", "gaussian_moments"):
            raise ValueError("invalid sol_route_topk_cutoff_mode")
        if self.sol_route_topk_execution not in (
            "threshold", "packed_external", "packed_external_no_route_qk", "fused"
        ):
            raise ValueError("invalid sol_route_topk_execution")
        if self.sol_sparse_video_scope not in ("target", "target_and_condition"):
            raise ValueError("sol_sparse_video_scope must be 'target' or 'target_and_condition'")
        if self.sol_video_tail_mode not in ("dense", "pad"):
            raise ValueError("sol_video_tail_mode must be 'dense' or 'pad'")
        if type(self.sol_legacy_full_query) is not bool:
            raise TypeError("sol_legacy_full_query must be bool")
        if self.sol_legacy_full_query and self.sol_video_tail_mode != "dense":
            raise ValueError(
                "sol_legacy_full_query is incompatible with sol_video_tail_mode='pad'"
            )
        if self.sol_landmark_preprocess and self.sol_landmark_preprocess_version != "v2":
            raise ValueError("this port supports landmark preprocessing version 'v2' only")
        from .reblock_hierarchy import normalize_fanout_mode
        object.__setattr__(self, "landmark_tree_v2_fanout_mode", normalize_fanout_mode(self.landmark_tree_v2_fanout_mode))
        if type(self.sol_route_global_weighted_mean) is not bool:
            raise TypeError("sol_route_global_weighted_mean must be bool")
        if self.sol_route_global_weighted_side not in ("both", "query", "key"):
            raise ValueError("sol_route_global_weighted_side must be both, query, or key")
        if self.sol_route_global_weighted_mean and self.sol_route_topk_ratio is None:
            raise ValueError("global weighted routing requires Top-K routing")
        if not math.isfinite(self.rope_sol_key_ridge_epsilon) or self.rope_sol_key_ridge_epsilon < 0:
            raise ValueError("rope_sol_key_ridge_epsilon must be finite and nonnegative")
        if self.landmark_tree_v2_m2_side not in ("both", "query", "key", "none"):
            raise ValueError("landmark_tree_v2_m2_side must be both, query, key, or none")
        if self.landmark_tree_v2_m2_estimator not in (
            "hilbert_midpoint", "flat64_block_mean", "flat64_midpoint_1",
            "flat64_midpoint_2", "flat64_midpoint_4", "flat64_mean_diag", "full",
        ):
            raise ValueError("invalid landmark_tree_v2_m2_estimator")
        if self.landmark_tree_v2_proxy_iterations not in (0, 1, 2, 4):
            raise ValueError("landmark_tree_v2_proxy_iterations must be 0, 1, 2, or 4")
        if self.landmark_tree_v2_proxy_seed_rule not in ("farthest_pair", "endpoint_order"):
            raise ValueError("invalid landmark_tree_v2_proxy_seed_rule")
        if self.landmark_tree_v2_proxy_update_rule not in ("mean", "medoid"):
            raise ValueError("invalid landmark_tree_v2_proxy_update_rule")
        if self.landmark_tree_v2_layout_reuse not in ("independent", "q_from_k", "k_from_q"):
            raise ValueError("invalid landmark_tree_v2_layout_reuse")
        if (self.sol_exact_block_radius is not None
                and self.sol_landmark_preprocess
                and self.landmark_tree_v2_layout_reuse == "independent"):
            raise ValueError(
                "sol_exact_block_radius with reblock requires "
                "landmark_tree_v2_layout_reuse='q_from_k' or 'k_from_q'"
            )
        if self.sol_virtual_query_route_score not in ("native_mean", "mean", "weighted_k", "weighted_mass"):
            raise ValueError("invalid sol_virtual_query_route_score")
        virtual_query_enabled = (
            self.sol_virtual_query_levels_up is not None
            or self.sol_virtual_query_target_blocks is not None
        )
        if (self.sol_reweight_summary_math != "tensorcore"
                or self.sol_reweight_logmass_key != "stored"
                or self.sol_reweight_components != "full") and not virtual_query_enabled:
            raise ValueError("reweight numeric ablations require virtual query summaries")
        if self.sol_tail_granularity in ("block", "block8x8") and not virtual_query_enabled:
            raise ValueError("block tail requires virtual query summaries")
        if (self.sol_tail_granularity in ("block", "block8x8")
                and self.sol_route_topk_ratio is not None
                and self.sol_route_topk_execution == "fused"):
            raise ValueError("block tail requires threshold or packed_external Top-K routing")
        if self.sol_virtual_query_route_score != "native_mean" and not virtual_query_enabled:
            raise ValueError("reweighted routing requires virtual query summaries")
        if self.sol_virtual_query_levels_up is not None and self.sol_virtual_query_target_blocks is not None:
            raise ValueError("choose either sol_virtual_query_levels_up or sol_virtual_query_target_blocks")
        if self.sol_virtual_query_levels_up is not None:
            if type(self.sol_virtual_query_levels_up) is not int or self.sol_virtual_query_levels_up < 0:
                raise ValueError("sol_virtual_query_levels_up must be None or a nonnegative integer")
        if self.sol_virtual_query_target_blocks is not None:
            if type(self.sol_virtual_query_target_blocks) is not int:
                raise ValueError("sol_virtual_query_target_blocks must be an integer or None")
            if not (
                type(self.sol_virtual_query_min_blocks) is int
                and type(self.sol_virtual_query_max_blocks) is int
                and 1 <= self.sol_virtual_query_min_blocks
                <= self.sol_virtual_query_target_blocks
                <= self.sol_virtual_query_max_blocks
            ):
                raise ValueError("virtual query blocks require 1 <= min <= target <= max")
        if virtual_query_enabled:
            if not self.sol_landmark_preprocess or self.sol_landmark_preprocess_version != "v2":
                raise ValueError("virtual query summaries require LMv2 Sol preprocessing")
            if self.sol_route_topk_ratio is None and self.sol_virtual_query_route_score != "native_mean":
                raise ValueError("tau routing with virtual query summaries requires native_mean scores")
        if (
            self.sol_video_tail_mode == "pad"
            and self.sol_route_topk_ratio is None
            and not virtual_query_enabled
        ):
            raise ValueError(
                "sol_video_tail_mode='pad' requires Top-K or virtual-query routing"
            )
        if self.landmark_tree_v2_moment_mode not in ("raw", "unit"):
            raise ValueError("landmark_tree_v2_moment_mode must be raw or unit")
        if self.landmark_tree_v2_mean_mode not in ("raw", "input_unit", "metric_unit"):
            raise ValueError("landmark_tree_v2_mean_mode must be raw, input_unit or metric_unit")
        if self.landmark_tree_v2_mean_mode != "raw" and self.landmark_tree_v2_distance != "cosine":
            raise ValueError("input-unit means require cosine distance")
        from .reblock_hierarchy import normalize_final_fanout, resolve_root_fanout
        from .landmark_tree_v2 import _normalize_fanout, _normalize_group_size, _normalize_order_mode
        from .landmark_initial_order import normalize_initial_order

        normalize_initial_order(self.landmark_tree_v2_initial_order)

        if self.landmark_tree_v2_fanout is not None:
            if self.landmark_tree_v2_children is not None and (
                    _normalize_fanout(self.landmark_tree_v2_children, None)
                    != _normalize_fanout(self.landmark_tree_v2_fanout, None)):
                raise ValueError("use landmark_tree_v2_fanout or landmark_tree_v2_children, not conflicting values")
            children = self.landmark_tree_v2_fanout
        elif self.landmark_tree_v2_children is None:
            children = 16
        else:
            children = self.landmark_tree_v2_children
        object.__setattr__(self, "landmark_tree_v2_children",
                           _normalize_fanout(children, None))
        if self.landmark_tree_v2_fanout is not None:
            object.__setattr__(self, "landmark_tree_v2_fanout", self.landmark_tree_v2_children)
        if self.landmark_tree_v2_root_fanout is not None:
            resolve_root_fanout(self.landmark_tree_v2_children, self.landmark_tree_v2_root_fanout)
        if self.landmark_tree_v2_final_fanout is not None:
            object.__setattr__(self, "landmark_tree_v2_final_fanout",
                               normalize_final_fanout(self.landmark_tree_v2_final_fanout))
        if self.landmark_tree_v2_landmark_count not in (32, 64, 128, 256):
            raise ValueError("landmark_tree_v2_landmark_count must be 32, 64, 128, or 256")
        if self.landmark_tree_v2_landmark_mode not in ("mean", "midpoint"):
            raise ValueError("landmark_tree_v2_landmark_mode must be mean or midpoint")
        if self.landmark_tree_v2_midpoint_direction_mode not in ("legacy", "fused"):
            raise ValueError("landmark_tree_v2_midpoint_direction_mode must be legacy or fused")
        if self.landmark_tree_v2_aggregation not in ("linear", "max") or (self.landmark_tree_v2_aggregation == "max" and self.landmark_tree_v2_distance != "cosine"):
            raise ValueError("max aggregation requires cosine distance")
        if self.landmark_tree_v2_distance not in ("euclidean", "cosine"):
            raise ValueError("landmark_tree_v2_distance must be euclidean or cosine")
        object.__setattr__(self, "landmark_tree_v2_order_mode",
                           _normalize_order_mode(self.landmark_tree_v2_order_mode))
        object.__setattr__(self, "landmark_tree_v2_group_size",
                           _normalize_group_size(self.landmark_tree_v2_group_size))
        if self.sol_virtual_query_route_score != "native_mean" and not self.sol_local_blocks_enabled:
            raise ValueError("reweighted route scores require sol_force_local_blocks=True")

    @property
    def sol_local_blocks_enabled(self):
        if self.sol_exact_block_radius is not None:
            return True
        if self.sol_force_local_blocks is not None:
            return self.sol_force_local_blocks
        return not self.sol_landmark_preprocess

    @property
    def sol_local_block_radius(self):
        """Kernel policy: -1 disables retention; nonnegative values are radii."""
        if self.sol_exact_block_radius is not None:
            return self.sol_exact_block_radius
        return 1 if self.sol_local_blocks_enabled else -1

    @property
    def sol_local_block_policy(self):
        """Preserve legacy bool calls while exposing an explicit radius."""
        if self.sol_exact_block_radius is not None:
            return self.sol_exact_block_radius
        return self.sol_local_blocks_enabled

    @classmethod
    def sol(cls, num_inference_steps: int = 50, **overrides):
        if type(num_inference_steps) is not int or num_inference_steps < 2:
            raise ValueError("num_inference_steps must be an integer of at least 2")
        return cls(total_evaluations=num_inference_steps - 1, **overrides)

    @classmethod
    def spark(cls, num_inference_steps: int = 20, **overrides) -> "H3SparseAttentionConfig":
        """Sol TopK10 + ungrouped fanout-16 LMv2 + global reweighting."""
        defaults = dict(
            sol_route_topk_ratio=0.1,
            sol_route_topk_cutoff_mode="gemm_radix",
            # Match the compiled Table 4 Spark path by default. The packed
            # external route and FP32 anchor remain explicit ablations.
            sol_route_topk_execution="threshold",
            sol_global_anchor_dtype="bfloat16",
            sol_landmark_preprocess=True,
            sol_landmark_preprocess_version="v2",
            landmark_tree_v2_children=16,
            landmark_tree_v2_midpoint_direction_mode="legacy",
            # Collapse the completed reblock hierarchy to its video root.  A
            # deliberately oversized levels-up value makes the global policy
            # independent of the number of hierarchy levels for a given grid.
            sol_virtual_query_target_blocks=None,
            sol_virtual_query_levels_up=99,
            sol_virtual_query_route_score="native_mean",
        )
        # Preserve target-189 as an explicit compatibility ablation without
        # making callers clear the global preset manually.
        if overrides.get("sol_virtual_query_target_blocks") is not None:
            if "sol_virtual_query_levels_up" not in overrides:
                defaults["sol_virtual_query_levels_up"] = None
            defaults["sol_virtual_query_min_blocks"] = SPARK_REWEIGHT_MIN_BLOCKS
            defaults["sol_virtual_query_max_blocks"] = SPARK_REWEIGHT_MAX_BLOCKS
        if overrides.get("sol_virtual_query_levels_up") is not None and "sol_virtual_query_target_blocks" not in overrides:
            defaults["sol_virtual_query_target_blocks"] = None
        if overrides.get("landmark_tree_v2_fanout") is not None and "landmark_tree_v2_children" not in overrides:
            defaults.pop("landmark_tree_v2_children", None)
        defaults.update(overrides)
        return cls.sol(num_inference_steps, **defaults)

    @property
    def dense_evaluations(self):
        return min(self.total_evaluations,
                   math.ceil((self.total_evaluations + 1) * self.warmup_percent / 100))


@dataclass
class PackedLayout:
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    grid: tuple[int, int, int]
    video_tokens: int
    sequence_length: int
    video_positions: torch.Tensor
    target_video_tokens: int | None = None
    condition_video_tokens: int = 0
    sparse_video_scope: str = "target"


def _packed_layout(
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    video_indices: torch.Tensor | None = None,
    timestep_indices: torch.Tensor | None = None,
    text_indices: torch.Tensor | None = None,
    sparse_video_scope: str = "target",
) -> PackedLayout:
    if token_tags.ndim != 1 or position_ids.shape != (token_tags.numel(), 3):
        raise ValueError("H3 token_tags/position_ids have unexpected shapes")
    if video_indices is not None and timestep_indices is not None and text_indices is not None:
        if text_indices.numel() == 0:
            raise ValueError("packed H3 sequence contains no text row to identify target-video timestep")
        if video_indices.numel() == 0:
            raise ValueError("packed H3 sequence contains no video rows")
        # H3 packs conditioning video rows first and the generated target as
        # the final contiguous video run at the end of the sequence. Timestep
        # equality alone is insufficient: while video_timestep is above
        # keyframe_noise_aug, conditioning and target rows intentionally share
        # the same timestep. Select the structural target suffix, then use the
        # text timestep as a consistency check.
        discontinuities = torch.nonzero(
            video_indices[1:] != video_indices[:-1] + 1, as_tuple=False
        ).flatten()
        target_start = int(discontinuities[-1].item() + 1) if discontinuities.numel() else 0
        target_video = video_indices[target_start:]
        condition_video = video_indices[:target_start]
        if target_video[-1] != token_tags.numel() - 1:
            raise ValueError("H3 target video rows must be the final packed-sequence suffix")
        target_timestep = timestep_indices[text_indices[0]]
        if not bool((timestep_indices.index_select(0, target_video) == target_timestep).all()):
            raise ValueError("H3 target video timestep does not match the text timestep")
        if video_indices.numel() and not bool((token_tags.index_select(0, video_indices) == 0).all()):
            raise ValueError("H3 video_indices selected rows whose modality tag is not video")
        if sparse_video_scope == "target":
            video = target_video
        elif sparse_video_scope == "target_and_condition":
            video = video_indices
        else:
            raise ValueError(
                "sparse_video_scope must be 'target' or 'target_and_condition'"
            )
    else:
        # Minimal/test transformers may expose only tags and positions. The real
        # H3 pipeline always takes the indexed path above.
        video = torch.nonzero(token_tags == 0, as_tuple=False).flatten()
        target_video = video
        condition_video = video[:0]
        if sparse_video_scope != "target":
            raise ValueError(
                "target_and_condition scope requires H3 video/timestep/text indices"
            )
    selected = torch.zeros(token_tags.numel(), device=token_tags.device, dtype=torch.bool)
    selected[video] = True
    nonvideo = torch.nonzero(~selected, as_tuple=False).flatten()
    if video.numel() == 0:
        raise ValueError("packed H3 sequence contains no video tokens")
    def dense_grid_order(indices):
        positions = position_ids.index_select(0, indices)
        unique_t, unique_h, unique_w = (
            positions[:, axis].unique(sorted=True) for axis in range(3)
        )
        shape = (unique_t.numel(), unique_h.numel(), unique_w.numel())
        key = (
            torch.searchsorted(unique_t, positions[:, 0].contiguous()) * shape[1] * shape[2]
            + torch.searchsorted(unique_h, positions[:, 1].contiguous()) * shape[2]
            + torch.searchsorted(unique_w, positions[:, 2].contiguous())
        )
        order = key.argsort()
        expected = torch.arange(math.prod(shape), device=key.device)
        is_dense = key.numel() == expected.numel() and torch.equal(
            key.index_select(0, order), expected
        )
        return positions, shape, order, is_dense

    def heterogeneous_time_plane_order(indices):
        """Raster-order a composite whose time planes may use different grids."""
        positions = position_ids.index_select(0, indices)
        ordered_indices = []
        ordered_positions = []
        for time_value in positions[:, 0].unique(sorted=True):
            plane_mask = positions[:, 0] == time_value
            plane = indices.index_select(0, torch.nonzero(plane_mask, as_tuple=False).flatten())
            plane_pos, plane_grid, plane_order, plane_dense = dense_grid_order(plane)
            if not plane_dense or plane_grid[0] != 1:
                return positions, indices, False
            ordered_indices.append(plane.index_select(0, plane_order))
            ordered_positions.append(plane_pos.index_select(0, plane_order))
        return positions, torch.cat(ordered_indices), True

    pos, grid, order, is_dense = dense_grid_order(video)
    if is_dense:
        ordered_video = video.index_select(0, order)
        ordered_positions = pos.index_select(0, order)
    elif sparse_video_scope == "target_and_condition" and condition_video.numel():
        # Ref2VA may combine a lower-resolution conditioning-video grid with a
        # 768p target grid.  Their union is not a Cartesian grid, even though
        # each structural segment is dense.  Keep both raster-ordered segments
        # contiguous and expose a flat composite grid to Spark's token tree.
        # This preserves all video rows without padding or inventing samples.
        segment_indices = []
        segment_positions = []
        for name, segment in (("condition", condition_video), ("target", target_video)):
            segment_pos, segment_grid, segment_order, segment_dense = dense_grid_order(segment)
            if not segment_dense and name == "condition":
                segment_pos, ordered_segment, segment_dense = heterogeneous_time_plane_order(segment)
                if segment_dense:
                    segment_indices.append(ordered_segment)
                    segment_positions.append(position_ids.index_select(0, ordered_segment))
                    continue
            if not segment_dense:
                raise ValueError(
                    f"{name} video tokens do not form a dense grid: "
                    f"grid={segment_grid}, rows={segment.numel()}"
                )
            segment_indices.append(segment.index_select(0, segment_order))
            segment_positions.append(segment_pos.index_select(0, segment_order))
        ordered_video = torch.cat(segment_indices)
        ordered_positions = torch.cat(segment_positions)
        grid = (1, 1, video.numel())
    else:
        raise ValueError(f"video tokens do not form a dense grid: grid={grid}, rows={video.numel()}")

    permutation = torch.cat((ordered_video, nonvideo))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)

    return PackedLayout(
        permutation=permutation,
        inverse_permutation=inverse,
        grid=grid,
        video_tokens=video.numel(),
        sequence_length=token_tags.numel(),
        video_positions=ordered_positions,
        target_video_tokens=target_video.numel(),
        condition_video_tokens=condition_video.numel(),
        sparse_video_scope=sparse_video_scope,
    )


class _Controller:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.evaluation_index = -1
        self.layout = None
        self.counts = Counter()
        self.sol_backend = None
        self.rope_sol_key_clustering_static = {}
        self._virtual_query_layout_cache = {}
        self.landmark_reblock_hierarchy = None
        self.sol_virtual_query_layout = None
        self.sol_route_density = None
        self.exact_block_verbose_accumulator = None
        self.head_topk_budget = None
        self.spark_reblock_frozen_tail_indices = None

    def begin_forward(self, _module, args, kwargs):
        self.evaluation_index += 1
        if self.evaluation_index >= self.config.total_evaluations:
            raise RuntimeError("received more transformer evaluations than configured; "
                               "call plugin.reset() before another pipeline invocation")
        def argument(name, index):
            value = kwargs.get(name)
            return args[index] if value is None and len(args) > index else value
        tags = argument("token_tags", 5)
        positions = argument("position_ids", 6)
        if tags is None or positions is None:
            raise RuntimeError("H3 Sol-Attn requires token_tags/position_ids")
        self.layout = _packed_layout(
            tags, positions, video_indices=argument("video_indices", 7),
            timestep_indices=argument("timestep_indices", 4),
            text_indices=argument("text_indices", 9),
            sparse_video_scope=self.config.sol_sparse_video_scope)

    @property
    def is_warmup(self):
        return self.evaluation_index < self.config.dense_evaluations

    def summary(self):
        exact_verbose = None
        if self.exact_block_verbose_accumulator is not None:
            accumulator = self.exact_block_verbose_accumulator
            rows = int(accumulator["route_rows"].item())
            local = int(accumulator["local_candidates"].item())
            overlap = int(accumulator["already_selected"].item())
            added = int(accumulator["added_exact_blocks"].item())
            exact_verbose = {
                "radius": accumulator["radius"],
                "route_rows": rows,
                "local_candidates": local,
                "already_selected_by_topk": overlap,
                "added_exact_blocks": added,
                "mean_local_candidates_per_route_row": local / rows,
                "mean_already_selected_per_route_row": overlap / rows,
                "mean_added_exact_blocks_per_route_row": added / rows,
            }
        return dict(method="sol", sol_backend=self.sol_backend,
                    completed_evaluations=self.evaluation_index + 1,
                    total_evaluations=self.config.total_evaluations,
                    dense_evaluations=self.config.dense_evaluations,
                    sol_extra_dense_evaluations=self.config.sol_extra_dense_evaluations,
                    sol_extra_dense_layers=self.config.sol_extra_dense_layers,
                    processor_calls=dict(self.counts),
                    sol_route_density=self.sol_route_density,
                    exact_block_verbose=exact_verbose,
                    sol_virtual_query_layout=self.sol_virtual_query_layout,
                    sol_force_local_blocks=self.config.sol_local_blocks_enabled,
                    sol_tail_granularity=self.config.sol_tail_granularity,
                    sol_landmark_preprocess=self.config.sol_landmark_preprocess,
                    landmark_tree_v2_children=self.config.landmark_tree_v2_children,
                    landmark_tree_v2_fanout_mode=self.config.landmark_tree_v2_fanout_mode,
                    landmark_tree_v2_midpoint_direction_mode=self.config.landmark_tree_v2_midpoint_direction_mode,
                    sol_sparse_video_scope=self.config.sol_sparse_video_scope,
                    sparse_video_tokens=(None if self.layout is None else self.layout.video_tokens),
                    target_video_tokens=(None if self.layout is None else self.layout.target_video_tokens),
                    condition_video_tokens=(None if self.layout is None else self.layout.condition_video_tokens),
                    landmark_tree_v2_m2_side=self.config.landmark_tree_v2_m2_side,
                    landmark_tree_v2_m2_estimator=self.config.landmark_tree_v2_m2_estimator,
                    landmark_tree_v2_proxy_iterations=self.config.landmark_tree_v2_proxy_iterations,
                    landmark_tree_v2_proxy_seed_rule=self.config.landmark_tree_v2_proxy_seed_rule,
                    landmark_tree_v2_proxy_update_rule=self.config.landmark_tree_v2_proxy_update_rule,
                    landmark_tree_v2_layout_reuse=self.config.landmark_tree_v2_layout_reuse)


def _sol_attention(controller, q, k, v, layout, layer, *, return_bthd=False,
                   inputs_bthd=False):
    from sol_attn import get_sol_attn_backend, sol_attn

    cfg = controller.config
    allocator = getattr(controller, "head_budget_allocator", None)
    if allocator is not None:
        if cfg.sol_route_topk_ratio is None:
            raise ValueError("head budget allocation requires a Top-K configuration")
        budget_values = v.permute(0, 2, 1, 3) if inputs_bthd else v
        controller.head_topk_budget = allocator(
            layer, controller.evaluation_index, budget_values, layout.video_tokens
        )
    else:
        controller.head_topk_budget = None
    if cfg.sol_landmark_preprocess or cfg.sol_route_topk_ratio is not None:
        from .spark_integration import spark_attention, spark_attention_bthd
        if inputs_bthd:
            return spark_attention_bthd(
                controller, q, k, v, layout, layer, return_bthd=return_bthd
            )
        return spark_attention(controller, q, k, v, layout, layer, return_bthd=return_bthd)
    q, k, v = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
    sink_start = layout.video_tokens
    sink_tokens = layout.sequence_length - sink_start
    controller.sol_backend = get_sol_attn_backend(q.device)
    output = sol_attn(q, k, v, tau=cfg.sol_tau,
                      thresh_type=cfg.sol_thresh_type, kv_splits=cfg.sol_kv_splits,
                      sink_start=sink_start, sink_tokens=sink_tokens,
                      force_local_blocks=cfg.sol_local_blocks_enabled)
    controller.counts["sol_official_calls"] += 1
    if sink_tokens:
        output[:, sink_start:] = F.scaled_dot_product_attention(
            q[:, sink_start:].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False).transpose(1, 2)
        controller.counts["sol_dense_context_queries"] += 1
    return output if return_bthd else output.permute(0, 2, 1, 3).contiguous()


class _H3SparseProcessor:
    def __init__(self, layer: int, original, controller: _Controller):
        self.layer = layer
        self.original = original
        self.controller = controller

    def _dense(self, attn, hidden_states, rotary_emb, attention_mask, reason: str):
        self.controller.counts[f"dense:{reason}"] += 1
        return self.original(attn, hidden_states, rotary_emb, attention_mask)

    @staticmethod
    def _apply_rotary_low_memory(hidden_states, rotary_emb, chunk_rows=32_768):
        """Apply the official rotary function in sequence chunks, reusing Q/K storage."""
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        if hidden_states.shape[1] <= chunk_rows:
            return _apply_rotary_emb(hidden_states, *rotary_emb)
        cos, sin = rotary_emb
        for start in range(0, hidden_states.shape[1], chunk_rows):
            end = min(start + chunk_rows, hidden_states.shape[1])
            rotated = _apply_rotary_emb(
                hidden_states[:, start:end], cos[start:end], sin[start:end]
            )
            hidden_states[:, start:end].copy_(rotated)
        return hidden_states

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        controller = self.controller
        layout = controller.layout
        if controller.is_warmup:
            return self._dense(attn, hidden_states, rotary_emb, attention_mask, "warmup")
        if controller.evaluation_index in controller.config.sol_extra_dense_evaluations:
            return self._dense(
                attn,
                hidden_states,
                rotary_emb,
                attention_mask,
                "extra_dense_evaluation",
            )
        if self.layer < controller.config.sol_dense_layers:
            return self._dense(
                attn,
                hidden_states,
                rotary_emb,
                attention_mask,
                "dense_layer",
            )
        if self.layer in controller.config.sol_extra_dense_layers:
            return self._dense(
                attn,
                hidden_states,
                rotary_emb,
                attention_mask,
                "extra_dense_layer",
            )
        if layout is None:
            raise RuntimeError("sparse H3 processor did not receive packed-layout metadata")
        if attention_mask is not None:
            raise RuntimeError("sparse H3 processor does not accept an external attention mask")
        if hidden_states.shape[0] != 1:
            raise RuntimeError("official sparse kernels currently require packed H3 batch size 1")
        if hidden_states.shape[1] != layout.sequence_length:
            raise RuntimeError(
                "packed layout length does not match H3 hidden states: "
                f"{layout.sequence_length} != {hidden_states.shape[1]}"
            )

        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        projections_preprocessed = False
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            shared_qkv = None
            if hasattr(attn.to_q, "forward_quantized"):
                # fp8-converted projections: one quantisation of x feeds all three.
                from .fp8_linear import forward_qkv_shared
                shared_qkv = forward_qkv_shared(attn, hidden_states)
            if shared_qkv is not None:
                query, key, value = shared_qkv
            else:
                # At 2x SelfLift resolution each full Q/K/V projection is
                # several GiB.  Materialize and preprocess them sequentially
                # so Q rotary does not transiently coexist with raw K and V.
                # This preserves the exact projection/norm/rotary operations;
                # only tensor lifetime changes.
                query = attn.norm_q(
                    attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
                )
                if rotary_emb is not None:
                    query = self._apply_rotary_low_memory(query, rotary_emb)
                key = attn.norm_k(
                    attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
                )
                if rotary_emb is not None:
                    key = self._apply_rotary_low_memory(key, rotary_emb)
                value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
                projections_preprocessed = True
        if not projections_preprocessed:
            query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
            key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
            value = value.unflatten(-1, (attn.heads, -1))
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
                key = _apply_rotary_emb(key, *rotary_emb)
        query_dtype = query.dtype

        permutation = layout.permutation
        # Reblock one projection at a time and release its producer-order
        # storage immediately.  A tuple comprehension keeps all three source
        # projections alive until all three packed copies exist; at SelfLift
        # 2x resolution that transiently adds more than 10 GiB for no numeric
        # benefit and can exhaust even a 96 GiB device.
        packed_q = query.index_select(1, permutation)
        del query
        packed_k = key.index_select(1, permutation)
        del key
        packed_v = value.index_select(1, permutation)
        del value
        from .spark_integration import _sm80_low_memory_reblock
        native_bthd = (
            controller.config.sol_landmark_preprocess
            and _sm80_low_memory_reblock(packed_q)
        )
        if native_bthd:
            q, k, v = packed_q, packed_k, packed_v
        else:
            q, k, v = (
                tensor.permute(0, 2, 1, 3)
                for tensor in (packed_q, packed_k, packed_v)
            )
        output = _sol_attention(
            controller, q, k, v, layout, self.layer,
            return_bthd=True, inputs_bthd=native_bthd,
        )
        if native_bthd:
            # Q is dead after attention and has the exact BTHD destination
            # shape. Reuse it for the producer-order gather so the long SM80
            # path does not retain another full attention output at to_out.
            torch.index_select(
                output, 1, layout.inverse_permutation, out=packed_q
            )
            output = packed_q
            del q, k, v, packed_k, packed_v
        else:
            output = output.index_select(1, layout.inverse_permutation)
        output = output.flatten(2, 3).to(query_dtype)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        controller.counts["sparse:sol"] += 1
        return output


class H3SparseAttentionPlugin:
    """Context-managed, reversible processor installation on one H3 transformer."""

    def __init__(self, transformer, config: H3SparseAttentionConfig):
        self.transformer = transformer
        self.config = config
        self.controller = _Controller(config)
        self._originals: list[tuple[Any, Any]] = []
        self._hook = None
        self._active = False

    def __enter__(self) -> "H3SparseAttentionPlugin":
        if self._active:
            raise RuntimeError("plugin is already active")
        self.controller.reset()
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None:
            raise TypeError("expected a MiniMax-H3 transformer with transformer_blocks")
        try:
            self._hook = self.transformer.register_forward_pre_hook(
                self.controller.begin_forward, with_kwargs=True
            )
            for layer, block in enumerate(blocks):
                attn = block.attn
                original = attn.get_processor()
                self._originals.append((attn, original))
                attn.set_processor(_H3SparseProcessor(layer, original, self.controller))
            self._active = True
            return self
        except Exception:
            self.remove()
            raise

    def reset(self) -> None:
        self.controller.reset()

    def summary(self) -> dict[str, Any]:
        return self.controller.summary()

    def remove(self) -> None:
        for attn, original in reversed(self._originals):
            attn.set_processor(original)
        self._originals.clear()
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self.controller.layout = None
        self.controller.rope_sol_key_clustering_static.clear()
        self.controller._virtual_query_layout_cache.clear()
        self.controller.landmark_reblock_hierarchy = None
        self._active = False

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.remove()
        return False


def install_h3_sparse_attention(
    transformer, config: H3SparseAttentionConfig
) -> H3SparseAttentionPlugin:
    """Return a context manager that installs and later restores H3 processors."""

    return H3SparseAttentionPlugin(transformer, config)


def install_h3_sol_attn(
    transformer, num_inference_steps: int = 50, **config_overrides
) -> H3SparseAttentionPlugin:
    """Return the official-policy Sol-Attn plugin for MiniMax-H3."""

    return install_h3_sparse_attention(
        transformer,
        H3SparseAttentionConfig.sol(num_inference_steps, **config_overrides),
    )


def install_h3_spark_attn(
    transformer, num_inference_steps: int = 20, **config_overrides
) -> H3SparseAttentionPlugin:
    """Return the Spark plugin: Sol TopK10, ungrouped LMv2, global reweight."""
    return install_h3_sparse_attention(
        transformer,
        H3SparseAttentionConfig.spark(num_inference_steps, **config_overrides),
    )
