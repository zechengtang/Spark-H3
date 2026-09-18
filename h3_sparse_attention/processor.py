"""Reversible Sol-Attn integration for packed MiniMax-H3 inference.

Adapted from MiniMax-H3-Sparse. Target video uses Sol-Attn; conditioning
video, text, and audio keys remain exact, and context queries run densely.
"""
from __future__ import annotations

import math
import os

_SOL_LAYOUT_FAST = os.environ.get("H3_SOL_LAYOUT_FAST", "1") == "1"
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
    sol_force_local_blocks: bool | None = None

    sol_route_topk_ratio: float | None = None
    sol_route_topk_cutoff_mode: Literal[
        "gemm_radix", "gaussian_moments"
    ] = "gemm_radix"
    sol_log_density: bool = True
    sol_landmark_preprocess: bool = False
    sol_landmark_preprocess_version: Literal["v1", "v2"] = "v1"
    landmark_tree_v2_minimum_frames: int | None = None
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
    landmark_tree_v2_children: int | list[int] | tuple[int, ...] = 8
    landmark_tree_v2_fanout: int | list[int] | tuple[int, ...] | None = None
    landmark_tree_v2_final_fanout: int | list[int] | tuple[int, ...] | None = None
    landmark_tree_v2_root_fanout: int | None = None
    landmark_tree_v2_landmark_mode: Literal["mean", "midpoint"] = "midpoint"
    landmark_tree_v2_landmark_count: int = 32
    landmark_tree_v2_aggregation: Literal["linear", "max"] = "linear"
    rope_sol_key_ridge_epsilon: float = 1e-3

    sol_route_global_weighted_mean: bool = False
    sol_route_global_weighted_side: Literal["both", "query", "key"] = "both"
    landmark_tree_v2_chunk_frames: int | None = None
    landmark_tree_v2_fanout_mode: Literal[
        "power_of_two_fanout", "arbitrary_fanout"
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
        if self.sol_force_local_blocks is not None and type(self.sol_force_local_blocks) is not bool:
            raise ValueError("sol_force_local_blocks must be bool")

        if self.sol_route_topk_ratio is not None and not 0 < self.sol_route_topk_ratio <= 1:
            raise ValueError("sol_route_topk_ratio must lie in (0, 1]")
        if self.sol_route_topk_cutoff_mode not in ("gemm_radix", "gaussian_moments"):
            raise ValueError("invalid sol_route_topk_cutoff_mode")
        if self.sol_landmark_preprocess and self.sol_landmark_preprocess_version != "v2":
            raise ValueError("this port supports landmark preprocessing version 'v2' only")
        if self.landmark_tree_v2_minimum_frames is not None and (
            type(self.landmark_tree_v2_minimum_frames) is not int or self.landmark_tree_v2_minimum_frames < 0):
            raise ValueError("landmark_tree_v2_minimum_frames must be None or a nonnegative integer")
        if self.landmark_tree_v2_chunk_frames is not None and (
            type(self.landmark_tree_v2_chunk_frames) is not int or self.landmark_tree_v2_chunk_frames < 0):
            raise ValueError("landmark_tree_v2_chunk_frames must be None or a nonnegative integer")
        if self.landmark_tree_v2_minimum_frames and self.landmark_tree_v2_chunk_frames:
            raise ValueError("choose either minimum_frames or chunk_frames")
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
        if self.sol_virtual_query_route_score not in ("native_mean", "mean", "weighted_k", "weighted_mass"):
            raise ValueError("invalid sol_virtual_query_route_score")
        virtual_query_enabled = (
            self.sol_virtual_query_levels_up is not None
            or self.sol_virtual_query_target_blocks is not None
        )
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

        object.__setattr__(self, "landmark_tree_v2_children",
                           _normalize_fanout(self.landmark_tree_v2_children, self.landmark_tree_v2_fanout))
        if self.landmark_tree_v2_fanout is not None:
            object.__setattr__(self, "landmark_tree_v2_fanout", self.landmark_tree_v2_children)
        if self.landmark_tree_v2_root_fanout is not None:
            resolve_root_fanout(self.landmark_tree_v2_children, self.landmark_tree_v2_root_fanout)
        if self.landmark_tree_v2_final_fanout is not None:
            object.__setattr__(self, "landmark_tree_v2_final_fanout",
                               normalize_final_fanout(self.landmark_tree_v2_final_fanout))
        if self.landmark_tree_v2_landmark_count not in (32, 128, 256):
            raise ValueError("landmark_tree_v2_landmark_count must be 32, 128, or 256")
        if self.landmark_tree_v2_landmark_mode not in ("mean", "midpoint"):
            raise ValueError("landmark_tree_v2_landmark_mode must be mean or midpoint")
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
        if self.sol_force_local_blocks is not None:
            return self.sol_force_local_blocks
        return not self.sol_landmark_preprocess

    @classmethod
    def sol(cls, num_inference_steps: int = 50, **overrides):
        if type(num_inference_steps) is not int or num_inference_steps < 2:
            raise ValueError("num_inference_steps must be an integer of at least 2")
        return cls(total_evaluations=num_inference_steps - 1, **overrides)

    @classmethod
    def spark(cls, num_inference_steps: int = 20, **overrides) -> "H3SparseAttentionConfig":
        """Sol TopK10 + minimum-10 fanout-16 LMv2 + target-189 reweighting."""
        defaults = dict(
            sol_route_topk_ratio=0.1,
            sol_route_topk_cutoff_mode="gemm_radix",
            sol_force_local_blocks=False,
            sol_landmark_preprocess=True,
            sol_landmark_preprocess_version="v2",
            landmark_tree_v2_minimum_frames=10,
            landmark_tree_v2_chunk_frames=0,
            landmark_tree_v2_children=16,
            sol_virtual_query_target_blocks=SPARK_REWEIGHT_TARGET_BLOCKS,
            sol_virtual_query_levels_up=None,
            sol_virtual_query_min_blocks=SPARK_REWEIGHT_MIN_BLOCKS,
            sol_virtual_query_max_blocks=SPARK_REWEIGHT_MAX_BLOCKS,
            sol_virtual_query_route_score="native_mean",
        )
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


def _packed_layout(
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    video_indices: torch.Tensor | None = None,
    timestep_indices: torch.Tensor | None = None,
    text_indices: torch.Tensor | None = None,
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
        video = video_indices[target_start:]
        if video[-1] != token_tags.numel() - 1:
            raise ValueError("H3 target video rows must be the final packed-sequence suffix")
        target_timestep = timestep_indices[text_indices[0]]
        if not bool((timestep_indices.index_select(0, video) == target_timestep).all()):
            raise ValueError("H3 target video timestep does not match the text timestep")
        if video.numel() and not bool((token_tags.index_select(0, video) == 0).all()):
            raise ValueError("H3 video_indices selected rows whose modality tag is not video")
    else:
        # Minimal/test transformers may expose only tags and positions. The real
        # H3 pipeline always takes the indexed path above.
        video = torch.nonzero(token_tags == 0, as_tuple=False).flatten()
    selected = torch.zeros(token_tags.numel(), device=token_tags.device, dtype=torch.bool)
    selected[video] = True
    nonvideo = torch.nonzero(~selected, as_tuple=False).flatten()
    if video.numel() == 0:
        raise ValueError("packed H3 sequence contains no video tokens")
    pos = position_ids.index_select(0, video)
    unique_t, unique_h, unique_w = (pos[:, axis].unique(sorted=True) for axis in range(3))
    grid = (unique_t.numel(), unique_h.numel(), unique_w.numel())
    key = (
        torch.searchsorted(unique_t, pos[:, 0].contiguous()) * grid[1] * grid[2]
        + torch.searchsorted(unique_h, pos[:, 1].contiguous()) * grid[2]
        + torch.searchsorted(unique_w, pos[:, 2].contiguous())
    )
    order = key.argsort()
    sorted_key = key.index_select(0, order)
    expected = torch.arange(math.prod(grid), device=key.device)
    if sorted_key.numel() != expected.numel() or not torch.equal(sorted_key, expected):
        raise ValueError(f"video tokens do not form a dense grid: grid={grid}, rows={video.numel()}")

    permutation = torch.cat((video.index_select(0, order), nonvideo))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)

    return PackedLayout(
        permutation=permutation,
        inverse_permutation=inverse,
        grid=grid,
        video_tokens=video.numel(),
        sequence_length=token_tags.numel(),
        video_positions=pos.index_select(0, order),
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
            text_indices=argument("text_indices", 9))

    @property
    def is_warmup(self):
        return self.evaluation_index < self.config.dense_evaluations

    def summary(self):
        return dict(method="sol", sol_backend=self.sol_backend,
                    completed_evaluations=self.evaluation_index + 1,
                    total_evaluations=self.config.total_evaluations,
                    dense_evaluations=self.config.dense_evaluations,
                    processor_calls=dict(self.counts),
                    sol_route_density=self.sol_route_density,
                    sol_virtual_query_layout=self.sol_virtual_query_layout,
                    sol_force_local_blocks=self.config.sol_local_blocks_enabled,
                    sol_landmark_preprocess=self.config.sol_landmark_preprocess,
                    landmark_tree_v2_minimum_frames=self.config.landmark_tree_v2_minimum_frames,
                    landmark_tree_v2_children=self.config.landmark_tree_v2_children,
                    landmark_tree_v2_chunk_frames=self.config.landmark_tree_v2_chunk_frames,
                    landmark_tree_v2_fanout_mode=self.config.landmark_tree_v2_fanout_mode)


def _sol_attention(controller, q, k, v, layout, layer, *, return_bthd=False):
    from sol_attn import get_sol_attn_backend, sol_attn

    cfg = controller.config
    if cfg.sol_landmark_preprocess or cfg.sol_route_topk_ratio is not None:
        from .spark_integration import spark_attention
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

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        controller = self.controller
        layout = controller.layout
        if controller.is_warmup:
            return self._dense(attn, hidden_states, rotary_emb, attention_mask, "warmup")
        if self.layer < controller.config.sol_dense_layers:
            return self._dense(
                attn,
                hidden_states,
                rotary_emb,
                attention_mask,
                "dense_layer",
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

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = (
                attn.to_q(hidden_states),
                attn.to_k(hidden_states),
                attn.to_v(hidden_states),
            )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        permutation = layout.permutation
        q, k, v = (
            tensor.index_select(1, permutation).permute(0, 2, 1, 3)
            for tensor in (query, key, value)
        )
        if not _SOL_LAYOUT_FAST:
            q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        output = _sol_attention(controller, q, k, v, layout, self.layer,
                                return_bthd=_SOL_LAYOUT_FAST)
        if not _SOL_LAYOUT_FAST:
            output = output.permute(0, 2, 1, 3)
        output = output.index_select(1, layout.inverse_permutation)
        output = output.flatten(2, 3).type_as(query)
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
    """Return the Spark plugin: Sol TopK10, minimum-10, target-189 reweight."""
    return install_h3_sparse_attention(
        transformer,
        H3SparseAttentionConfig.spark(num_inference_steps, **config_overrides),
    )
