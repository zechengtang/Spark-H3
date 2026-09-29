"""ComfyUI model patch for the native MiniMax-H3 implementation.

The native ComfyUI model packs tokens as ``[conditioning | target audio |
target video]`` while Spark-H3 expects the target video first.  This adapter
uses ComfyUI's published ``minimax_h3_layout`` to move only the target-video
rows to the front, runs Spark, and restores the original order afterwards.
"""
from __future__ import annotations

import logging
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

try:
    from .comfyui_backend import (
        ComfyPackedLayout,
        ComfySparkConfig,
        ComfySparkController,
        comfy_kitchen_spark_attention,
        reblock_profile_summary,
    )
except ImportError:  # standalone source-tree tests
    from comfyui_backend import (
        ComfyPackedLayout,
        ComfySparkConfig,
        ComfySparkController,
        comfy_kitchen_spark_attention,
        reblock_profile_summary,
    )


log = logging.getLogger(__name__)
_SM120 = (12, 0)
_PRODUCER_CHUNK = int(os.environ.get("H3_SPARK_PRODUCER_CHUNK", "16384"))
if _PRODUCER_CHUNK <= 0:
    raise ValueError("H3_SPARK_PRODUCER_CHUNK must be positive")


class _Unsupported(RuntimeError):
    pass


class _Incompatible(_Unsupported):
    pass


def _target_video_span(layout, sequence_length: int) -> tuple[int, int]:
    segments = getattr(layout, "segments", ())
    spans = [(int(start), int(stop)) for start, stop, kind in segments if kind == "video"]
    if len(spans) != 1:
        raise _Incompatible("minimax_h3_layout does not contain exactly one target video segment")
    start, stop = spans[0]
    if not (0 <= start < stop <= sequence_length):
        raise _Incompatible("target video segment is outside the packed sequence")
    return start, stop


def _make_spark_layout(layout, sequence_length: int, device: torch.device) -> ComfyPackedLayout:
    start, stop = _target_video_span(layout, sequence_length)
    position_ids = getattr(layout, "position_ids", None)
    if not torch.is_tensor(position_ids) or position_ids.shape != (sequence_length, 3):
        raise _Incompatible("minimax_h3_layout.position_ids has an unexpected shape")

    video_positions = position_ids[start:stop].to(device=device)
    unique_t, unique_h, unique_w = (
        video_positions[:, axis].unique(sorted=True) for axis in range(3)
    )
    grid = (unique_t.numel(), unique_h.numel(), unique_w.numel())
    if math.prod(grid) != stop - start:
        raise _Incompatible(
            f"target video rows do not form a dense grid: grid={grid}, rows={stop - start}"
        )

    video = torch.arange(start, stop, device=device, dtype=torch.long)
    before = torch.arange(0, start, device=device, dtype=torch.long)
    after = torch.arange(stop, sequence_length, device=device, dtype=torch.long)
    permutation = torch.cat((video, before, after))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(sequence_length, device=device)
    return ComfyPackedLayout(
        permutation=permutation,
        inverse_permutation=inverse,
        grid=grid,
        video_tokens=stop - start,
        sequence_length=sequence_length,
        video_positions=video_positions,
        video_start=start,
    )


@dataclass
class _RunState:
    steps: int
    warmup_percent: float
    controller: ComfySparkController
    warmup_mode: str = "warmup_percent"
    warmup_steps: int = 4
    previous_sigma: float | None = None
    layout_cache: dict[tuple[int, int, str], ComfyPackedLayout] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        steps: int,
        warmup_percent: float,
        topk_ratio: float,
        ablation_mode: str = "full",
        video_tail_mode: str = "dense",
        global_anchor_dtype: str = "float32",
        tail_granularity: str = "query",
        midpoint_direction_mode: str = "fused",
        topk_mode: str = "topk_ratio",
        topk_blocks: int = 228,
        warmup_mode: str = "warmup_percent",
        warmup_steps: int = 4,
    ):
        if warmup_mode not in ("warmup_percent", "warmup_steps"):
            raise ValueError("warmup_mode must be 'warmup_percent' or 'warmup_steps'")
        if not 0.0 <= warmup_percent <= 100.0:
            raise ValueError("warmup_percent must lie in [0, 100]")
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        # Spark's public factory models the Diffusers pipeline's N-1 evaluations.
        # ComfyUI executes one model evaluation per sampler step, hence steps + 1.
        common = dict(
            topk_ratio=topk_ratio,
            topk_mode=topk_mode,
            topk_blocks=topk_blocks,
            video_tail_mode=video_tail_mode,
            global_anchor_dtype=global_anchor_dtype,
            tail_granularity=tail_granularity,
            landmark_tree_v2_midpoint_direction_mode=midpoint_direction_mode,
        )
        if ablation_mode == "full":
            config = ComfySparkConfig(**common)
        elif ablation_mode == "full_reuse2":
            config = ComfySparkConfig(**common)
        elif ablation_mode == "full_group841":
            config = ComfySparkConfig(
                landmark_tree_v2_group_size=(8, 4, 1),
                **common,
            )
        elif ablation_mode == "topk_only":
            config = ComfySparkConfig(**common)
        else:
            raise ValueError(
                f"unsupported ComfyUI Spark mode {ablation_mode!r}; only "
                "comfy-kitchen global modes are available"
            )
        state = cls(
            steps, warmup_percent, ComfySparkController(config),
            warmup_mode=warmup_mode, warmup_steps=warmup_steps,
        )
        state.controller.topk_only = ablation_mode == "topk_only"
        # The exact kernel can scatter video-first rows directly into H3's
        # natural layout. Keep an opt-out for regression A/B comparisons.
        state.controller.direct_output = os.environ.get("H3_SPARK_DIRECT_OUTPUT", "1") == "1"
        state.controller.reblock_reuse_layers = 2 if ablation_mode == "full_reuse2" else 1
        state.controller.reblock_reuse_start_layer = 1
        return state

    @property
    def warmup_evaluations(self) -> int:
        if self.warmup_mode == "warmup_steps":
            return min(self.steps, self.warmup_steps)
        return min(self.steps, math.ceil(self.steps * self.warmup_percent / 100.0))

    def begin_evaluation(self, transformer_options) -> None:
        sigmas = (transformer_options or {}).get("sigmas")
        sigma = (
            float(sigmas.flatten()[0])
            if torch.is_tensor(sigmas) and sigmas.numel() > 0
            else None
        )
        next_index = self.controller.evaluation_index + 1
        new_run = next_index >= self.steps
        if sigma is not None and self.previous_sigma is not None:
            new_run = new_run or sigma > self.previous_sigma + 1e-7
        if new_run:
            self.controller.reset()
            self.layout_cache.clear()
            next_index = 0
        self.controller.evaluation_index = next_index
        self.previous_sigma = sigma

    @property
    def is_warmup(self) -> bool:
        return self.controller.evaluation_index < self.warmup_evaluations

    def spark_layout(self, layout, sequence_length: int, device: torch.device):
        key = (id(layout), sequence_length, str(device))
        cached = self.layout_cache.get(key)
        if cached is None:
            cached = _make_spark_layout(layout, sequence_length, device)
            self.layout_cache[key] = cached
        return cached


class _ActivationLog:
    def __init__(self):
        self.active = False
        self.fallbacks: set[str] = set()

    def hit(self, tokens: int, video_tokens: int) -> None:
        if not self.active:
            log.info(
                "[Spark-H3] active on SM120 (%d packed tokens, %d target-video tokens)",
                tokens,
                video_tokens,
            )
            self.active = True

    def miss(self, reason: str) -> None:
        if reason not in self.fallbacks:
            self.fallbacks.add(reason)
            log.warning("[Spark-H3] dense fallback: %s", reason)

    def reset(self) -> None:
        self.active = False
        self.fallbacks.clear()


def _native_spark_eligible(attn, x, rope_freqs, transformer_options, layer, policy, state, strict):
    if layer == 0:
        state.begin_evaluation(transformer_options)
    if not torch.is_tensor(x) or x.ndim != 2:
        if strict:
            raise _Incompatible("attention input is not a rank-2 tensor")
        return "attention input is not a rank-2 tensor", True
    if state.is_warmup:
        return (
            f"dense warmup evaluation {state.controller.evaluation_index + 1}/"
            f"{state.warmup_evaluations}",
            False,
        )
    reason = policy.dense_reason(transformer_options, x.shape[0], layer)
    if reason is not None:
        return reason, False
    if rope_freqs is None:
        reason = "MiniMax-H3 RoPE table is unavailable"
        if strict:
            raise _Incompatible(reason)
        return reason, True
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        reason = "Spark SM120 requires CUDA bfloat16 activations"
    elif int(attn.head_dim) != 128:
        reason = f"head_dim {attn.head_dim} != 128"
    elif torch.cuda.get_device_capability(x.device) != _SM120:
        capability = torch.cuda.get_device_capability(x.device)
        reason = f"this node targets SM120, found SM{capability[0]}{capability[1]}"
    elif transformer_options.get("minimax_h3_layout") is None:
        reason = "ComfyUI did not publish minimax_h3_layout"
    else:
        return None, False
    if strict:
        raise _Incompatible(reason)
    return reason, True


def _native_spark_qkv(attn, x, rope_freqs, permutation, *, video_start=None, video_tokens=None):
    """ComfyUI-style chunked H3 producer yielding contiguous BTHD Q/K/V."""
    import comfy.model_management
    import comfy.model_prefetch
    from comfy_kitchen.backends import cuda as ck

    tokens = x.shape[0]
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    inner = heads * head_dim
    q = torch.empty((1, tokens, heads, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    rot_dim = rope_freqs.shape[-3] * 2
    stream = torch.cuda.current_stream(x.device).cuda_stream
    wrap = ck._wrap_for_dlpack
    bsa_materializer = os.environ.get("H3_SPARK_BSA_MATERIALIZER", "0") == "1"
    bsa_original = os.environ.get("H3_SPARK_BSA_ORIGINAL_PRODUCER", "0") == "1"
    if bsa_materializer and bsa_original:
        raise RuntimeError("select only one experimental BSA producer")
    if bsa_materializer and not hasattr(ck._C, "bsa_materialize_qkv_chunk"):
        raise RuntimeError("H3_SPARK_BSA_MATERIALIZER requires rebuilt comfy-kitchen CUDA extension")
    if bsa_materializer and not getattr(_native_spark_qkv, "_bsa_logged", False):
        print("H3_SPARK_BSA_MATERIALIZER active: BSA tile BF16 QKV materializer, Spark downstream", flush=True)
        _native_spark_qkv._bsa_logged = True
    if bsa_original:
        if not hasattr(ck._C, "sol_producer_chunk_materialize"):
            raise RuntimeError("H3_SPARK_BSA_ORIGINAL_PRODUCER requires rebuilt comfy-kitchen CUDA extension")
        if not getattr(_native_spark_qkv, "_bsa_original_logged", False):
            print("H3_SPARK_BSA_ORIGINAL_PRODUCER active: original sol_producer_kernel plus BF16 QKV, Spark downstream", flush=True)
            _native_spark_qkv._bsa_original_logged = True
        bsa_plan = ck._C.sol_attn_plan(1, tokens, heads)
        bsa_workspace = torch.empty(bsa_plan["total"], dtype=torch.uint8, device=x.device)
        stale_kmean = torch.zeros((heads, head_dim), dtype=torch.float32, device=x.device)
        stale_vscale = torch.ones((heads, head_dim), dtype=torch.float32, device=x.device)
        ck._C.sol_producer_begin(wrap(bsa_workspace), 1, tokens, heads, stream, 0)

    with comfy.model_prefetch.pause_malloc_graph():
        for start in range(0, tokens, _PRODUCER_CHUNK):
            stop = min(start + _PRODUCER_CHUNK, tokens)
            offset = None
            if video_start is not None and video_tokens is not None:
                video_end = video_start + video_tokens
                if stop <= video_tokens:
                    offset = video_start
                elif start >= video_tokens and stop <= video_end:
                    offset = -video_tokens
                elif start >= video_end:
                    offset = 0
            if offset is None:
                indices = permutation[start:stop]
                source = x.index_select(0, indices)
                freqs = rope_freqs.index_select(1, indices)
            else:
                source = x[start + offset:stop + offset]
                freqs = rope_freqs[:, start + offset:stop + offset]
            projected = attn.qkv_proj(source)
            if bsa_original:
                fab = ck._packed_rope_fab(freqs, stop - start, rot_dim)
                ck._C.sol_producer_chunk_materialize(
                    *(wrap(t) for t in (
                        bsa_workspace, projected.contiguous(), fab, qw, kw,
                        stale_kmean, stale_vscale, q, k, v,
                    )),
                    attn.q_norm.eps, rot_dim, start, stop - start,
                    tokens, heads, stream,
                )
                verify_path = os.environ.get("H3_SPARK_BSA_ORIGINAL_VERIFY_PATH")
                if start == 0 and verify_path and not getattr(_native_spark_qkv, "_bsa_original_verified", False):
                    qc, kc, vc = projected.split(inner, dim=-1)
                    qc = qc.view(1, stop - start, heads, head_dim)
                    kc = kc.view(1, stop - start, heads, head_dim)
                    qref = torch.empty_like(q[:, start:stop])
                    kref = torch.empty_like(k[:, start:stop])
                    ck._C.rms_rope(
                        *(wrap(t) for t in (qc, kc, freqs, qw, kw, qref, kref)),
                        attn.q_norm.eps, stream, True, rot_dim,
                    )
                    vref = vc.view(1, stop - start, heads, head_dim)
                    parity = {}
                    for name, actual, reference in (
                        ("q", q[:, start:stop], qref),
                        ("k", k[:, start:stop], kref),
                        ("v", v[:, start:stop], vref),
                    ):
                        delta = (actual.float() - reference.float()).abs()
                        parity[name] = {
                            "bitwise": bool(torch.equal(actual, reference)),
                            "equal_fraction": float((actual == reference).float().mean()),
                            "max_abs": float(delta.max()),
                            "rmse": float(delta.square().mean().sqrt()),
                        }
                    Path(verify_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(verify_path).write_text(json.dumps({
                        "tokens": stop - start, "heads": heads, "rot_dim": rot_dim,
                        "producer_workspace_bytes": int(bsa_plan["total"]),
                        "parity": parity,
                    }, indent=2))
                    _native_spark_qkv._bsa_original_verified = True
                continue
            if bsa_materializer:
                # This experimental entry reuses the official BSA producer's
                # 64-row tile RMSNorm/RoPE, but retains BF16 Q/K/V so Spark's
                # reblock, reweight, route and exact kernels remain unchanged.
                fab = ck._packed_rope_fab(freqs, stop - start, rot_dim)
                verify_path = os.environ.get("H3_SPARK_BSA_VERIFY_PATH")
                verify = start == 0 and verify_path and not getattr(_native_spark_qkv, "_bsa_verified", False)
                if verify:
                    candidate_start = torch.cuda.Event(enable_timing=True)
                    candidate_end = torch.cuda.Event(enable_timing=True)
                    candidate_start.record()
                ck._C.bsa_materialize_qkv_chunk(
                    *(wrap(t) for t in (projected.contiguous(), fab, qw, kw, q, k, v)),
                    attn.q_norm.eps, rot_dim, start, stop - start,
                    tokens, heads, stream,
                )
                if verify:
                    candidate_end.record()
                    qc, kc, vc = projected.split(inner, dim=-1)
                    qc = qc.view(1, stop - start, heads, head_dim)
                    kc = kc.view(1, stop - start, heads, head_dim)
                    qref = torch.empty_like(q[:, start:stop])
                    kref = torch.empty_like(k[:, start:stop])
                    baseline_start = torch.cuda.Event(enable_timing=True)
                    baseline_end = torch.cuda.Event(enable_timing=True)
                    baseline_start.record()
                    ck._C.rms_rope(
                        *(wrap(t) for t in (qc, kc, freqs, qw, kw, qref, kref)),
                        attn.q_norm.eps, stream, True, rot_dim,
                    )
                    vref = vc.view(1, stop - start, heads, head_dim).clone()
                    baseline_end.record()
                    baseline_end.synchronize()
                    parity = {}
                    for name, actual, reference in (
                        ("q", q[:, start:stop], qref),
                        ("k", k[:, start:stop], kref),
                        ("v", v[:, start:stop], vref),
                    ):
                        delta = (actual.float() - reference.float()).abs()
                        parity[name] = {
                            "bitwise": bool(torch.equal(actual, reference)),
                            "equal_fraction": float((actual == reference).float().mean()),
                            "max_abs": float(delta.max()),
                            "rmse": float(delta.square().mean().sqrt()),
                        }
                    Path(verify_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(verify_path).write_text(json.dumps({
                        "tokens": stop - start, "heads": heads, "rot_dim": rot_dim,
                        "candidate_kernel_ms": candidate_start.elapsed_time(candidate_end),
                        "baseline_rope_copy_ms": baseline_start.elapsed_time(baseline_end),
                        "parity": parity,
                    }, indent=2))
                    _native_spark_qkv._bsa_verified = True
                continue
            qc, kc, vc = projected.split(inner, dim=-1)
            qc = qc.view(1, stop - start, heads, head_dim)
            kc = kc.view(1, stop - start, heads, head_dim)
            ck._C.rms_rope(
                *(wrap(t) for t in (qc, kc, freqs, qw, kw,
                                      q[:, start:stop], k[:, start:stop])),
                attn.q_norm.eps, stream, True, rot_dim,
            )
            v[:, start:stop].copy_(vc.view(1, stop - start, heads, head_dim))
    return q, k, v


def _native_spark_attention(attn, x, rope_freqs, transformer_options, layer, state, activation_log):
    layout = state.spark_layout(
        transformer_options["minimax_h3_layout"], x.shape[0], x.device
    )
    qkv_profile = None
    if os.environ.get("SPARK_PROFILE_REBLOCK") == "1":
        qkv_start = torch.cuda.Event(enable_timing=True)
        qkv_end = torch.cuda.Event(enable_timing=True)
        qkv_start.record()
        qkv_profile = ("qkv_materialize", qkv_start, qkv_end)
    q, k, v = _native_spark_qkv(
        attn, x, rope_freqs, layout.permutation,
        video_start=layout.video_start, video_tokens=layout.video_tokens,
    )
    if qkv_profile is not None:
        qkv_profile[2].record()
        state.controller.reblock_profile.append(qkv_profile)
    output = comfy_kitchen_spark_attention(state.controller, q, k, v, layout, layer)
    unpack_profile = None
    if os.environ.get("SPARK_PROFILE_REBLOCK") == "1":
        unpack_start = torch.cuda.Event(enable_timing=True)
        unpack_end = torch.cuda.Event(enable_timing=True)
        unpack_start.record()
        unpack_profile = ("output_unpack_and_projection", unpack_start, unpack_end)
    if not state.controller.direct_output:
        output = output.index_select(1, layout.inverse_permutation)
    state.controller.counts["sparse:spark_comfy_native"] += 1
    activation_log.hit(x.shape[0], layout.video_tokens)
    output = attn.out_proj(
        output.reshape(x.shape[0], int(attn.heads) * int(attn.head_dim))
    )
    if unpack_profile is not None:
        unpack_profile[2].record()
        state.controller.reblock_profile.append(unpack_profile)
    return output


def _make_native_spark_block_patch(block, layer, policy, state, activation_log, strict):
    def attention(h, rope_freqs=None, transformer_options={}):
        return _native_spark_attention(
            block.attn, h, rope_freqs, transformer_options, layer, state, activation_log
        )

    def block_patch(args, extra):
        reason, warn = _native_spark_eligible(
            block.attn,
            args["img"],
            args["rope_freqs"],
            args["transformer_options"],
            layer,
            policy,
            state,
            strict,
        )
        if reason is not None:
            policy.log_once(("dense", args["img"].shape[0], reason),
                            f"dense ({args['img'].shape[0]} tokens): {reason}")
            if warn:
                activation_log.miss(reason)
            return extra["original_block"](args)
        try:
            return extra["original_block"]({**args, "attention": attention})
        except Exception as exc:
            if strict:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            policy.log_once(("kernel_fallback", reason), f"dense after Spark failure: {reason}")
            activation_log.miss(reason)
            return extra["original_block"](args)

    return block_patch


class MiniMaxH3SparkAttentionSM120:
    """Run Spark through ComfyUI's native MiniMax-H3 block-patch architecture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "接入 MiniMax H3 模型：连接在模型加载节点之后、采样器之前。"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "是否启用 Spark；关闭时原样输出输入模型。"}),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 200, "step": 1,
                     "tooltip": "本次采样的模型调用次数，通常与采样器 steps 相同；用于计算预热阶段。"},
                ),
                "warmup_mode": (["warmup_percent", "warmup_steps"],
                                {"default": "warmup_percent",
                                 "tooltip": "选择 dense 预热长度的表示法：百分比或固定模型调用次数；只会使用对应的 warmup 参数。"}),
                "warmup_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0,
                     "tooltip": "前多少百分比的采样调用保持 dense。建议输入 20–25（代表 20%–25%，不是 0.2–0.25）；仅在 warmup_mode=warmup_percent 时生效。"},
                ),
                "warmup_steps": (
                    "INT",
                    {"default": 4, "min": 0, "max": 200, "step": 1,
                     "tooltip": "开始时保持 dense 的模型调用次数；仅在 warmup_mode=warmup_steps 时生效。20 步采样设 4 相当于 20%。"},
                ),
                "topk_mode": (["topk_ratio", "topk_blocks"],
                              {"default": "topk_ratio",
                               "tooltip": "选择 Top-K 预算表示法：默认 topk_ratio 使用随视频长度变化的比例；topk_blocks 使用固定块数。"}),
                "topk_ratio": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
                     "tooltip": "精确计算的候选视频块比例：0.1≈10%（速度最快），0.3≈30%（保真度最好），建议 0.15 或 0.2 取得均衡；仅在 topk_mode=topk_ratio 时生效。"},
                ),
                "topk_blocks": (
                    "INT",
                    {"default": 228, "min": 1, "max": 262144, "step": 1,
                     "tooltip": "每个 query 块精确计算的候选视频块数。10 秒 768p 样本中 114≈10%、228≈20%、342≈30%；10% 通常最快，30% 保真度较高，建议先用 228。仅在 topk_mode=topk_blocks 时生效。"},
                ),
                "dense_layers": (
                    "INT",
                    {"default": 1, "min": 0, "max": 50, "step": 1,
                     "tooltip": "始终使用 dense attention 的前 N 个 Transformer 层；1 表示第 0 层保持 dense，0 表示不固定任何层。"},
                ),
                "min_tokens": (
                    "INT",
                    {"default": 12288, "min": 256, "max": 262144, "step": 256,
                     "tooltip": "序列 token 数低于该值时保持 dense；用于避免短序列上稀疏计算反而增加开销。"},
                ),
                "strict": ("BOOLEAN", {"default": True,
                                       "tooltip": "启用时，Spark 不兼容的输入会报错；关闭时对不兼容输入回退原始 attention，并在日志记录原因。"}),
            },
            "optional": {
                "tail_granularity": (["query", "block"],
                                     {"default": "query",
                                      "tooltip": "近似分支的 Q 粒度：query（默认）对应 sol-engine Sol，只下采样 K/V、保留逐条真实 Q；block 对应 ComfyUI Sol，Q 和 K/V 都按块下采样，同一 Q 块共享近似结果。精确分支仍使用真实 Q。"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "Patch ComfyUI's native MiniMax H3 DiT with Spark-H3 block-sparse "
        "attention on RTX 50-series SM120 GPUs. Uses ComfyUI's official H3 "
        "block-replacement, sigma scheduling, cleanup, and chunked QKV producer "
        "architecture. Connect it after UNETLoader and set steps to the "
        "sampler's model-evaluation count."
    )

    def patch(
        self,
        model,
        enabled,
        steps,
        warmup_percent,
        topk_ratio,
        dense_layers,
        min_tokens,
        strict,
        ablation_mode="full",
        video_tail_mode="dense",
        tail_granularity="query",
        topk_mode="topk_ratio",
        topk_blocks=228,
        warmup_mode="warmup_percent",
        warmup_steps=4,
        global_anchor_dtype="float32",
        midpoint_direction_mode="fused",
    ):
        if not enabled:
            return (model,)
        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            raise TypeError("Spark-H3 expects ComfyUI's native MiniMaxH3Model")

        import comfy.patcher_extension
        from comfy_extras.nodes_sparse_attention import SparseAttnPatch

        patched = model.clone()
        state = _RunState.create(
            int(steps), float(warmup_percent), float(topk_ratio),
            ablation_mode=str(ablation_mode),
            video_tail_mode=str(video_tail_mode),
            global_anchor_dtype=str(global_anchor_dtype),
            tail_granularity=str(tail_granularity),
            midpoint_direction_mode=str(midpoint_direction_mode),
            topk_mode=str(topk_mode),
            topk_blocks=int(topk_blocks),
            warmup_mode=str(warmup_mode),
            warmup_steps=int(warmup_steps),
        )
        model_sampling = model.get_model_object("model_sampling")
        policy = SparseAttnPatch(
            tau=1.0,
            topk_ratio=float(topk_ratio),
            vsa=False,
            # The legacy node exposes an exact model-evaluation count through
            # steps, so _RunState owns warmup.  Keep the official policy's
            # start bound fully open and use it for end/min-token/block gates.
            sigma_start=float(model_sampling.percent_to_sigma(0.0)),
            sigma_end=float(model_sampling.percent_to_sigma(1.0)),
            min_tokens=int(min_tokens),
            dense_blocks=set(range(int(dense_layers))),
            sink_conditioning="exact_kv_and_rows",
            extra_tokens=0,
            verbose=True,
        )
        activation_log = _ActivationLog()
        for layer, block in enumerate(blocks):
            patched.set_model_patch_replace(
                _make_native_spark_block_patch(
                    block, layer, policy, state, activation_log, bool(strict)
                ),
                "dit",
                "double_block",
                layer,
            )

        def cleanup(_model_patcher):
            profile = reblock_profile_summary(state.controller)
            if profile is not None:
                log.info("[Spark-H3] reblock CUDA profile: %s", profile)
            policy.reset()
            state.controller.reset()
            state.previous_sigma = None
            state.layout_cache.clear()
            activation_log.reset()

        patched.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            "spark_h3_native_attention",
            cleanup,
        )
        log.info(
            "[Spark-H3] installed ComfyUI-native producer on %d H3 blocks "
            "(mode %s, tail %s, anchor %s, approximation %s, midpoint %s, Top-K %s, warmup %s, dense blocks %s)",
            len(blocks),
            str(ablation_mode),
            str(video_tail_mode),
            state.controller.config.global_anchor_dtype,
            str(tail_granularity),
            state.controller.config.landmark_tree_v2_midpoint_direction_mode,
            (f"{int(topk_blocks)} blocks" if topk_mode == "topk_blocks"
             else f"{100.0 * float(topk_ratio):.0f}%"),
            (f"{state.warmup_evaluations} steps" if warmup_mode == "warmup_steps"
             else f"{float(warmup_percent):.0f}%"),
            "none" if int(dense_layers) == 0 else f"0..{int(dense_layers) - 1}",
        )
        return (patched,)


class MiniMaxH3SolAttentionSM120(MiniMaxH3SparkAttentionSM120):
    """Compatibility wrapper around ComfyUI's official sparse-attention patch."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        required = inputs["required"]
        required.pop("topk_mode")
        required.pop("topk_ratio")
        required.pop("topk_blocks")
        required.pop("warmup_mode")
        required.pop("warmup_steps")
        inputs["optional"].pop("tail_granularity")
        required["tau"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        )
        return inputs

    DESCRIPTION = (
        "Compatibility alias for ComfyUI's official MiniMax H3 Sol-Attn "
        "implementation. It uses the official block patch, chunked producer, "
        "and comfy-kitchen kernel; no Spark-H3 Triton/CuTe backend is reachable."
    )

    def patch(
        self,
        model,
        enabled,
        steps,
        warmup_percent,
        dense_layers,
        min_tokens,
        strict,
        tau,
    ):
        if not enabled:
            return (model,)
        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            raise TypeError("Spark-H3 Sol expects ComfyUI's native MiniMaxH3Model")

        from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention

        # Keep the legacy node's inputs for workflow compatibility.  All actual
        # patching and attention execution belongs to ComfyUI's official node.
        del steps, strict
        patched = apply_block_sparse_attention(
            model,
            tau=float(tau),
            topk_ratio=0.0,
            vsa=False,
            start_percent=float(warmup_percent) / 100.0,
            end_percent=1.0,
            min_tokens=int(min_tokens),
            dense_blocks=set(range(int(dense_layers))),
            sink_conditioning="exact_kv_and_rows",
            extra_tokens=0,
            verbose=True,
        )
        log.info(
            "[Spark-H3 Sol] delegated %d MiniMax H3 blocks to ComfyUI's official Sol patch "
            "(tau %.2f, warmup %.0f%%, dense layers %d)",
            len(blocks),
            float(tau),
            float(warmup_percent),
            int(dense_layers),
        )
        return (patched,)


class MiniMaxH3BSAAllExactScheduled:
    """Dense reference with Spark's warmup/layer gates and BSA exact active calls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
            "warmup_percent": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0}),
            "dense_layers": ("INT", {"default": 1, "min": 0, "max": 50}),
            "min_tokens": ("INT", {"default": 12288, "min": 0, "max": 262144}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = "Experiment control: SDPA for warmup and dense layers; full-sink BSA for active H3 attention."

    def patch(self, model, steps, warmup_percent, dense_layers, min_tokens):
        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            raise TypeError("BSA all-exact scheduled control requires MiniMaxH3Model")
        if dense_layers > len(blocks):
            raise ValueError("dense_layers exceeds H3 block count")

        import comfy.patcher_extension
        from comfy_extras.nodes_sparse_attention import SparseAttnPatch, h3_eligible, h3_sparse_attention

        steps = int(steps)
        warmup_count = min(steps, math.ceil(steps * float(warmup_percent) / 100.0))
        policy = SparseAttnPatch(
            tau=1.0, topk_ratio=0.0, vsa=False,
            sigma_start=float("inf"), sigma_end=-float("inf"),
            min_tokens=0, dense_blocks=set(), sink_conditioning="off",
            extra_tokens=0, verbose=True, all_exact=True,
        )
        state = {"evaluation_index": -1, "previous_sigma": None}
        patched = model.clone()

        def advance(transformer_options):
            sigmas = (transformer_options or {}).get("sigmas")
            sigma = float(sigmas.flatten()[0]) if torch.is_tensor(sigmas) and sigmas.numel() else None
            next_index = state["evaluation_index"] + 1
            new_run = next_index >= steps
            if sigma is not None and state["previous_sigma"] is not None:
                new_run = new_run or sigma > state["previous_sigma"] + 1e-7
            if new_run:
                next_index = 0
                policy.reset()
            state["evaluation_index"] = next_index
            state["previous_sigma"] = sigma
            log.info("[BSA-all-exact] evaluation %d/%d: %s; first %d layers SDPA",
                     next_index + 1, steps,
                     "SDPA warmup" if next_index < warmup_count else "BSA active",
                     dense_layers)

        for layer, block in enumerate(blocks):
            def make_block_patch(layer=layer, block=block):
                def attention(h, rope_freqs=None, transformer_options={}):
                    return h3_sparse_attention(
                        block.attn, h, rope_freqs, transformer_options, policy, layer)

                def block_patch(args, extra):
                    if layer == 0:
                        advance(args["transformer_options"])
                    if (state["evaluation_index"] < warmup_count
                            or layer < dense_layers
                            or args["img"].shape[0] < min_tokens):
                        return extra["original_block"](args)
                    if not h3_eligible(block.attn, args["img"], args["rope_freqs"],
                                       args["transformer_options"], policy, layer):
                        raise RuntimeError(f"BSA all-exact expected active H3 layer {layer}, but it was ineligible")
                    return extra["original_block"]({**args, "attention": attention})

                return block_patch

            patched.set_model_patch_replace(make_block_patch(), "dit", "double_block", layer)

        def cleanup(_model_patcher):
            policy.reset()
            state["evaluation_index"] = -1
            state["previous_sigma"] = None

        patched.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            "bsa_all_exact_scheduled_attention", cleanup,
        )
        log.info("[BSA-all-exact] installed on %d H3 layers (warmup %d/%d, dense layers %d, min_tokens %d)",
                 len(blocks), warmup_count, steps, dense_layers, min_tokens)
        return (patched,)


class SaveMiniMaxH3AVLatentCache:
    """Save ComfyUI's two-stream MiniMax-H3 latent without decoding it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "cache_path": ("STRING", {"default": "/tmp/minimax_h3_av.safetensors"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model/latent/minimax"
    DESCRIPTION = "Save the video and audio streams of a MiniMax-H3 latent for deferred VAE decoding."

    def save(self, samples, cache_path):
        import safetensors.torch

        nested = samples.get("samples")
        streams = nested.unbind() if getattr(nested, "is_nested", False) else None
        if streams is None or len(streams) != 2:
            raise TypeError("expected a MiniMax-H3 nested video+audio latent")
        path = Path(cache_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        safetensors.torch.save_file(
            {
                "video": streams[0].detach().to("cpu", copy=True).contiguous(),
                "audio": streams[1].detach().to("cpu", copy=True).contiguous(),
            },
            str(temporary),
            metadata={"format": "minimax_h3_av_latent_v1"},
        )
        temporary.replace(path)
        return {"ui": {"cache_path": [str(path)]}, "result": (samples,)}


class LoadMiniMaxH3AVLatentCache:
    """Load a two-stream latent written by SaveMiniMaxH3AVLatentCache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_path": ("STRING", {"default": "/tmp/minimax_h3_av.safetensors"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "model/latent/minimax"

    @classmethod
    def IS_CHANGED(cls, cache_path):
        path = Path(cache_path).expanduser().resolve()
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def load(self, cache_path):
        import comfy.nested_tensor
        import safetensors.torch

        path = Path(cache_path).expanduser().resolve()
        tensors = safetensors.torch.load_file(str(path), device="cpu")
        if set(tensors) != {"video", "audio"}:
            raise ValueError(f"invalid MiniMax-H3 AV latent cache: {path}")
        return ({"samples": comfy.nested_tensor.NestedTensor((tensors["video"], tensors["audio"]))},)


class SaveVideoLosslessUltrafast:
    """Save a ComfyUI VIDEO as lossless H.264 using the ultrafast preset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "output_path": ("STRING", {"default": "/tmp/comfyui_lossless.mp4"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "video"
    DESCRIPTION = "Lossless H.264/MP4 output with ultrafast CPU encoding for benchmark archives."

    def save(self, video, output_path):
        from comfy_api.latest._util import VideoCodec, VideoContainer

        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
        video.save_to(
            str(temporary),
            format=VideoContainer.MP4,
            codec=VideoCodec.H264,
            crf=0,
            preset="ultrafast",
        )
        temporary.replace(path)
        return {"ui": {"output_path": [str(path)]}}


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SparkAttentionSM120": MiniMaxH3SparkAttentionSM120,
    "MiniMaxH3SolAttentionSM120": MiniMaxH3SolAttentionSM120,
    "MiniMaxH3BSAAllExactScheduled": MiniMaxH3BSAAllExactScheduled,
    "SaveMiniMaxH3AVLatentCache": SaveMiniMaxH3AVLatentCache,
    "LoadMiniMaxH3AVLatentCache": LoadMiniMaxH3AVLatentCache,
    "SaveVideoLosslessUltrafast": SaveVideoLosslessUltrafast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SparkAttentionSM120": "MiniMax H3 Spark Attention (SM120)",
    "MiniMaxH3SolAttentionSM120": "MiniMax H3 Sol Attention (Official Compatibility)",
    "MiniMaxH3BSAAllExactScheduled": "MiniMax H3 BSA All-Exact (Scheduled Dense Control)",
    "SaveMiniMaxH3AVLatentCache": "Save MiniMax H3 AV Latent Cache",
    "LoadMiniMaxH3AVLatentCache": "Load MiniMax H3 AV Latent Cache",
    "SaveVideoLosslessUltrafast": "Save Video Lossless (Ultrafast)",
}
