"""ComfyUI model patch for the native MiniMax-H3 implementation.

The native ComfyUI model packs tokens as ``[conditioning | target audio |
target video]`` while Spark-H3 expects the target video first.  This adapter
uses ComfyUI's published ``minimax_h3_layout`` to move only the target-video
rows to the front, runs Spark, and restores the original order afterwards.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from h3_sparse_attention.processor import (
    H3SparseAttentionConfig,
    PackedLayout as SparkPackedLayout,
    _Controller,
    _sol_attention,
)
from h3_sparse_attention.spark_integration import spark_attention_bthd


log = logging.getLogger(__name__)
_SM120 = (12, 0)
_PRODUCER_CHUNK = 4096


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


def _make_spark_layout(layout, sequence_length: int, device: torch.device) -> SparkPackedLayout:
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
    return SparkPackedLayout(
        permutation=permutation,
        inverse_permutation=inverse,
        grid=grid,
        video_tokens=stop - start,
        sequence_length=sequence_length,
        video_positions=video_positions,
    )


@dataclass
class _RunState:
    steps: int
    warmup_percent: float
    controller: _Controller
    previous_sigma: float | None = None
    layout_cache: dict[tuple[int, int, str], SparkPackedLayout] = field(default_factory=dict)

    @classmethod
    def create(cls, steps: int, warmup_percent: float, topk_ratio: float):
        # Spark's public factory models the Diffusers pipeline's N-1 evaluations.
        # ComfyUI executes one model evaluation per sampler step, hence steps + 1.
        config = H3SparseAttentionConfig.spark(
            steps + 1,
            warmup_percent=0.0,
            sol_dense_layers=0,
            sol_route_topk_ratio=topk_ratio,
        )
        return cls(steps, warmup_percent, _Controller(config))

    @classmethod
    def create_sol(cls, steps: int, warmup_percent: float, tau: float):
        config = H3SparseAttentionConfig.sol(
            steps + 1,
            warmup_percent=0.0,
            sol_tau=tau,
            sol_dense_layers=0,
        )
        return cls(steps, warmup_percent, _Controller(config))

    @property
    def warmup_evaluations(self) -> int:
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


def _native_spark_qkv(attn, x, rope_freqs, permutation):
    """ComfyUI-style chunked H3 producer yielding contiguous BTHD Q/K/V."""
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.quant_ops

    tokens = x.shape[0]
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    inner = heads * head_dim
    q = torch.empty((1, tokens, heads, head_dim), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    rot_dim = rope_freqs.shape[-3] * 2

    with comfy.model_prefetch.pause_malloc_graph():
        for start in range(0, tokens, _PRODUCER_CHUNK):
            stop = min(start + _PRODUCER_CHUNK, tokens)
            indices = permutation[start:stop]
            projected = attn.qkv_proj(x.index_select(0, indices))
            qc, kc, vc = projected.split(inner, dim=-1)
            qc = qc.view(1, stop - start, heads, head_dim)
            kc = kc.view(1, stop - start, heads, head_dim)
            comfy.quant_ops.ck.rms_rope_split_half_(
                qc,
                kc,
                rope_freqs.index_select(1, indices),
                qw,
                kw,
                epsilon=attn.q_norm.eps,
                rot_dim=rot_dim,
            )
            q[:, start:stop].copy_(qc)
            k[:, start:stop].copy_(kc)
            v[:, start:stop].copy_(vc.view(1, stop - start, heads, head_dim))
    return q, k, v


def _native_spark_attention(attn, x, rope_freqs, transformer_options, layer, state, activation_log):
    layout = state.spark_layout(
        transformer_options["minimax_h3_layout"], x.shape[0], x.device
    )
    q, k, v = _native_spark_qkv(attn, x, rope_freqs, layout.permutation)
    output = spark_attention_bthd(
        state.controller, q, k, v, layout, layer, return_bthd=True
    )
    output = output.index_select(1, layout.inverse_permutation)
    state.controller.counts["sparse:spark_comfy_native"] += 1
    activation_log.hit(x.shape[0], layout.video_tokens)
    return attn.out_proj(output.reshape(x.shape[0], int(attn.heads) * int(attn.head_dim)))


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


def _make_attention_forward(
    attn,
    fallback_forward,
    layer: int,
    dense_layers: int,
    min_tokens: int,
    strict: bool,
    state: _RunState,
    activation_log: _ActivationLog,
):
    heads, head_dim = int(attn.heads), int(attn.head_dim)
    inner = heads * head_dim

    def forward(x, rope_freqs=None, transformer_options={}):
        # KJNodes may hand ownership over through a one-element list.  Do not
        # consume it until every condition that can choose dense fallback passes.
        handoff = isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0])
        tensor = x[0] if handoff else x
        handoff_released = False
        try:
            if layer == 0:
                state.begin_evaluation(transformer_options)
            if not torch.is_tensor(tensor) or tensor.ndim != 2:
                raise _Incompatible("attention input is not a rank-2 tensor")
            sequence_length = tensor.shape[0]
            if state.is_warmup:
                raise _Unsupported(
                    f"dense warmup evaluation {state.controller.evaluation_index + 1}/"
                    f"{state.warmup_evaluations}"
                )
            if layer < dense_layers:
                raise _Unsupported(f"transformer layer {layer} is configured dense")
            if sequence_length < min_tokens or tensor.requires_grad:
                raise _Unsupported("below min_tokens or autograd requested")
            if tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
                raise _Incompatible("Spark SM120 requires CUDA bfloat16 activations")
            if head_dim != 128:
                raise _Incompatible(f"head_dim {head_dim} != 128")
            capability = torch.cuda.get_device_capability(tensor.device)
            if capability != _SM120:
                raise _Incompatible(
                    f"this node targets SM120, found SM{capability[0]}{capability[1]}"
                )
            comfy_layout = (transformer_options or {}).get("minimax_h3_layout")
            if comfy_layout is None:
                raise _Incompatible(
                    "ComfyUI did not publish minimax_h3_layout; update ComfyUI to 0.30.0+"
                )
            spark_layout = state.spark_layout(comfy_layout, sequence_length, tensor.device)

            if handoff:
                tensor = x.pop()
            device = tensor.device
            q, k, v = attn.qkv_proj(tensor).split(inner, dim=-1)
            if handoff:
                del tensor
                handoff_released = True
            q = q.view(1, sequence_length, heads, head_dim)
            k = k.view(1, sequence_length, heads, head_dim)
            v = v.view(1, sequence_length, heads, head_dim)

            if rope_freqs is not None:
                import comfy.model_management
                import comfy.quant_ops

                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=device)
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q,
                    k,
                    rope_freqs,
                    qw,
                    kw,
                    epsilon=attn.q_norm.eps,
                    rot_dim=rope_freqs.shape[-3] * 2,
                )
            else:
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            permutation = spark_layout.permutation
            q, k, v = (
                value.index_select(1, permutation).permute(0, 2, 1, 3)
                for value in (q, k, v)
            )
            output = _sol_attention(
                state.controller,
                q,
                k,
                v,
                spark_layout,
                layer,
                return_bthd=True,
            )
            output = output.index_select(1, spark_layout.inverse_permutation)
            activation_log.hit(sequence_length, spark_layout.video_tokens)
            return attn.out_proj(output.reshape(sequence_length, inner))
        except _Incompatible as exc:
            if strict:
                raise
            activation_log.miss(str(exc))
        except _Unsupported as exc:
            activation_log.miss(str(exc))
        except Exception as exc:
            if strict or handoff_released:
                raise
            activation_log.miss(f"{type(exc).__name__}: {exc}")

        if handoff and not x:
            # The only path that can empty the handoff is followed by either a
            # successful return or a re-raised kernel error above.
            raise RuntimeError("Spark-H3 consumed a low-VRAM activation before fallback")
        return fallback_forward(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    forward._spark_h3_fallback = fallback_forward
    return forward


class MiniMaxH3SparkAttentionSM120:
    """Run Spark through ComfyUI's native MiniMax-H3 block-patch architecture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 200, "step": 1},
                ),
                "warmup_percent": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0},
                ),
                "topk_ratio": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "dense_layers": (
                    "INT",
                    {"default": 1, "min": 0, "max": 50, "step": 1},
                ),
                "min_tokens": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 262144, "step": 256},
                ),
                "strict": ("BOOLEAN", {"default": True}),
            }
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
        state = _RunState.create(int(steps), float(warmup_percent), float(topk_ratio))
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
            "(Top-K %.0f%%, warmup %.0f%%, dense blocks %s)",
            len(blocks),
            100.0 * float(topk_ratio),
            float(warmup_percent),
            "none" if int(dense_layers) == 0 else f"0..{int(dense_layers) - 1}",
        )
        return (patched,)


class MiniMaxH3SolAttentionSM120(MiniMaxH3SparkAttentionSM120):
    """Patch ComfyUI's native MiniMax-H3 model with Spark-H3's base Sol kernel."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        required = inputs["required"]
        required.pop("topk_ratio")
        required["tau"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        )
        return inputs

    DESCRIPTION = (
        "Patch ComfyUI's native MiniMax H3 DiT with the base Sol-Attn kernel "
        "bundled by Spark-H3. Spark-Reblock, Spark-Reweight, and Top-K routing "
        "are disabled. Set steps to the sampler's model-evaluation count."
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

        patched = model.clone()
        state = _RunState.create_sol(int(steps), float(warmup_percent), float(tau))
        activation_log = _ActivationLog()
        for layer in range(len(blocks)):
            path = f"diffusion_model.blocks.{layer}.attn.forward"
            attn = patched.get_model_object(f"diffusion_model.blocks.{layer}.attn")
            prior = getattr(patched, "object_patches", {}).get(path)
            fallback = prior if prior is not None else attn.forward
            if hasattr(fallback, "_spark_h3_fallback"):
                fallback = fallback._spark_h3_fallback
            patched.add_object_patch(
                path,
                _make_attention_forward(
                    attn,
                    fallback,
                    layer,
                    int(dense_layers),
                    int(min_tokens),
                    bool(strict),
                    state,
                    activation_log,
                ),
            )
        log.info(
            "[Spark-H3 Sol] patched %d MiniMax H3 blocks (tau %.2f, warmup %.0f%%, dense layers %d)",
            len(blocks),
            float(tau),
            float(warmup_percent),
            int(dense_layers),
        )
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
    "SaveMiniMaxH3AVLatentCache": SaveMiniMaxH3AVLatentCache,
    "LoadMiniMaxH3AVLatentCache": LoadMiniMaxH3AVLatentCache,
    "SaveVideoLosslessUltrafast": SaveVideoLosslessUltrafast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SparkAttentionSM120": "MiniMax H3 Spark Attention (SM120)",
    "MiniMaxH3SolAttentionSM120": "MiniMax H3 Sol Attention (Spark-H3 SM120)",
    "SaveMiniMaxH3AVLatentCache": "Save MiniMax H3 AV Latent Cache",
    "LoadMiniMaxH3AVLatentCache": "Load MiniMax H3 AV Latent Cache",
    "SaveVideoLosslessUltrafast": "Save Video Lossless (Ultrafast)",
}
