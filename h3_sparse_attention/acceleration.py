"""Reversible per-transformer-block compilation for MiniMax-H3 inference.

Compile the blocks rather than the whole transformer: the denoiser inspects
``transformer.forward`` to forward layout arguments, which a whole-module
``OptimizedModule`` would hide. Sparse attention remains an eager boundary so
its mutable routing state does not trigger repeated recompilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class H3AccelerationConfig:
    torch_compile: bool = True
    compile_mode: str | None = None
    compile_dynamic: bool | None = None
    compile_deterministic: bool = True
    vae_fp16: bool = False


_MISSING = object()


class H3AccelerationPlugin:
    def __init__(self, pipe: Any, config: H3AccelerationConfig):
        self.pipe = pipe
        self.config = config
        self._forward_originals: list[tuple[Any, Any]] = []
        self._attention_forward_originals: list[tuple[Any, Any]] = []
        self._vae_dtype: torch.dtype | None = None
        self._active = False

    def __enter__(self) -> "H3AccelerationPlugin":
        if self._active:
            raise RuntimeError("acceleration plugin is already active")
        try:
            if self.config.vae_fp16:
                self._install_vae_fp16()
            if self.config.torch_compile:
                self._install_compile()
            self._active = True
            return self
        except Exception:
            self.remove()
            raise

    def _install_vae_fp16(self) -> None:
        vae = getattr(self.pipe, "vae", None)
        if vae is None:
            raise TypeError("expected a MiniMax-H3 pipeline with a vae component")
        self._vae_dtype = next(vae.parameters()).dtype
        if self._vae_dtype != torch.float16:
            vae.to(torch.float16)

    def _install_compile(self) -> None:
        transformer = getattr(self.pipe, "transformer", None)
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise TypeError("expected a MiniMax-H3 pipeline with transformer blocks")

        # Keep Inductor's reduction choices deterministic across worker processes.
        options = dict(torch._inductor.list_mode_options(self.config.compile_mode or "default"))
        options["deterministic"] = self.config.compile_deterministic
        for block in blocks:
            attention = getattr(block, "attn", None)
            if attention is not None:
                self._attention_forward_originals.append(
                    (attention, attention.__dict__.get("forward", _MISSING))
                )
                attention.forward = torch.compiler.disable(attention.forward)
            self._forward_originals.append(
                (block, block.__dict__.get("forward", _MISSING))
            )
            block.forward = torch.compile(
                block.forward,
                options=options,
                dynamic=self.config.compile_dynamic,
            )

    def remove(self) -> None:
        for block, original in reversed(self._forward_originals):
            if original is _MISSING:
                del block.forward
            else:
                block.forward = original
        self._forward_originals.clear()
        for attention, original in reversed(self._attention_forward_originals):
            if original is _MISSING:
                del attention.forward
            else:
                attention.forward = original
        self._attention_forward_originals.clear()
        if self._vae_dtype is not None:
            self.pipe.vae.to(self._vae_dtype)
            self._vae_dtype = None
        self._active = False

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.remove()
        return False


def install_h3_acceleration(
    pipe: Any, config: H3AccelerationConfig | None = None
) -> H3AccelerationPlugin:
    return H3AccelerationPlugin(pipe, config or H3AccelerationConfig())
