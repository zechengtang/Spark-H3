"""Load the fused ComfyUI MiniMax-H3 LoRA layout into Diffusers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LoRALinear(nn.Module):
    """Frozen linear layer plus an inference-only, strength-one LoRA."""

    def __init__(self, base: nn.Linear, down: torch.Tensor, up: torch.Tensor):
        super().__init__()
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError("LoRA tensors must be matrices")
        if down.shape[1] != base.in_features or up.shape != (base.out_features, down.shape[0]):
            raise ValueError(
                f"LoRA {tuple(down.shape)}/{tuple(up.shape)} is incompatible with "
                f"Linear({base.in_features}, {base.out_features})"
            )
        self.base = base.requires_grad_(False)
        self.lora_down = nn.Parameter(down.contiguous(), requires_grad=False)
        self.lora_up = nn.Parameter(up.contiguous(), requires_grad=False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, hidden_states):
        output = self.base(hidden_states)
        down = self.lora_down.to(device=hidden_states.device, dtype=hidden_states.dtype)
        up = self.lora_up.to(device=hidden_states.device, dtype=hidden_states.dtype)
        update = F.linear(F.linear(hidden_states, down), up)
        return output + update.to(output.dtype)


def _replace_linear(parent, name: str, down: torch.Tensor, up: torch.Tensor) -> None:
    module = parent
    parts = name.split(".")
    for part in parts[:-1]:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    attribute = parts[-1]
    base = module[int(attribute)] if attribute.isdigit() else getattr(module, attribute)
    wrapped = _LoRALinear(base, down, up)
    if attribute.isdigit():
        module[int(attribute)] = wrapped
    else:
        setattr(module, attribute, wrapped)


def _pop_pair(state: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return state.pop(prefix + ".lora_A.weight"), state.pop(prefix + ".lora_B.weight")
    except KeyError as error:
        raise KeyError(f"missing ComfyUI LoRA pair for {prefix}") from error


def _comfy_fc1_up_to_diffusers(up: torch.Tensor) -> torch.Tensor:
    """Undo MiniMax-H3 ComfyUI's SwiGLU output-half permutation.

    Diffusers stores the fused MLP-up output as ``[value; gate]`` while the
    Comfy H3 implementation stores ``[gate; value]``.  The published LoRA's
    metadata confirms that its ``mlp.fc1`` B matrices were permuted during
    conversion, so loading them back into Diffusers must reverse that step.
    """
    if up.ndim != 2 or up.shape[0] % 2:
        raise ValueError(f"invalid fused SwiGLU LoRA-up shape: {tuple(up.shape)}")
    gate, value = up.chunk(2, dim=0)
    return torch.cat((value, gate), dim=0)


@torch.no_grad()
def apply_comfy_h3_lora(transformer: nn.Module, checkpoint: str | Path) -> int:
    """Apply a fused-qkv ComfyUI H3 LoRA without PEFT or a Comfy runtime.

    The checkpoint stores each Q/K/V adapter as one block-diagonal ``qkv_proj``
    pair. Diffusers exposes three projection modules, so the pair is split back
    into its three independent rank-R adapters before installation.

    Returns the number of wrapped linear modules.
    """
    from safetensors.torch import load_file

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    state = dict(load_file(str(path), device="cpu"))
    wrapped = 0

    groups = [
        (f"diffusion_model.blocks.{index}", transformer.transformer_blocks[index])
        for index in range(len(transformer.transformer_blocks))
    ] + [
        (f"diffusion_model.token_refiner.blocks.{index}", transformer.token_refiner.refiner_blocks[index])
        for index in range(len(transformer.token_refiner.refiner_blocks))
    ]

    for source, block in groups:
        qkv_down, qkv_up = _pop_pair(state, source + ".attn.qkv_proj")
        if qkv_down.shape[0] % 3 or qkv_up.shape[0] % 3 or qkv_up.shape[1] % 3:
            raise ValueError(f"invalid fused QKV LoRA shape at {source}")
        down_parts = qkv_down.chunk(3, dim=0)
        output_parts = qkv_up.chunk(3, dim=0)
        rank = qkv_down.shape[0] // 3
        for index, name in enumerate(("to_q", "to_k", "to_v")):
            column = output_parts[index][:, index * rank : (index + 1) * rank]
            off_diagonal = torch.cat(
                (output_parts[index][:, : index * rank], output_parts[index][:, (index + 1) * rank :]), dim=1
            )
            if off_diagonal.numel() and torch.count_nonzero(off_diagonal).item():
                raise ValueError(f"fused QKV LoRA at {source} is not block diagonal")
            _replace_linear(block, f"attn.{name}", down_parts[index], column)
            wrapped += 1

        mappings = (
            ("attn.out_proj", "attn.to_out.0"),
            ("mlp.fc1", "ff.net.0.proj"),
            ("mlp.fc2", "ff.net.2"),
        )
        for source_tail, target in mappings:
            down, up = _pop_pair(state, f"{source}.{source_tail}")
            if source_tail == "mlp.fc1":
                up = _comfy_fc1_up_to_diffusers(up)
            _replace_linear(block, target, down, up)
            wrapped += 1

    if state:
        preview = ", ".join(sorted(state)[:5])
        raise KeyError(f"unused tensors in H3 LoRA ({len(state)}), e.g. {preview}")
    transformer.eval().requires_grad_(False)
    return wrapped
