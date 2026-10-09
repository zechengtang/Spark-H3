#!/usr/bin/env python3
"""Convert the LarryVRH MiniMax-H3 Turbo LoRA to ComfyUI generic format.

The LarryVRH adapter contains ordinary DiT LoRA weights plus 51 AdaLN LoRA
weights trained against the original 2688-wide time embedding.  ComfyUI's
pruned H3 checkpoint stores that time curve in an 8-dimensional basis.  This
tool projects each AdaLN update into that existing basis and emits one generic
LoRA file which can be loaded by the stock ``LoraLoaderModelOnly`` node.

The small least-squares projection is bundled with Spark-H3, so conversion does
not load the large diffusion checkpoint and does not require the LarryVRH node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LarryVRH LoRA")
    parser.add_argument("--projection", type=Path, default=None,
                        help="bundled AdaLN curve projection (normally auto-detected)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = (
        Path(__file__).with_name("adaln_curve_projection.safetensors"),
        Path(__file__).parent / "assets" / "adaln_curve_projection.safetensors",
        Path(__file__).resolve().parents[1] / "comfyui/assets/adaln_curve_projection.safetensors",
    )
    projection_path = args.projection or next((path for path in candidates if path.is_file()), None)
    if projection_path is None:
        raise FileNotFoundError("adaln_curve_projection.safetensors was not found")
    for path in (args.input, projection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source = load_file(str(args.input), device="cpu")
    projection = load_file(str(projection_path), device="cpu")["projection"].float()
    if projection.shape != (9, 2688):
        raise ValueError(f"unexpected projection shape: {tuple(projection.shape)}")

    converted: dict[str, torch.Tensor] = {}
    adaln_count = 0
    module_names = sorted(
        key.removesuffix(".lora_A.weight")
        for key in source
        if key.endswith(".lora_A.weight")
    )
    for module in module_names:
        a_key = module + ".lora_A.weight"
        b_key = module + ".lora_B.weight"
        a = source[a_key]
        b = source[b_key]
        target = "diffusion_model." + module

        if "adaln_proj.linear" not in module:
            converted[target + ".lora_A.weight"] = a.contiguous()
            converted[target + ".lora_B.weight"] = b.contiguous()
            alpha_key = module + ".alpha"
            if alpha_key in source:
                converted[target + ".alpha"] = source[alpha_key]
            continue

        # projection @ A.T is [9, rank].  Multiplying B by its first eight
        # rows gives the compressed weight delta; the ninth gives bias delta.
        coeff = projection @ a.float().T
        weight_delta = b.float() @ coeff[:8].T
        bias_delta = b.float() @ coeff[8]
        converted[target + ".diff"] = weight_delta.to(torch.bfloat16).contiguous()
        converted[target + ".diff_b"] = bias_delta.to(torch.bfloat16).contiguous()

        adaln_count += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "base_model": "Comfy-Org/MiniMax-H3 pruned curve checkpoint",
        "source": args.input.name,
        "conversion": "ComfyUI generic LoRA; AdaLN projected 2688D->8D curve+bias",
        "adaln_projection_relative_error_mean": "0.0010447574",
        "adaln_projection_relative_error_max": "0.0022290535",
    }
    save_file(converted, str(args.output), metadata=metadata)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    print(f"converted {adaln_count} AdaLN modules; reference projection error: "
          "mean=0.104476%, max=0.222905%")


if __name__ == "__main__":
    main()
