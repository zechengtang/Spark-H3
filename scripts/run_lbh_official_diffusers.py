#!/usr/bin/env python3
"""Run LBH's published two-pass MiniMax-H3 schedule through Diffusers."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils.export_utils import encode_video

from h3_lbh import (
    H3LBHOfficialConfig,
    LBHMiniMaxH3LatentUpscaler,
    apply_comfy_h3_lora,
    megapixel_canvas,
    patch_lbh_official_into_pipeline,
)
from h3_sparse_attention import H3AccelerationConfig, install_h3_acceleration


DEFAULT_MODEL = Path("/autodl-fs/data/models/MiniMax-H3")
DEFAULT_LORA = Path(
    "/autodl-fs/data/models/Kijai-MiniMax-H3_comfy/loras/"
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"
)
DEFAULT_LBH = Path(
    "/autodl-fs/data/models/Minimax_h3_latent_Upscaler/"
    "minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
)
BLOG_CASES = REPO_ROOT / "docs/blogs/spark-attn/integration/turbo-lora.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("0685", "0753"))
    parser.add_argument("--prompt")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--lbh-model", type=Path, default=DEFAULT_LBH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--height", type=int,
        help="explicit target height; paired with --width and overrides --target-megapixels",
    )
    parser.add_argument(
        "--width", type=int,
        help="explicit target width; paired with --height and overrides --target-megapixels",
    )
    parser.add_argument(
        "--target-megapixels", type=float, default=0.98,
        help="target MP for Comfy ResolutionSelector mode (default: 0.98, producing 1344x768)",
    )
    parser.add_argument("--low-height", type=int, help="explicit pass-1 height; must be paired with --low-width")
    parser.add_argument("--low-width", type=int, help="explicit pass-1 width; must be paired with --low-height")
    parser.add_argument("--low-megapixels", type=float, default=0.2)
    parser.add_argument("--aspect-ratio", type=float, default=16 / 9, help="width / height for MP modes")
    parser.add_argument("--align", type=int, default=32, help="pixel grid used by official ResolutionSelector")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lowres-scale", type=float,
        help="legacy proportional pass-1 size; overrides --low-megapixels",
    )
    parser.add_argument(
        "--group-size", type=int, default=0,
        help="high-resolution block-group offload size; 0 disables it (default)",
    )
    spark = parser.add_mutually_exclusive_group()
    spark.add_argument("--use-spark", dest="use_spark", action="store_true")
    spark.add_argument("--no-spark", dest="use_spark", action="store_false")
    parser.set_defaults(use_spark=False)
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true")
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=True)
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument("--runtime-warmup", dest="runtime_warmup", action="store_true")
    warmup_group.add_argument("--no-runtime-warmup", dest="runtime_warmup", action="store_false")
    parser.set_defaults(runtime_warmup=True)
    parser.add_argument("--spark-topk-ratio", type=float, default=0.1)
    parser.add_argument(
        "--spark-dense-layers", type=int, default=0,
        help="Number of leading transformer layers kept dense in Spark mode (default: 0).",
    )
    parser.add_argument(
        "--spark-layout-reuse",
        choices=("independent", "q_from_k", "k_from_q"),
        default="q_from_k",
        help="Spark reblock layout reuse; q_from_k is q_reuse_k",
    )
    parser.add_argument("--denoise-only", action="store_true")
    parser.add_argument(
        "--low-only",
        action="store_true",
        help="decode the pass-1 denoised_output before LBH; diagnostic only",
    )
    parser.add_argument(
        "--lift-only",
        action="store_true",
        help="decode the learned LBH output before re-noising; diagnostic only",
    )
    args = parser.parse_args()
    if args.low_only and args.lift_only:
        parser.error("--low-only and --lift-only are mutually exclusive")
    if bool(args.case) == bool(args.prompt):
        parser.error("pass exactly one of --case or --prompt")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be supplied together")
    if (args.low_height is None) != (args.low_width is None):
        parser.error("--low-height and --low-width must be supplied together")
    return args


def resolve_canvases(args) -> tuple[tuple[int, int], tuple[int, int]]:
    if args.height is not None:
        high_h, high_w = args.height, args.width
    else:
        high_h, high_w = megapixel_canvas(args.target_megapixels, args.aspect_ratio, args.align)
    if args.low_height is not None:
        low_h, low_w = args.low_height, args.low_width
    elif args.lowres_scale is not None:
        low_h = max(args.align, round(high_h * args.lowres_scale / args.align) * args.align)
        low_w = max(args.align, round(high_w * args.lowres_scale / args.align) * args.align)
    else:
        low_h, low_w = megapixel_canvas(args.low_megapixels, high_w / high_h, args.align)
    if any(value % 16 for value in (low_h, low_w, high_h, high_w)):
        raise ValueError("all H3 canvas dimensions must be divisible by the VAE factor 16")
    if low_h >= high_h or low_w >= high_w:
        raise ValueError("pass-1 dimensions must be smaller than target dimensions on both axes")
    return (low_h, low_w), (high_h, high_w)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prompt(args) -> tuple[str, str]:
    if args.prompt:
        return "custom", args.prompt
    cases = json.loads(BLOG_CASES.read_text())["cases"]
    return f"vbench_all_{args.case}", cases[args.case]["generation_prompt"]


def get_workflow(model: Path):
    template = ModularPipeline.from_pretrained(str(model), local_files_only=True)
    return template.blocks.get_workflow("t2va")


def release_cuda_arenas() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_state(state):
    from diffusers.modular_pipelines import PipelineState

    values = {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in state.values.items()
    }
    return PipelineState(values=values, kwargs_mapping=copy.deepcopy(state.kwargs_mapping))


def prepare_conditioning(model: Path, prompt: str, device: torch.device):
    """Run Qwen conditioning as a separate resident stage, without offload hooks."""
    text_block = get_workflow(model).sub_blocks["text_encoder"]
    manager = ComponentsManager()
    conditioner = text_block.init_pipeline(str(model), components_manager=manager)
    conditioner.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path={"default": str(model)},
        local_files_only=True,
    )
    conditioner.text_encoder.to(device)
    with torch.inference_mode():
        state = conditioner(prompt=prompt)
    for key, value in list(state.values.items()):
        if isinstance(value, torch.Tensor):
            state.values[key] = value.detach().cpu()
    conditioner.text_encoder.to("cpu")
    del conditioner, manager
    release_cuda_arenas()
    return state


def make_denoiser(model: Path):
    workflow = get_workflow(model)
    workflow.sub_blocks.pop("text_encoder")
    workflow.sub_blocks.pop("decode.video")
    workflow.sub_blocks.pop("decode.audio")
    manager = ComponentsManager()
    pipe = workflow.init_pipeline(str(model), components_manager=manager)
    pipe.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path={"default": str(model)},
        local_files_only=True,
    )
    return pipe, manager


def decode_latents(model: Path, latents, audio_latents, device: torch.device):
    workflow = get_workflow(model)
    for name in list(workflow.sub_blocks):
        if not name.startswith("decode."):
            workflow.sub_blocks.pop(name)
    manager = ComponentsManager()
    decoder = workflow.init_pipeline(str(model), components_manager=manager)
    decoder.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path={"default": str(model)},
        local_files_only=True,
    )
    decoder.vae.to(device)
    decoder.audio_vae.to(device)
    with torch.inference_mode():
        result = decoder(
            latents=latents.to(device),
            audio_latents=audio_latents.to(device),
            output_type="np",
            output=["videos", "audio", "sampling_rate"],
        )
    decoder.vae.to("cpu")
    decoder.audio_vae.to("cpu")
    del decoder, manager
    release_cuda_arenas()
    return result


def main():
    args = parse_args()
    (low_height, low_width), (height, width) = resolve_canvases(args)
    sample_id, prompt = resolve_prompt(args)
    for path in (args.model, args.lora, args.lbh_model):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    conditioning_started = time.perf_counter()
    state = prepare_conditioning(args.model, prompt, device)
    conditioning_seconds = time.perf_counter() - conditioning_started

    pipe, manager = make_denoiser(args.model)
    if (float(pipe.scheduler.shift), float(pipe.audio_scheduler.shift)) != (12.0, 3.0):
        raise RuntimeError(
            "the published LBH LightX2V-v0.1 recipe requires video/audio scheduler shifts 12/3"
        )
    wrapped = apply_comfy_h3_lora(pipe.transformer, args.lora)
    if wrapped != 312:
        raise RuntimeError(f"expected 312 LoRA-wrapped projections, got {wrapped}")

    lifter = LBHMiniMaxH3LatentUpscaler.from_pretrained(
        args.lbh_model, variant="fp16", torch_dtype=torch.float16, device="cpu"
    )
    spark_overrides = {
        "warmup_percent": 0.0,
        "sol_route_topk_ratio": args.spark_topk_ratio,
        "sol_dense_layers": args.spark_dense_layers,
        "landmark_tree_v2_layout_reuse": args.spark_layout_reuse,
    }
    config = H3LBHOfficialConfig(
        lowres_scale=(low_height / height, low_width / width),
        highres_group_offload_blocks=args.group_size or None,
        low_only=args.low_only,
        lift_only=args.lift_only,
        use_spark=args.use_spark,
        spark_overrides=spark_overrides,
    )
    patch_lbh_official_into_pipeline(pipe, lifter, config)
    pipe.transformer.to(device)

    acceleration = install_h3_acceleration(
        pipe,
        H3AccelerationConfig(torch_compile=args.compile, vae_fp16=False),
    )
    acceleration.__enter__()
    compile_wrapped_blocks = len(acceleration._forward_originals)
    if args.compile and compile_wrapped_blocks != 50:
        acceleration.remove()
        raise RuntimeError(
            f"blog compile mode expected 50 wrapped transformer blocks, got {compile_wrapped_blocks}"
        )

    warmup_seconds = None
    if args.runtime_warmup:
        # Build both shape-specialized block graphs with the cheapest complete
        # LBH trajectory: one low-resolution dense evaluation, lift, then one
        # high-resolution Spark evaluation. Attention itself is an eager
        # boundary, so the high-resolution block graph is reusable by both the
        # measured Dense and Spark variants.
        denoise_block = pipe._blocks.sub_blocks["denoise.denoise"]
        warmup_config = H3LBHOfficialConfig(
            lowres_scale=config.lowres_scale,
            low_grid_points=config.low_grid_points,
            low_evaluations=1,
            high_sigmas=(config.high_sigmas[0], 0.0),
            highres_group_offload_blocks=None,
            use_spark=True,
            spark_overrides=spark_overrides,
        )
        warmup_state = clone_state(state)
        warmup_state.values["prompt_embeds"] = warmup_state.values["prompt_embeds"].to(device)
        denoise_block.lbh_config = warmup_config
        warmup_started = time.perf_counter()
        warmup_result = None
        try:
            with torch.inference_mode():
                warmup_result = pipe(
                    state=warmup_state,
                    num_frames=args.frames,
                    height=height,
                    width=width,
                    num_inference_steps=8,
                    generator=torch.Generator(device="cpu").manual_seed(args.seed),
                    output=["latents", "audio_latents"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmup_seconds = time.perf_counter() - warmup_started
        finally:
            denoise_block.lbh_config = config
        del warmup_result, warmup_state
        release_cuda_arenas()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(
                state=state,
                num_frames=args.frames,
                height=height,
                width=width,
                # Comfy BasicScheduler(simple, steps=8) has eight evaluations and a
                # terminal sigma; the custom loop consumes its first four evaluations.
                num_inference_steps=8,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output=["latents", "audio_latents"],
            )
    finally:
        acceleration.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    denoise_seconds = time.perf_counter() - started
    peak_denoise_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    )
    latents = result["latents"].detach().cpu()
    audio_latents = result["audio_latents"].detach().cpu()
    pipe.transformer.to("cpu")
    del result, pipe, manager, state
    release_cuda_arenas()

    latent_path = args.output.with_suffix(".latents.pt")
    torch.save(
        {
            "latents": latents,
            "audio_latents": audio_latents,
            "sample_id": sample_id,
            "prompt": prompt,
            "seed": args.seed,
        },
        latent_path,
    )
    if not args.denoise_only:
        decoded = decode_latents(args.model, latents, audio_latents, device)
        frames = decoded["videos"][0][: args.frames]
        audio_samples = round(args.frames / 24 * decoded["sampling_rate"])
        audio = decoded["audio"][0][..., :audio_samples]
        encode_video(
            frames,
            fps=24,
            output_path=str(args.output),
            audio=audio,
            audio_sample_rate=decoded["sampling_rate"],
        )

    record = {
        "sample_id": sample_id,
        "prompt": prompt,
        "seed": args.seed,
        "requested_frames": args.frames,
        "height": height,
        "width": width,
        "target_megapixels_actual": height * width / 1024**2,
        "low_height": low_height,
        "low_width": low_width,
        "low_megapixels_actual": low_height * low_width / 1024**2,
        "spatial_lift": [height / low_height, width / low_width],
        "low_nfe": config.low_evaluations,
        "high_sigmas": list(config.high_sigmas),
        "high_nfe": len(config.high_sigmas) - 1,
        "scheduler_shifts": {"video": 12.0, "audio": 3.0},
        "highres_group_offload_blocks": config.highres_group_offload_blocks,
        "low_only": args.low_only,
        "lift_only": args.lift_only,
        "lora": str(args.lora.resolve()),
        "lora_sha256": sha256(args.lora),
        "lbh": str(args.lbh_model.resolve()),
        "use_spark": args.use_spark,
        "spark_scope": "high_resolution_only" if args.use_spark else None,
        "spark_overrides": spark_overrides if args.use_spark else None,
        "torch_compile": args.compile,
        "compile_scope": "per_transformer_block_attention_eager" if args.compile else None,
        "compile_wrapped_blocks": compile_wrapped_blocks,
        "runtime_warmup": (
            {
                "discarded": True,
                "low_evaluations": 1,
                "low_attention": "dense",
                "high_evaluations": 1,
                "high_attention": "spark",
                "seconds": warmup_seconds,
            }
            if args.runtime_warmup else None
        ),
        "component_auto_cpu_offload": False,
        "transformer_placement": "resident",
        "conditioning_seconds": conditioning_seconds,
        "denoise_seconds": denoise_seconds,
        "elapsed_seconds": conditioning_seconds + denoise_seconds,
        "peak_cuda_allocated_gib": peak_denoise_gib,
        "latent_path": str(latent_path.resolve()),
        "output": None if args.denoise_only else str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
