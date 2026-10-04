#!/usr/bin/env python3
"""Measure deterministic MiniMax-H3 VAE reconstruction quality on the Ref2VA video."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
sys.path.insert(0, str(BENCH / "scripts"))
import ref2va_official_case as official  # noqa: E402

NAME = "ref2va_vae_reconstruction_metrics_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SOURCE = official.VIDEO_REFERENCE
CASES = {
    "768p_24fps": (768, 1344, 24),
    "480p_24fps": (480, 832, 24),
    "768p_12fps": (768, 1344, 12),
    "768p_8fps": (768, 1344, 8),
}


def write(path: Path, value) -> None:
    official.write(path, value)


def vae_valid_frame_count(n: int) -> int:
    """Largest 17*n+5 frame count not exceeding n (same rule as Ref2VA encoder)."""
    return max(5, max(1, (n - 5) // 17) * 17 + 5)


def normalized_frames(short_edge: int, max_width: int, fps: int) -> np.ndarray:
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep

    ref = MiniMaxH3VideoReference.from_file(SOURCE)
    frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
        ref.frames,
        ref.fps,
        10_000,
        32,
        short_edge,
        short_edge * max_width,
        fps,
    )
    frames = frames[: vae_valid_frame_count(len(frames))]
    assert frames.dtype == np.uint8 and frames.shape[1:3] == (short_edge, max_width), frames.shape
    assert (len(frames) - 5) % 17 == 0, len(frames)
    return np.ascontiguousarray(frames)


def metric_summary(original: np.ndarray, reconstructed: np.ndarray, device: str) -> dict:
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    assert original.shape == reconstructed.shape
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    psnr, ssim, perceptual = [], [], []
    with torch.inference_mode():
        for index, (a8, b8) in enumerate(zip(original, reconstructed, strict=True)):
            a = a8.astype(np.float32) / 255.0
            b = b8.astype(np.float32) / 255.0
            mse = float(np.mean((a - b) ** 2, dtype=np.float64))
            psnr.append(float("inf") if mse == 0 else float(-10.0 * np.log10(mse)))
            ssim.append(float(structural_similarity(a, b, channel_axis=-1, data_range=1.0)))
            ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
            tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device)
            perceptual.append(float(loss_fn(ta.mul(2).sub(1), tb.mul(2).sub(1)).item()))
            if (index + 1) % 10 == 0:
                print(f"metrics {index + 1}/{len(original)}", flush=True)
    del loss_fn
    torch.cuda.empty_cache()
    return {
        "psnr_db": {"mean": float(np.mean(psnr)), "std": float(np.std(psnr)), "per_frame": psnr},
        "ssim": {"mean": float(np.mean(ssim)), "std": float(np.std(ssim)), "per_frame": ssim},
        "lpips_alex": {
            "mean": float(np.mean(perceptual)), "std": float(np.std(perceptual)), "per_frame": perceptual
        },
    }


def main() -> None:
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.utils.export_utils import encode_video

    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running",
        "source": str(SOURCE),
        "source_sha256": official.sha(SOURCE),
        "cases": CASES,
        "posterior": "mode (deterministic; no posterior sampling)",
        "pixel_space": "ImageNet-normalized RGB [0,1], reconstruction clamped to [0,1]",
        "vae_temporal_alignment": "largest 17*n+5 count not exceeding resampled source",
        "metrics": "framewise RGB PSNR/SSIM and LPIPS-Alex, then arithmetic mean",
        "runner_sha256": official.sha(Path(__file__).resolve()),
    }
    write(ROOT / "protocol.json", protocol)

    lock = official.acquire_weight_load_lock()
    try:
        vae = AutoencoderKLMiniMaxH3.from_pretrained(str(official.MODEL / "vae"), torch_dtype=torch.float32)
    finally:
        official.release_weight_load_lock(lock)
    vae.to("cuda").eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1, 1)
    records = {}

    for name, (height, width, fps) in CASES.items():
        case_dir = OUT / name
        case_dir.mkdir(parents=True, exist_ok=True)
        meta_path = case_dir / "metrics.json"
        if meta_path.is_file():
            records[name] = json.loads(meta_path.read_text())
            print(f"cache hit {name}", flush=True)
            continue
        frames = normalized_frames(height, width, fps)
        print(f"roundtrip {name}: {frames.shape}", flush=True)
        started = time.perf_counter()
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to("cuda", torch.float32)
        pixels = (pixels.div_(255.0) - mean) / std
        with torch.inference_mode():
            posterior = vae.encode(pixels, return_dict=False)[0]
            latents = posterior.mode()
            del posterior, pixels
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                decoded = vae.decode(latents, return_dict=False)[0]
            decoded = (decoded.float() * std + mean).clamp_(0, 1)
        torch.cuda.synchronize()
        roundtrip_seconds = time.perf_counter() - started
        reconstructed = (
            decoded[0].permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        latent_shape = list(latents.shape)
        del decoded, latents
        torch.cuda.empty_cache()

        metrics = metric_summary(frames, reconstructed, "cuda")
        video_path = case_dir / "reconstruction.mp4"
        encode_video(reconstructed.astype(np.float32) / 255.0, fps=fps, output_path=str(video_path))
        record = {
            "case": name,
            "resolution": [width, height],
            "fps": fps,
            "frames": len(frames),
            "duration_seconds": len(frames) / fps,
            "latent_shape": latent_shape,
            "roundtrip_seconds": roundtrip_seconds,
            "metrics": metrics,
            "reconstruction_video": str(video_path),
            "reconstruction_sha256": official.sha(video_path),
        }
        write(meta_path, record)
        records[name] = record
        print(
            f"complete {name}: PSNR={metrics['psnr_db']['mean']:.4f} "
            f"SSIM={metrics['ssim']['mean']:.6f} LPIPS={metrics['lpips_alex']['mean']:.6f}",
            flush=True,
        )

    del vae
    torch.cuda.empty_cache()
    result = {"status": "complete", "records": records}
    write(ROOT / "results.json", result)
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
