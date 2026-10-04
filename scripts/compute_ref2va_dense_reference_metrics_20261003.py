#!/usr/bin/env python3
"""Frame-aligned metrics for low-FPS Ref2VA variants against the 24fps Dense output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np


GALLERY = Path("/autodl-fs/data/h3_outputs/ref2va_768_480_dense_spark_20261003")
VARIANTS = Path("/autodl-fs/data/h3_outputs/ref2va_condition_fps_rope_20261003")
EXPERIMENT = Path("/autodl-fs/data/h3_experiments/ref2va_condition_fps_rope_20261003")


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"could not decode frames from {path}")
    return np.stack(frames)


def compare(reference: np.ndarray, candidate: np.ndarray, batch_size: int = 4) -> dict:
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    if reference.shape != candidate.shape:
        raise ValueError(f"video shape mismatch: {reference.shape} vs {candidate.shape}")
    difference = reference.astype(np.float32) - candidate.astype(np.float32)
    frame_mse = np.mean(difference * difference, axis=(1, 2, 3), dtype=np.float64)
    psnr = np.where(frame_mse == 0, np.inf, 10.0 * np.log10((255.0**2) / frame_mse))
    ssim = np.asarray([
        structural_similarity(a, b, channel_axis=-1, data_range=255)
        for a, b in zip(reference, candidate, strict=True)
    ])

    model = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    perceptual = []
    with torch.inference_mode():
        for start in range(0, len(reference), batch_size):
            a = torch.from_numpy(reference[start:start + batch_size]).cuda().permute(0, 3, 1, 2).float()
            b = torch.from_numpy(candidate[start:start + batch_size]).cuda().permute(0, 3, 1, 2).float()
            values = model(a.div(127.5).sub(1), b.div(127.5).sub(1)).flatten()
            perceptual.extend(values.cpu().tolist())
    del model
    torch.cuda.empty_cache()
    perceptual = np.asarray(perceptual)
    return {
        "frames": len(reference),
        "psnr_db": {"mean": float(psnr.mean()), "std": float(psnr.std()), "per_frame": psnr.tolist()},
        "ssim": {"mean": float(ssim.mean()), "std": float(ssim.std()), "per_frame": ssim.tolist()},
        "lpips_alex": {
            "mean": float(perceptual.mean()), "std": float(perceptual.std()),
            "per_frame": perceptual.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, choices=(5, 10), required=True)
    parser.add_argument("--group", choices=("fps", "other", "all"), default="fps")
    args = parser.parse_args()
    duration = args.duration
    expected_frames = 120 if duration == 5 else 240
    reference_path = GALLERY / "original768" / f"{duration}s" / "dense" / "video.mp4"
    reference = read_video(reference_path)
    assert reference.shape == (expected_frames, 768, 1344, 3), reference.shape
    fps_variants = {
        f"{fps}fps_{duration}s_{mode}": VARIANTS / f"{fps}fps" / f"{duration}s" / mode / "video.mp4"
        for fps in (12, 8) for mode in ("legacy", "physical_time")
    }
    other_variants = {
        f"original768_{duration}s_spark_10pct": GALLERY / "original768" / f"{duration}s" / "spark_10pct" / "video.mp4",
        f"original768_{duration}s_spark_target_condition_video_10pct": (
            GALLERY / "expanded_scope" / f"{duration}s" / "spark_target_condition_video_10pct" / "video.mp4"
        ),
        f"reference480_{duration}s_dense": GALLERY / "reference480" / f"{duration}s" / "dense" / "video.mp4",
        f"reference480_{duration}s_spark_10pct": GALLERY / "reference480" / f"{duration}s" / "spark_10pct" / "video.mp4",
    }
    variants = fps_variants if args.group == "fps" else other_variants if args.group == "other" else {
        **fps_variants, **other_variants
    }
    records = {}
    for name, path in variants.items():
        print(f"METRICS {name}", flush=True)
        started = time.perf_counter()
        candidate = read_video(path)
        metrics = compare(reference, candidate)
        records[name] = {
            "variant": name,
            "reference_video": str(reference_path),
            "candidate_video": str(path),
            "alignment": "frame index; decoded RGB uint8; no resize",
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(
            f"DONE {name} PSNR={metrics['psnr_db']['mean']:.4f} "
            f"SSIM={metrics['ssim']['mean']:.6f} LPIPS={metrics['lpips_alex']['mean']:.6f}",
            flush=True,
        )
    suffix = "" if args.group == "fps" else f"_{args.group}"
    output = EXPERIMENT / f"dense_reference_metrics_{duration}s{suffix}.json"
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps({"duration": duration, "records": records}, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
