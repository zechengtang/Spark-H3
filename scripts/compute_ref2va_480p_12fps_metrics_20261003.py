#!/usr/bin/env python3
"""Metrics for the 480p/12fps Ref2VA Dense and Spark outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compute_ref2va_dense_reference_metrics_20261003 import compare, read_video  # noqa: E402


REFERENCE = Path(
    "/autodl-fs/data/h3_outputs/ref2va_768_480_dense_spark_20261003/"
    "original768/5s/dense/video.mp4"
)
ROOT = Path("/autodl-fs/data/h3_outputs/ref2va_480p_12fps_dense_spark_20261003")
EXPERIMENT = Path("/autodl-fs/data/h3_experiments/ref2va_480p_12fps_dense_spark_20261003")


def main() -> None:
    reference = read_video(REFERENCE)
    assert reference.shape == (120, 768, 1344, 3), reference.shape
    records = {}
    for method in ("dense", "spark_target_condition_video_10pct"):
        path = ROOT / method / "video.mp4"
        started = time.perf_counter()
        metrics = compare(reference, read_video(path))
        records[method] = {
            "reference_video": str(REFERENCE),
            "candidate_video": str(path),
            "alignment": "frame index; decoded RGB uint8; no resize",
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(
            f"DONE {method} PSNR={metrics['psnr_db']['mean']:.4f} "
            f"SSIM={metrics['ssim']['mean']:.6f} "
            f"LPIPS={metrics['lpips_alex']['mean']:.6f}",
            flush=True,
        )
    output = EXPERIMENT / "dense_reference_metrics.json"
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps({"records": records}, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
