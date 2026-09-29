#!/usr/bin/env python3
"""Decode the 15-prompt Spark tail ablation and score paired RGB quality."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base


ROOT = base.ROOT
OUT = base.OUT
CASES = tuple(range(1, 16))
METHODS = base.METHODS
SPARSE_METHODS = base.SPARSE_METHODS
GPUS = base.GPUS
FV_EVAL = base.IMPL.parent / "FastVideo/examples/inference/eval"
QUALITY_SCRIPT = base.SCRIPTS / "h3_quality_video_pair.py"


def parse_decode_args():
    return base.pipeline().build_parser().parse_args([
        "decode",
        "--samples", str(base.SAMPLES),
        "--output", str(OUT),
        "--method", "dense",
        "--steps", str(base.STEPS),
        "--frames", str(base.FRAMES),
        "--height", str(base.HEIGHT),
        "--width", str(base.WIDTH),
        "--workers", "1",
        "--no-torch-compile",
    ])


def decode_worker(rank):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    if str(FV_EVAL) not in sys.path:
        sys.path.insert(0, str(FV_EVAL))
    from fasth3_vbench_archive import archive, file_sha

    p = base.pipeline()
    all_cases = p.load_cases(
        base.SAMPLES, list(CASES), expected_indices=base.SOURCE_CASES
    )
    cases = all_cases[rank::len(GPUS)]
    pipe, manager, acceleration = p.load_decoder(parse_decode_args())
    try:
        for case in cases:
            stem = p.case_stem(case)
            for method in METHODS:
                latent = OUT / method / "latents" / f"{stem}.pt"
                target = OUT / method / "videos" / f"{stem}.mkv"
                record = ROOT / "decode_records" / method / f"case_{case['index']:02}.json"
                if record.exists() and target.exists():
                    old = base.read(record)
                    if file_sha(target) == old["sha256"]:
                        print("REUSE", rank, method, case["index"], flush=True)
                        continue
                payload = torch.load(latent, map_location="cpu", weights_only=True)
                with torch.inference_mode():
                    result = pipe(
                        latents=payload["latents"].to("cuda"),
                        audio_latents=payload["audio_latents"].to("cuda"),
                        output_type="np",
                        output=["videos", "audio", "sampling_rate"],
                    )
                floats = result["videos"][0][:base.FRAMES]
                assert np.isfinite(floats).all()
                frames = np.clip(np.round(floats * 255), 0, 255).astype(np.uint8)
                assert frames.shape == (base.FRAMES, base.HEIGHT, base.WIDTH, 3)
                metadata = archive(
                    frames, result["audio"][0], int(result["sampling_rate"]),
                    target, fps=base.FPS, threads=8,
                )
                base.write(record, dict(
                    status="complete", method=method, case=case["index"],
                    sample_id=case["sample_id"], output_path=str(target),
                    sha256=file_sha(target), archive=metadata,
                ))
                print("DECODED", rank, method, case["index"], flush=True)
                del payload, result, floats, frames
        base.write(ROOT / f"quality_decode_gpu{rank}.json", dict(status="complete"))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def spawn_decode():
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"quality_decode_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(base.PYTHON), str(__file__), "decode_worker", str(rank)],
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"decode workers failed: {codes}")


def manifests():
    p = base.pipeline()
    cases = p.load_cases(
        base.SAMPLES, list(CASES), expected_indices=base.SOURCE_CASES
    )
    for method in METHODS:
        records = []
        for case in cases:
            decoded = base.read(
                ROOT / "decode_records" / method / f"case_{case['index']:02}.json"
            )
            assert base.sha(decoded["output_path"]) == decoded["sha256"]
            records.append(dict(
                index=case["index"], sample_id=case["sample_id"],
                prompt_sha256=case["prompt_sha256"],
                evaluation_prompt=case["original_prompt"],
                vbench_dimensions=case["vbench_dimensions"],
                output_path=decoded["output_path"], sha256=decoded["sha256"],
                video=dict(
                    frames=base.FRAMES, width=base.WIDTH,
                    height=base.HEIGHT, fps=float(base.FPS),
                ),
            ))
        base.write(OUT / method / "generation_manifest.json", dict(
            schema_version=1, status="passed", method=method,
            sample_count=len(records), settings=dict(
                seed=base.SEED, steps=base.STEPS, frames=base.FRAMES,
                height=base.HEIGHT, width=base.WIDTH, fps=base.FPS,
            ), records=records,
        ))


def quality():
    summaries = {}
    for method in SPARSE_METHODS:
        work = ROOT / "quality" / method
        config = dict(
            work_dir=str(work),
            reference_manifest=str(OUT / "dense" / "generation_manifest.json"),
            candidate_manifest=str(OUT / method / "generation_manifest.json"),
            method=method,
            cases=list(CASES),
            workers=len(GPUS),
        )
        config_path = ROOT / "quality" / f"{method}_config.json"
        base.write(config_path, config)
        subprocess.run(
            [str(base.PYTHON), str(QUALITY_SCRIPT), "run", "--config", str(config_path)],
            check=True,
            env={**os.environ, **base.ENV, "H3_NUM_GPUS": str(len(GPUS))},
        )
        result = base.read(work / f"{method}_quality_results.json")
        summaries[method] = result["summary"]
    base.write(ROOT / "quality_results.json", dict(
        status="complete", reference="dense", cases=list(CASES),
        summaries=summaries,
    ))
    status_path = base.IMPL / "tail_exp_status.json"
    status = base.read(status_path)
    status.update(
        quality_status="complete",
        quality_reference="dense",
        quality_results_path=str(ROOT / "quality_results.json"),
        quality=summaries,
    )
    base.write(status_path, status)
    print(json.dumps(summaries, indent=2), flush=True)


def run():
    status_path = base.IMPL / "tail_exp_status.json"
    status = base.read(status_path)
    status.update(
        quality_status="running",
        quality_reference="dense",
        quality_message="Decoding 60 latents and computing paired PSNR/SSIM/LPIPS.",
    )
    base.write(status_path, status)
    spawn_decode()
    manifests()
    quality()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "decode_worker":
        decode_worker(int(sys.argv[2]))
    elif command in globals():
        globals()[command]()
    else:
        raise SystemExit(f"unknown command: {command}")
