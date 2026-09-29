"""Score 72 VDN8 ComfyUI videos against their same-model dense references."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess

import comfyui_sol_4way_benchmark_20260923 as api
import comfyui_vdn8_four_model_72_run_20260929 as trial
from comfyui_vdn8_decode_local_20260929 import LOCAL


QUALITY_SCRIPT = trial.BENCH / "scripts" / "h3_quality_video_pair.py"


def score_model(gpu):
    model = trial.MODELS[gpu]
    root = trial.ROOT / model
    local_videos = LOCAL / model / "videos"
    local_videos.mkdir(parents=True, exist_ok=True)
    cases = trial.cases()
    for method in trial.METHODS:
        records = []
        for case in cases:
            stem = f"{case['label']}_{method}"
            original = trial.read(root / "videos" / f"{stem}.json")
            local = local_videos / f"{stem}.mp4"
            if not local.exists():
                shutil.copyfile(original["output_path"], local)
            digest = api.sha256(local)
            if digest != original["sha256"]:
                raise RuntimeError(f"video copy hash mismatch: {local}")
            records.append(dict(index=case["index"], output_path=str(local),
                                sha256=digest, video=original["video"]))
        trial.write(root / "quality" / f"{method}_manifest.json",
                    dict(status="passed", method=method, sample_count=3,
                         records=records))
    summaries = {}
    for method in trial.METHODS[1:]:
        config = dict(work_dir=str(root / "quality" / method),
                      reference_manifest=str(root / "quality" / "dense_manifest.json"),
                      candidate_manifest=str(root / "quality" / f"{method}_manifest.json"),
                      method=method, cases=[1, 2, 3], workers=1)
        config_path = root / "quality" / f"{method}_config.json"
        trial.write(config_path, config)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               "H3_NUM_GPUS": "1", "HF_HUB_OFFLINE": "1",
               "PYTHONUNBUFFERED": "1"}
        subprocess.run([trial.PYTHON, str(QUALITY_SCRIPT), "run", "--config",
                        str(config_path)], check=True, env=env)
        summaries[method] = trial.read(root / "quality" / method /
                                       f"{method}_quality_results.json")["summary"]
        print("SCORED", model, method, summaries[method], flush=True)
    trial.write(root / "quality_results.json", dict(status="complete", model=model,
                                                   summary=summaries))


def aggregate():
    rows = []
    for model in trial.MODELS:
        root = trial.ROOT / model
        for method in trial.METHODS[1:]:
            for case in trial.cases():
                values = trial.read(root / "quality" / method /
                                    f"{method}_{case['index']:02}.json")
                sparse = trial.read(root / "records" /
                                    f"{case['label']}_{method}.json")
                dense = trial.read(root / "records" /
                                   f"{case['label']}_dense.json")
                rows.append(dict(model=model, duration=case["label"], method=method,
                                 psnr_db=values["psnr_db"], ssim=values["ssim"],
                                 lpips=values["lpips"],
                                 dense_sampler_seconds=dense["sampler_seconds"],
                                 sparse_sampler_seconds=sparse["sampler_seconds"],
                                 speedup=dense["sampler_seconds"] / sparse["sampler_seconds"],
                                 reference_video=str(trial.OUT / model /
                                                     f"{case['label']}_dense.mp4"),
                                 sparse_video=str(trial.OUT / model /
                                                  f"{case['label']}_{method}.mp4")))
    trial.write(trial.ROOT / "results.json", dict(status="complete", videos=72,
                                                  scored_pairs=60, rows=rows))
    header = "model,duration,method,psnr_db,ssim,lpips,dense_sampler_seconds,sparse_sampler_seconds,speedup"
    lines = [header] + [",".join(str(row[k]) for k in header.split(",")) for row in rows]
    (trial.ROOT / "results.csv").write_text("\n".join(lines) + "\n")
    print("ALL SCORES COMPLETE", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.gpu is not None:
        score_model(args.gpu)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(score_model, gpu) for gpu in range(4)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        aggregate()


if __name__ == "__main__":
    main()
