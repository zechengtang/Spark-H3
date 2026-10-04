#!/usr/bin/env python3
"""Two-prompt full-warmup reproduction of blog Table 3 Dense and Sol-H3."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
SEPT13 = Path("/autodl-fs/data/h3_experiments/vbench20pct_768p10s_seed42_20260913")
ISO = SEPT13 / "isolated_run"
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
NAME = "blog_table3_dense_sol_full_warmup_2gpu_20261004"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = ISO / "vbench_core5_percent_subsets/20pct/samples.json"
CASE_INDICES = (2, 13)
GPUS = (0, 1)
METHODS = ("dense", "sol")
BLOG_SECONDS = {"dense": 582.1, "sol": 364.9}
FRAMES, HEIGHT, WIDTH, STEPS, SEED = 240, 768, 1344, 20, 42
PYTHON = Path("/root/miniconda3/bin/python")
ENV = {
    "HF_HUB_OFFLINE": "1",
    "OMP_NUM_THREADS": "4",
    "TORCHINDUCTOR_COMPILE_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "FLASHINFER_CUDA_ARCH_LIST": "12.0",
}


def write(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*items: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *items], text=True).strip()


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "blog_dense_sol_pipeline", ISO / "scripts/minimax_h3_vbench_4gpu_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def args(pipeline):
    return pipeline.build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--case-indices", *map(str, CASE_INDICES),
        "--steps", str(STEPS), "--frames", str(FRAMES),
        "--height", str(HEIGHT), "--width", str(WIDTH), "--workers", "1",
    ])


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(
            SOURCE / name, target_is_directory=name.endswith("cache")
        )
    pipeline = load_pipeline()
    selected = pipeline.load_cases(SAMPLES, list(CASE_INDICES))
    pipeline.configure_denoise_workflow(args(pipeline), selected)
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
    protocol = {
        "status": "running",
        "name": NAME,
        "purpose": "reproduce Dense and stock Sol-H3 latency rows in blog Table 3",
        "cases": selected,
        "gpus": list(GPUS),
        "methods": list(METHODS),
        "method_order": {"gpu0": ["dense", "sol"], "gpu1": ["sol", "dense"]},
        "blog_seconds": BLOG_SECONDS,
        "formal": {"steps": STEPS, "evaluations": 19, "samples_per_method": 2},
        "warmup": {
            "per_method_per_gpu": True,
            "steps": STEPS,
            "evaluations": 19,
            "excluded_from_timing_results": True,
            "reason": "Dense and stock Sol require a complete denoise warmup",
        },
        "sol_code": str(ISO / "h3_sparse_attention"),
        "sol_config": "frozen Sept-13 H3SparseAttentionConfig.sol(20)",
        "dense_config": "no attention plugin",
        "frames": FRAMES,
        "resolution": [WIDTH, HEIGHT],
        "seed": SEED,
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": sha(Path(__file__).resolve()),
        "timing": "CUDA-synchronized denoise only; complete warmup excluded",
    }
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def run_once(pipe, pipeline, original_state, method: str, sol_config):
    import torch

    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    if method == "sol":
        import h3_sparse_attention as frozen_h3
        context = frozen_h3.install_h3_sparse_attention(pipe.transformer, sol_config)
    else:
        context = contextlib.nullcontext()
    with context, torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(
            state=state,
            num_frames=FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    assert all(
        torch.isfinite(output[name]).all().item()
        for name in ("latents", "audio_latents")
    )
    del output, state
    return seconds


def worker(rank: int):
    import torch

    rank = int(rank)
    gpu = GPUS[rank]
    torch.set_num_threads(4)
    pipeline = load_pipeline()
    # The frozen pipeline prepends ISO before importing its attention package.
    import h3_sparse_attention as frozen_h3
    if not str(Path(frozen_h3.__file__).resolve()).startswith(str(ISO.resolve())):
        raise RuntimeError(f"expected frozen Sol package, got {frozen_h3.__file__}")
    sol_config = frozen_h3.H3SparseAttentionConfig.sol(STEPS)
    selected = pipeline.load_cases(SAMPLES, list(CASE_INDICES))
    workflow, states = pipeline.configure_denoise_workflow(args(pipeline), selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args(pipeline), workflow)
    method_order = METHODS if rank == 0 else tuple(reversed(METHODS))
    try:
        for order, method in enumerate(method_order):
            warm_seconds = run_once(pipe, pipeline, states[rank], method, sol_config)
            write(ROOT / "records" / f"gpu{gpu}_{method}_warmup.json", {
                "phase": "discarded_full_warmup", "gpu": gpu, "order": order,
                "method": method, "case": selected[rank]["index"],
                "sample_id": selected[rank]["sample_id"], "placement": placement,
                "steps": STEPS, "evaluations": 19, "seconds": warm_seconds,
            })
            print("WARMUP", method, round(warm_seconds, 3), flush=True)
            seconds = run_once(pipe, pipeline, states[rank], method, sol_config)
            write(ROOT / "records" / f"gpu{gpu}_{method}_measured.json", {
                "phase": "measured", "gpu": gpu, "order": order,
                "method": method, "case": selected[rank]["index"],
                "sample_id": selected[rank]["sample_id"],
                "prompt_sha256": selected[rank]["prompt_sha256"],
                "placement": placement, "steps": STEPS, "evaluations": 19,
                "seconds": seconds,
            })
            print("MEASURED", method, round(seconds, 3), flush=True)
        write(ROOT / f"worker_gpu{gpu}.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager


def summarize():
    rows = [
        json.loads(path.read_text())
        for path in (ROOT / "records").glob("*_measured.json")
    ]
    result = {"status": "complete", "methods": {}}
    for method in METHODS:
        chosen = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: row["gpu"]
        )
        if len(chosen) != len(GPUS):
            raise RuntimeError(f"missing {method}: {len(chosen)}")
        values = [row["seconds"] for row in chosen]
        mean = statistics.mean(values)
        target = BLOG_SECONDS[method]
        result["methods"][method] = {
            "seconds": values,
            "mean_seconds": mean,
            "blog_seconds": target,
            "delta_seconds": mean - target,
            "delta_percent": (mean / target - 1.0) * 100.0,
            "within_2_percent": abs(mean / target - 1.0) <= 0.02,
        }
    result["all_within_2_percent"] = all(
        row["within_2_percent"] for row in result["methods"].values()
    )
    write(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    print(json.dumps(result, indent=2), flush=True)


def launch():
    prepare()
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"gpu{gpu}.log").open("a")
        proc = subprocess.Popen(
            [str(PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, proc, log))
    codes = [(gpu, proc.wait()) for gpu, proc, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"workers failed: {codes}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(int(sys.argv[2]))
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
