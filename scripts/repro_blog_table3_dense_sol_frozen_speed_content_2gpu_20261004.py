#!/usr/bin/env python3
"""Frozen Sept-13 Dense/Sol timing and latent consistency on cases 14/20."""
from __future__ import annotations

import contextlib
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
NAME = "blog_table3_dense_sol_frozen_speed_content_2gpu_20261004"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = ISO / "vbench_core5_percent_subsets/20pct/samples.json"
CASE_INDICES = (14, 20)
CASE_GROUPS = {14: "remapped", 20: "non_remapped"}
REMAPPED_CASES = (2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 29,
                  31, 33, 35, 37, 39, 41, 43, 45, 47, 49)
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
        "blog_dense_sol_frozen_content_pipeline",
        ISO / "scripts/minimax_h3_vbench_4gpu_pipeline.py",
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


def reference_latent(method: str, case: dict) -> Path:
    return SOURCE / method / "latents" / f"{case['index']:02d}_{case['sample_id']}.pt"


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(SOURCE / name, target_is_directory=name.endswith("cache"))
    pipeline = load_pipeline()
    selected = pipeline.load_cases(SAMPLES, list(CASE_INDICES))
    assert selected[0]["index"] in REMAPPED_CASES
    assert selected[1]["index"] not in REMAPPED_CASES
    pipeline.configure_denoise_workflow(args(pipeline), selected)
    references = {}
    for method in METHODS:
        references[method] = {}
        for case in selected:
            path = reference_latent(method, case)
            if not path.is_file():
                raise FileNotFoundError(path)
            references[method][str(case["index"])] = {
                "path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            }
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
    protocol = {
        "status": "running",
        "name": NAME,
        "purpose": "protocol-correct rerun of frozen Sept-13 Dense/Sol on one remapped and one non-remapped prompt",
        "supersedes_for_dense_sol": "blog_table3_speed_content_2gpu_20261004",
        "cases": [{**case, "provenance_group": CASE_GROUPS[case["index"]]} for case in selected],
        "gpus": list(GPUS),
        "methods": list(METHODS),
        "method_order": {"gpu0": ["dense", "sol"], "gpu1": ["sol", "dense"]},
        "warmup": {
            "exactly_once_per_gpu_per_method": True,
            "dense": {"steps": 2, "evaluations": 1},
            "sol": {"steps": 20, "evaluations": 19},
            "excluded_from_timing": True,
        },
        "formal": {"steps": STEPS, "evaluations": 19, "seed": SEED,
                   "frames": FRAMES, "resolution": [WIDTH, HEIGHT]},
        "sol_code": str(ISO / "h3_sparse_attention"),
        "sol_config": "frozen Sept-13 H3SparseAttentionConfig.sol(20)",
        "dense_config": "frozen Sept-13 pipeline; no attention plugin",
        "references": references,
        "content_artifacts": "latents and audio_latents persisted as CPU float32 .pt files",
        "content_comparison": "sha256, torch.equal, max/mean absolute error, RMSE, cosine",
        "blog_seconds": BLOG_SECONDS,
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": sha(Path(__file__).resolve()),
        "timing": "CUDA-synchronized denoise only; transfer/save/comparison excluded",
    }
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def compare_payload(payload: dict, reference_path: Path):
    import torch

    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    result = {}
    for key in ("latents", "audio_latents"):
        actual = payload[key]
        expected = reference[key].contiguous()
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(
                f"content schema mismatch for {key}: "
                f"{actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}"
            )
        delta = actual.double() - expected.double()
        result[key] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "torch_equal": bool(torch.equal(actual, expected)),
            "max_abs_diff": float(delta.abs().max().item()),
            "mean_abs_diff": float(delta.abs().mean().item()),
            "rmse": float(delta.square().mean().sqrt().item()),
            "cosine": float(torch.nn.functional.cosine_similarity(
                actual.flatten().double(), expected.flatten().double(), dim=0
            ).item()),
        }
    del reference
    return result


def run_once(pipe, pipeline, original_state, method: str, sol_config, *, steps: int,
             save_path: Path | None):
    import torch

    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    if method == "sol":
        import h3_sparse_attention as frozen_h3
        config = sol_config if steps == STEPS else frozen_h3.H3SparseAttentionConfig.sol(steps)
        context = frozen_h3.install_h3_sparse_attention(pipe.transformer, config)
    else:
        context = contextlib.nullcontext()
    with context, torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(
            state=state, num_frames=FRAMES, height=HEIGHT, width=WIDTH,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            output=["latents", "audio_latents"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    assert all(torch.isfinite(output[key]).all().item() for key in ("latents", "audio_latents"))
    payload = None
    if save_path is not None:
        payload = {
            key: output[key].detach().cpu().float().contiguous()
            for key in ("latents", "audio_latents")
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.atomic_torch_save(payload, save_path)
    del output, state
    return seconds, payload


def worker(rank: int):
    import torch

    rank = int(rank)
    gpu = GPUS[rank]
    torch.set_num_threads(4)
    pipeline = load_pipeline()
    import h3_sparse_attention as frozen_h3
    if not str(Path(frozen_h3.__file__).resolve()).startswith(str(ISO.resolve())):
        raise RuntimeError(f"expected frozen Sol package, got {frozen_h3.__file__}")
    sol_config = frozen_h3.H3SparseAttentionConfig.sol(STEPS)
    selected = pipeline.load_cases(SAMPLES, list(CASE_INDICES))
    workflow, states = pipeline.configure_denoise_workflow(args(pipeline), selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args(pipeline), workflow)
    method_order = METHODS if rank == 0 else tuple(reversed(METHODS))
    case = selected[rank]
    try:
        for order, method in enumerate(method_order):
            warm_steps = 2 if method == "dense" else STEPS
            warm_seconds, _ = run_once(
                pipe, pipeline, states[rank], method, sol_config,
                steps=warm_steps, save_path=None,
            )
            write(ROOT / "records" / f"gpu{gpu}_{method}_warmup.json", {
                "phase": "discarded_warmup", "gpu": gpu, "order": order,
                "method": method, "case": case["index"], "group": CASE_GROUPS[case["index"]],
                "steps": warm_steps, "evaluations": warm_steps - 1,
                "seconds": warm_seconds,
            })
            print("WARMUP", method, round(warm_seconds, 3), flush=True)
            latent_path = OUT / method / "latents" / f"{case['index']:02d}_{case['sample_id']}.pt"
            seconds, payload = run_once(
                pipe, pipeline, states[rank], method, sol_config,
                steps=STEPS, save_path=latent_path,
            )
            reference = reference_latent(method, case)
            comparison = compare_payload(payload, reference)
            record = {
                "phase": "measured", "gpu": gpu, "order": order,
                "method": method, "case": case["index"], "sample_id": case["sample_id"],
                "group": CASE_GROUPS[case["index"]], "prompt_sha256": case["prompt_sha256"],
                "placement": placement, "steps": STEPS, "evaluations": STEPS - 1,
                "seconds": seconds,
                "latent_path": str(latent_path), "latent_sha256": sha(latent_path),
                "reference_latent_path": str(reference), "reference_latent_sha256": sha(reference),
                "file_sha256_equal": sha(latent_path) == sha(reference),
                "content_comparison": comparison,
            }
            write(ROOT / "records" / f"gpu{gpu}_{method}_measured.json", record)
            print(
                "MEASURED", method, round(seconds, 3),
                "latent_equal", comparison["latents"]["torch_equal"], flush=True,
            )
            del payload
        write(ROOT / f"worker_gpu{gpu}.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager


def summarize():
    rows = [json.loads(path.read_text()) for path in (ROOT / "records").glob("*_measured.json")]
    result = {"status": "complete", "methods": {}, "content": {}}
    for method in METHODS:
        chosen = sorted((row for row in rows if row["method"] == method), key=lambda row: row["gpu"])
        if len(chosen) != len(GPUS):
            raise RuntimeError(f"missing {method}: {len(chosen)}")
        values = [row["seconds"] for row in chosen]
        mean = statistics.mean(values)
        target = BLOG_SECONDS[method]
        result["methods"][method] = {
            "seconds": values, "mean_seconds": mean, "blog_seconds": target,
            "delta_seconds": mean - target, "delta_percent": (mean / target - 1.0) * 100.0,
        }
        result["content"][method] = {
            str(row["case"]): {
                "group": row["group"], "file_sha256_equal": row["file_sha256_equal"],
                "latents": row["content_comparison"]["latents"],
                "audio_latents": row["content_comparison"]["audio_latents"],
                "artifact": row["latent_path"], "reference": row["reference_latent_path"],
            }
            for row in chosen
        }
    result["all_latent_tensors_equal"] = all(
        item["latents"]["torch_equal"] and item["audio_latents"]["torch_equal"]
        for method in result["content"].values() for item in method.values()
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
            stdout=log, stderr=subprocess.STDOUT,
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
