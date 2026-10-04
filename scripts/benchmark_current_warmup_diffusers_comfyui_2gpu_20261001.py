#!/usr/bin/env python3
"""Measure current Dense versus Spark-10 on Diffusers and ComfyUI.

GPU 0 runs the compiled Diffusers BF16 stack. GPU 1 runs the native ComfyUI
INT8-convrot stack. Both use one shared cached conditioning, 20 requested
steps, Spark's 20% dense schedule warmup, one permanently dense transformer
layer, one excluded full generation per method and shape, and two measured
generations.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


NAME = "current_warmup_diffusers_comfyui_2gpu_20261001_v2"
REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
COMFY = REPO.parent / "ComfyUI"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
DIFF_OUT = OUT / "diffusers"
COMFY_OUT = OUT / "comfyui"
PYTHON = Path("/root/miniconda3/bin/python")
SAMPLES = BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
CONDITIONING_ROOT = Path(
    "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913"
)
CONDITIONING_FILE = CONDITIONING_ROOT / "conditioning_cache" / (
    "5f6378bba24cd7e3ee647fa1e6dfea3c42acd5f53b30a577d7084242204ac362.pt"
)
EMBEDDING_DIR = COMFY / "models" / "embeddings" / NAME
EMBEDDING_FILE = EMBEDDING_DIR / "shared_case.safetensors"
EMBEDDING_NAME = f"{NAME}/shared_case.safetensors"
BASE_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
WIDTH, HEIGHT, STEPS, SEED = 1344, 768, 20, 42
MEASURED_SEEDS = (42, 43)
DURATIONS = {
    "5s": {"diffusers_frames": 120, "model_frames": 124},
    "10s": {"diffusers_frames": 240, "model_frames": 243},
    "14p4s": {"diffusers_frames": 345, "model_frames": 345},
}
METHODS = ("dense", "spark_10pct")
PORT = 8731


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def prepare() -> None:
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    for path in (
        ROOT / "diffusers" / "records",
        ROOT / "comfyui" / "records",
        DIFF_OUT,
        COMFY_OUT,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for name in ("conditioning_cache", "conditioning_manifest.json"):
        source = CONDITIONING_ROOT / name
        (DIFF_OUT / name).symlink_to(
            source, target_is_directory=source.is_dir()
        )

    import safetensors.torch
    import torch

    cached = torch.load(CONDITIONING_FILE, map_location="cpu", weights_only=False)
    values = cached["values"]
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(
        {
            "conditioning": values["prompt_embeds"].contiguous(),
            "minimax_token_tags": values["text_token_tags"].contiguous(),
        },
        str(EMBEDDING_FILE),
        metadata={"conditioning_options": "{}"},
    )

    source_files = [
        REPO / "h3_sparse_attention" / "processor.py",
        REPO / "h3_sparse_attention" / "spark_integration.py",
        REPO / "h3_sparse_attention" / "global_weighted_route.py",
        REPO / "comfyui_nodes.py",
        REPO / "comfyui_backend.py",
        COMFY / "comfy_extras" / "nodes_sparse_attention.py",
        BENCH / "scripts" / "minimax_h3_vbench_4gpu_pipeline.py",
        Path(__file__).resolve(),
    ]
    config = diffusers_spark_config()
    protocol = {
        "status": "running",
        "name": NAME,
        "purpose": (
            "Current-stack 5s/10s/14.4s Dense versus Spark-H3 Top-K 10% "
            "timing on Diffusers and ComfyUI"
        ),
        "started_unix": time.time(),
        "git_revision": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(REPO), "diff", "--binary"])
        ).hexdigest(),
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "conditioning": {
            "source": str(CONDITIONING_FILE),
            "source_sha256": sha256(CONDITIONING_FILE),
            "converted": str(EMBEDDING_FILE),
            "converted_sha256": sha256(EMBEDDING_FILE),
            "prompt_sha256": hashlib.sha256(cached["prompt"].encode()).hexdigest(),
            "embedding_shape": list(values["prompt_embeds"].shape),
            "embedding_dtype": str(values["prompt_embeds"].dtype),
        },
        "hardware": {
            "diffusers_physical_gpu": 0,
            "comfyui_physical_gpu": 1,
            "required": "NVIDIA RTX PRO 6000 Blackwell, SM120",
        },
        "shared": {
            "resolution": [WIDTH, HEIGHT],
            "requested_steps": STEPS,
            "seed": SEED,
            "measured_seeds": list(MEASURED_SEEDS),
            "durations": DURATIONS,
            "methods": list(METHODS),
            "runtime_warmup": (
                "one excluded full generation for every backend, duration, and method"
            ),
            "timed_repeats": len(MEASURED_SEEDS),
        },
        "diffusers": {
            "model": "/autodl-fs/data/models/MiniMax-H3",
            "dtype": "bfloat16",
            "torch_compile": True,
            "requested_steps": 20,
            "actual_transformer_evaluations": 19,
            "spark_dense_schedule_evaluations": config.dense_evaluations,
            "spark_config": dataclasses.asdict(config),
            "timing": "CUDA-synchronized denoising pipeline only",
        },
        "comfyui": {
            "model": BASE_MODEL,
            "model_path": str(
                Path("/autodl-fs/data/models/ComfyUI-MiniMax-H3/diffusion_models")
                / BASE_MODEL
            ),
            "requested_steps": 20,
            "actual_model_evaluations": 20,
            "spark_dense_schedule_evaluations": 4,
            "spark": {
                "warmup_mode": "warmup_ratio",
                "warmup_ratio": 0.2,
                "warmup_steps": 4,
                "topk_mode": "topk_ratio",
                "topk_ratio": 0.1,
                "dense_layers": 1,
                "min_tokens": 12288,
                "tail_granularity": "query",
                "global_anchor_dtype": "float32 (node default)",
                "midpoint_direction_mode": "fused (node default)",
            },
            "timing": "SamplerCustomAdvanced node wall time only",
        },
    }
    write_json(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def pipeline_module():
    scripts = BENCH / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import _impl_bootstrap  # noqa: F401
    import minimax_h3_vbench_4gpu_pipeline

    return minimax_h3_vbench_4gpu_pipeline


def diffusers_args():
    return pipeline_module().build_parser().parse_args(
        [
            "denoise",
            "--samples",
            str(SAMPLES),
            "--output",
            str(DIFF_OUT),
            "--method",
            "dense",
            "--case-indices",
            "1",
            "--steps",
            str(STEPS),
            "--frames",
            "120",
            "--height",
            str(HEIGHT),
            "--width",
            str(WIDTH),
            "--workers",
            "1",
        ]
    )


def diffusers_spark_config():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from h3_sparse_attention import H3SparseAttentionConfig

    return H3SparseAttentionConfig.spark(
        STEPS,
        warmup_percent=20.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=0.10,
        sol_log_density=False,
    )


def run_diffusers_once(pipe, state, *, frames: int, seed: int, plugin=None) -> dict:
    import torch

    p = pipeline_module()
    inference_state = p.clone_state(state)
    inference_state.values["prompt_embeds"] = inference_state.values[
        "prompt_embeds"
    ].to("cuda")
    if plugin is not None:
        plugin.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = pipe(
            state=inference_state,
            num_frames=frames,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output=["latents", "audio_latents"],
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    finite = all(
        torch.isfinite(output[name]).all().item()
        for name in ("latents", "audio_latents")
    )
    summary = None if plugin is None else plugin.summary()
    record = {
        "seconds": seconds,
        "seed": seed,
        "finite": finite,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "attention_summary": summary,
    }
    del output, inference_state
    return record


def diffusers_worker() -> None:
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    p = pipeline_module()
    args = diffusers_args()
    cases = p.load_cases(SAMPLES, [1], expected_indices=tuple(range(1, 51)))
    workflow, states = p.configure_denoise_workflow(args, cases)
    loaded_at = time.perf_counter()
    pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    load_seconds = time.perf_counter() - loaded_at
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "placement": placement,
        "load_seconds": load_seconds,
    }
    write_json(ROOT / "diffusers" / "worker.json", metadata)
    try:
        for duration_index, (duration, shape) in enumerate(DURATIONS.items()):
            method_order = METHODS if duration_index % 2 == 0 else METHODS[::-1]
            for method in method_order:
                config = diffusers_spark_config() if method == "spark_10pct" else None
                context = (
                    install_h3_sparse_attention(pipe.transformer, config)
                    if config is not None
                    else contextlib.nullcontext(None)
                )
                with context as plugin:
                    for phase, repeat, seed in (
                        ("warmup", 0, 41),
                        ("measured", 1, MEASURED_SEEDS[0]),
                        ("measured", 2, MEASURED_SEEDS[1]),
                    ):
                        result = run_diffusers_once(
                            pipe,
                            states[0],
                            frames=shape["diffusers_frames"],
                            seed=seed,
                            plugin=plugin,
                        )
                        summary = result["attention_summary"]
                        if summary is not None:
                            assert summary["completed_evaluations"] == 19, summary
                            assert summary["dense_evaluations"] == 4, summary
                        record = {
                            "backend": "diffusers",
                            "duration": duration,
                            "method": method,
                            "phase": phase,
                            "repeat": repeat,
                            "requested_frames": shape["diffusers_frames"],
                            "model_frames": shape["model_frames"],
                            **result,
                        }
                        target = ROOT / "diffusers" / "records" / (
                            f"{duration}_{method}_{phase}_{repeat}.json"
                        )
                        write_json(target, record)
                        print(
                            "DIFFUSERS",
                            duration,
                            method,
                            phase,
                            repeat,
                            f"{result['seconds']:.3f}s",
                            flush=True,
                        )
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def comfy_graph(method: str, frames: int, seed: int, target: Path) -> dict:
    nodes = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": BASE_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "ConditioningLoader",
            "inputs": {"conditioning_name": EMBEDDING_NAME},
        },
        "4": {
            "class_type": "EmptyMiniMaxH3LatentAV",
            "inputs": {"width": WIDTH, "height": HEIGHT, "length": frames},
        },
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": None,
                "scheduler": "simple",
                "steps": STEPS,
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": None, "conditioning": ["3", 0]},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["10", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["4", 0],
            },
        },
        "12": {
            "class_type": "SaveMiniMaxH3AVLatentCache",
            "inputs": {"samples": ["11", 0], "cache_path": str(target)},
        },
    }
    model = ["1", 0]
    if method == "spark_10pct":
        nodes["6"] = {
            "class_type": "MiniMaxH3SparkAttentionSM120",
            "inputs": {
                "model": model,
                "enabled": True,
                "steps": STEPS,
                "warmup_mode": "warmup_ratio",
                "warmup_ratio": 0.2,
                "warmup_steps": 4,
                "topk_mode": "topk_ratio",
                "topk_ratio": 0.1,
                "topk_blocks": 114,
                "dense_layers": 1,
                "min_tokens": 12288,
                "strict": True,
                "tail_granularity": "query",
            },
        }
        model = ["6", 0]
    nodes["9"]["inputs"]["model"] = model
    nodes["10"]["inputs"]["model"] = model
    return nodes


def start_comfy_server():
    for name in ("user", "input", "temp", "latents"):
        (ROOT / "comfyui" / name).mkdir(parents=True, exist_ok=True)
    log = (ROOT / "comfyui" / "server.log").open("a")
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    command = [
        str(PYTHON),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--disable-auto-launch",
        "--disable-cuda-malloc",
        "--preview-method",
        "none",
        "--output-directory",
        str(COMFY_OUT),
        "--temp-directory",
        str(ROOT / "comfyui" / "temp"),
        "--input-directory",
        str(ROOT / "comfyui" / "input"),
        "--user-directory",
        str(ROOT / "comfyui" / "user"),
    ]
    return (
        subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT),
        log,
    )


def comfyui_worker() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import comfyui_sol_4way_benchmark_20260923 as api

    process, log = start_comfy_server()
    try:
        stats = api.wait_server(PORT)
        write_json(ROOT / "comfyui" / "worker.json", stats)
        for duration_index, (duration, shape) in enumerate(DURATIONS.items()):
            method_order = METHODS[::-1] if duration_index % 2 == 0 else METHODS
            for method in method_order:
                for phase, repeat, seed in (
                    ("warmup", 0, 41),
                    ("measured", 1, MEASURED_SEEDS[0]),
                    ("measured", 2, MEASURED_SEEDS[1]),
                ):
                    target = ROOT / "comfyui" / "latents" / (
                        f"{duration}_{method}_{phase}_{repeat}.safetensors"
                    )
                    result = api.queue_and_wait(
                        PORT,
                        comfy_graph(method, shape["model_frames"], seed, target),
                        "11",
                    )
                    if result.get("sampler_seconds") is None or not target.is_file():
                        raise RuntimeError(
                            f"missing ComfyUI timing or latent: {duration} {method} {phase}"
                        )
                    target.unlink()
                    result.pop("history", None)
                    record = {
                        "backend": "comfyui",
                        "duration": duration,
                        "method": method,
                        "phase": phase,
                        "repeat": repeat,
                        "seed": seed,
                        "model_frames": shape["model_frames"],
                        "seconds": result["sampler_seconds"],
                        **result,
                    }
                    write_json(
                        ROOT / "comfyui" / "records" / (
                            f"{duration}_{method}_{phase}_{repeat}.json"
                        ),
                        record,
                    )
                    print(
                        "COMFYUI",
                        duration,
                        method,
                        phase,
                        repeat,
                        f"{record['seconds']:.3f}s",
                        flush=True,
                    )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def load_measured(backend: str) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((ROOT / backend / "records").glob("*.json"))
        if json.loads(path.read_text())["phase"] == "measured"
    ]


def summarize() -> None:
    summary = {}
    for backend in ("diffusers", "comfyui"):
        rows = load_measured(backend)
        summary[backend] = {}
        for duration in DURATIONS:
            summary[backend][duration] = {}
            for method in METHODS:
                values = [
                    row["seconds"]
                    for row in rows
                    if row["duration"] == duration and row["method"] == method
                ]
                if len(values) != len(MEASURED_SEEDS):
                    raise RuntimeError(
                        f"missing measurements: {backend} {duration} {method}: {values}"
                    )
                summary[backend][duration][method] = {
                    "seconds": values,
                    "median_seconds": statistics.median(values),
                    "mean_seconds": statistics.mean(values),
                    "range_seconds": max(values) - min(values),
                }
            dense = summary[backend][duration]["dense"]["median_seconds"]
            spark = summary[backend][duration]["spark_10pct"]["median_seconds"]
            summary[backend][duration]["speedup_x"] = dense / spark

    result = {
        "status": "complete",
        "completed_unix": time.time(),
        "summary": summary,
    }
    write_json(ROOT / "results.json", result)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete"
    protocol["completed_unix"] = result["completed_unix"]
    write_json(ROOT / "protocol.json", protocol)

    lines = [
        "# Current warmup: Diffusers vs ComfyUI Dense/Spark-10",
        "",
        "One excluded full generation per backend/method/shape; two measured generations. "
        "Speedup is the ratio of Dense median to Spark median.",
        "",
        "| Backend | Duration | Dense runs (s) | Spark-10 runs (s) | Dense median (s) | Spark median (s) | Speedup |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for backend in ("diffusers", "comfyui"):
        for duration in DURATIONS:
            row = summary[backend][duration]
            dense = row["dense"]
            spark = row["spark_10pct"]
            lines.append(
                f"| {backend} | {duration.replace('14p4s', '14.4s')} | "
                f"{', '.join(f'{x:.2f}' for x in dense['seconds'])} | "
                f"{', '.join(f'{x:.2f}' for x in spark['seconds'])} | "
                f"{dense['median_seconds']:.2f} | {spark['median_seconds']:.2f} | "
                f"{row['speedup_x']:.3f}x |"
            )
    lines.extend(
        [
            "",
            "Diffusers requests 120/240/345 frames and internally aligns them to "
            "124/243/345; ComfyUI receives the aligned lengths directly. Diffusers "
            "executes 19 transformer evaluations for 20 requested scheduler points, "
            "whereas ComfyUI executes 20 sampler model evaluations. Both use four "
            "initial dense evaluations and dense transformer layer 0.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def run_all() -> None:
    prepare()
    common = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "FLASHINFER_CUDA_ARCH_LIST": "12.0",
        "H3_SOL_LAYOUT_FAST": "1",
        "H3_METRIC_FACTOR": "cholesky",
        "H3_LMV2_COS_PRECISION": "fp16",
        "H3_LMV2_FUSED_NODE": "1",
        "H3_LMV2_GROUP1_FAST": "1",
        "H3_LMV2_COS_FAST": "1",
        "H3_LMV2_SMALL_PROXY_FAST": "1",
        "H3_LMV2_FP8_FEATURES": "0",
        "H3_IMPL_REPO": str(REPO),
    }
    jobs = []
    for backend, gpu in (("diffusers", 0), ("comfyui", 1)):
        log = (ROOT / f"{backend}.log").open("a")
        env = {**common, "CUDA_VISIBLE_DEVICES": str(gpu)}
        process = subprocess.Popen(
            [str(PYTHON), str(Path(__file__).resolve()), f"{backend}-worker"],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((backend, process, log))

    failure = None
    while any(process.poll() is None for _, process, _ in jobs):
        for backend, process, _ in jobs:
            if process.poll() not in (None, 0):
                failure = f"{backend} worker failed with exit code {process.returncode}"
                break
        if failure:
            break
        time.sleep(5)
    if failure:
        for _, process, _ in jobs:
            if process.poll() is None:
                process.terminate()
        for _, process, log in jobs:
            process.wait()
            log.close()
        write_json(ROOT / "failure.json", {"status": "failed", "error": failure})
        raise RuntimeError(failure)

    for backend, process, log in jobs:
        returncode = process.wait()
        log.close()
        if returncode:
            raise RuntimeError(f"{backend} worker failed with exit code {returncode}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run_all()
    elif command == "diffusers-worker":
        diffusers_worker()
    elif command == "comfyui-worker":
        comfyui_worker()
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
