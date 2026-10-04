#!/usr/bin/env python3
"""Two new prompts: Table-3 timing plus persisted latent consistency."""
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

import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base


NAME = "blog_table3_speed_content_2gpu_20261004"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = base.BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
SOURCE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
SPARK12 = Path("/autodl-fs/data/h3_outputs/topk_reblock_reweight_50prompt_20260920")
SPARK30 = Path("/autodl-fs/data/h3_outputs/diffusers_spark_topk30_50prompt_10s768p_20260928")
CASE_INDICES = (14, 20)
CASE_GROUPS = {14: "remapped", 20: "non_remapped"}
REMAPPED_CASES = (2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 29,
                  31, 33, 35, 37, 39, 41, 43, 45, 47, 49)
GPUS = (0, 1)
METHODS = ("dense", "sol", "spark10", "spark20", "spark30")
RATIOS = {"spark10": 0.10, "spark20": 0.20, "spark30": 0.30}
BLOG_SECONDS = {"dense": 582.1, "sol": 364.9, "spark10": 341.7,
                "spark20": 378.2, "spark30": 408.6}
FRAMES, HEIGHT, WIDTH, STEPS, SEED = 240, 768, 1344, 20, 42


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
    return subprocess.check_output(["git", "-C", str(base.IMPL), *items], text=True).strip()


def cases():
    return base.pipeline().load_cases(
        SAMPLES, list(CASE_INDICES), expected_indices=tuple(range(1, 51))
    )


def args():
    return base.pipeline().build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(OUT),
        "--method", "dense", "--case-indices", *map(str, CASE_INDICES),
        "--steps", str(STEPS), "--frames", str(FRAMES),
        "--height", str(HEIGHT), "--width", str(WIDTH), "--workers", "1",
    ])


def attention_config(method: str, steps: int):
    from h3_sparse_attention import H3SparseAttentionConfig

    if method == "sol":
        return H3SparseAttentionConfig.sol(steps)
    ratio = RATIOS[method]
    return H3SparseAttentionConfig.spark(
        steps,
        warmup_percent=20.0 if steps == STEPS else 33.0,
        sol_dense_layers=1,
        sol_route_topk_ratio=ratio,
        sol_log_density=False,
        landmark_tree_v2_midpoint_direction_mode="legacy",
        sol_route_topk_execution="threshold",
    )


def reference_latent(method: str, case: dict) -> Path:
    index = case["index"]
    stem = f"{index:02d}_{case['sample_id']}"
    if method in ("dense", "sol"):
        return SOURCE / method / "latents" / f"{stem}.pt"
    if method in ("spark10", "spark20"):
        ratio = 10 if method == "spark10" else 20
        return SPARK12 / "latents" / f"topk{ratio}_reblock_global_reweight_{index:02d}.pt"
    return SPARK30 / "spark_topk30" / "latents" / f"{stem}.pt"


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(
            SOURCE / name, target_is_directory=name.endswith("cache")
        )
    selected = cases()
    assert selected[0]["index"] in REMAPPED_CASES
    assert selected[1]["index"] not in REMAPPED_CASES
    base.pipeline().configure_denoise_workflow(args(), selected)
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
    diff = subprocess.check_output(["git", "-C", str(base.IMPL), "diff", "--binary"])
    configs = {
        method: dataclasses.asdict(attention_config(method, STEPS))
        for method in METHODS if method != "dense"
    }
    protocol = {
        "status": "running",
        "name": NAME,
        "purpose": "Table-3 timing and persisted latent consistency on one remapped and one non-remapped prompt",
        "cases": [{**case, "provenance_group": CASE_GROUPS[case["index"]]} for case in selected],
        "remapped_cases": list(REMAPPED_CASES),
        "gpus": list(GPUS),
        "methods": list(METHODS),
        "method_order": {"gpu0": list(METHODS), "gpu1": list(reversed(METHODS))},
        "warmup": {
            "exactly_once_per_gpu_per_method": True,
            "dense": {"steps": 2, "evaluations": 1},
            "sol": {"steps": 20, "evaluations": 19},
            "spark_variants": {"steps": 3, "evaluations": 2, "dense": 1, "sparse": 1},
            "excluded_from_timing": True,
        },
        "formal": {"steps": STEPS, "evaluations": 19, "seed": SEED,
                   "frames": FRAMES, "resolution": [WIDTH, HEIGHT]},
        "configs": configs,
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


def run_once(pipe, pipeline, original_state, method: str, *, steps: int, save_path: Path | None):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    state = pipeline.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    if method == "dense":
        context = contextlib.nullcontext(None)
    else:
        context = install_h3_sparse_attention(pipe.transformer, attention_config(method, steps))
    with context as plugin, torch.inference_mode():
        if plugin is not None:
            plugin.reset()
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
        summary = None if plugin is None else plugin.summary()
    expected_evaluations = steps - 1
    if summary is not None:
        assert summary["completed_evaluations"] == expected_evaluations, summary
        if method.startswith("spark"):
            assert summary["dense_evaluations"] == (4 if steps == STEPS else 1), summary
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
    return seconds, summary, payload


def worker(rank: int):
    import torch

    rank = int(rank)
    gpu = GPUS[rank]
    torch.set_num_threads(4)
    pipeline = base.pipeline()
    selected = cases()
    workflow, states = pipeline.configure_denoise_workflow(args(), selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args(), workflow)
    method_order = METHODS if rank == 0 else tuple(reversed(METHODS))
    case = selected[rank]
    try:
        for order, method in enumerate(method_order):
            warm_steps = 2 if method == "dense" else STEPS if method == "sol" else 3
            warm_seconds, warm_summary, _ = run_once(
                pipe, pipeline, states[rank], method, steps=warm_steps, save_path=None
            )
            write(ROOT / "records" / f"gpu{gpu}_{method}_warmup.json", {
                "phase": "discarded_warmup", "gpu": gpu, "order": order,
                "method": method, "case": case["index"], "group": CASE_GROUPS[case["index"]],
                "steps": warm_steps, "evaluations": warm_steps - 1,
                "seconds": warm_seconds, "attention_summary": warm_summary,
            })
            print("WARMUP", method, round(warm_seconds, 3), flush=True)
            latent_path = OUT / method / "latents" / f"{case['index']:02d}_{case['sample_id']}.pt"
            seconds, summary, payload = run_once(
                pipe, pipeline, states[rank], method, steps=STEPS, save_path=latent_path
            )
            reference = reference_latent(method, case)
            comparison = compare_payload(payload, reference)
            record = {
                "phase": "measured", "gpu": gpu, "order": order,
                "method": method, "case": case["index"], "sample_id": case["sample_id"],
                "group": CASE_GROUPS[case["index"]], "prompt_sha256": case["prompt_sha256"],
                "placement": placement, "steps": STEPS, "evaluations": STEPS - 1,
                "seconds": seconds, "attention_summary": summary,
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
        pipeline.release_cpu_arenas()


def summarize():
    rows = [json.loads(path.read_text()) for path in (ROOT / "records").glob("*_measured.json")]
    result = {"status": "complete", "methods": {}, "content": {}}
    for method in METHODS:
        chosen = sorted((row for row in rows if row["method"] == method), key=lambda row: row["gpu"])
        if len(chosen) != 2:
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
            [str(base.PYTHON), str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
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
