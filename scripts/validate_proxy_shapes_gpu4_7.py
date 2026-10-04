#!/usr/bin/env python3
"""Paired proxy-shape validation on physical GPUs 4--7.

The pilot compares 5s/768p and 10s/480p with the 10s/768p target across
several dense/Sol/Spark arms.  It records denoiser latency and latent-space
distance from a setting-matched dense reference.  Each GPU owns one prompt,
loads the model once, and runs every setting/arm pair for paired comparisons.

The orchestrator intentionally points H3_IMPL_REPO at a detached clean HEAD
worktree.  This keeps the result independent of unrelated dirty-worktree
changes and makes the exact source revision auditable in protocol.json.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


NAME = "proxy_shape_5s768_vs_10s480_target_10s768_gpu4_7_20261002"
REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
BENCH_SCRIPTS = BENCH / "scripts"
CLEAN_IMPL = REPO.parent / "Spark-H3-proxy-head"
MODEL = Path("/mnt/CFS/tangzecheng/models/MiniMax-H3")
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
LEGACY_CONDITIONING = Path("/mnt/CFS/tangzecheng/calibration_f16_tau_20260919")
ROOT = Path("/mnt/CFS/tangzecheng/experiments") / NAME
GPUS = (4, 5, 6, 7)
CASES = (1, 2, 3, 4)
SEED = 42
STEPS = 20
WARMUP_STEPS = 6  # four dense evaluations plus one sparse evaluation
SETTINGS = {
    "5s768p": {"frames": 120, "height": 768, "width": 1344},
    "10s480p": {"frames": 240, "height": 480, "width": 832},
    "10s768p": {"frames": 240, "height": 768, "width": 1344},
}
ARMS = (
    "dense",
    "sol",
    "spark10",
    "spark20",
    "spark30",
    "spark10_no_reweight",
    "spark10_fp32",
)
RANK_ARMS = tuple(arm for arm in ARMS if arm != "dense")
PROXIES = ("5s768p", "10s480p")
TARGET = "10s768p"


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
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


def command_output(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def import_pipeline():
    if str(BENCH_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BENCH_SCRIPTS))
    import _impl_bootstrap

    resolved = Path(_impl_bootstrap.IMPL_REPO).resolve()
    if resolved != CLEAN_IMPL.resolve():
        raise RuntimeError(f"implementation mismatch: {resolved} != {CLEAN_IMPL}")
    import minimax_h3_vbench_4gpu_pipeline as pipeline

    return pipeline


def cases():
    p = import_pipeline()
    return p.load_cases(SAMPLES, list(CASES), expected_indices=tuple(range(1, 26)))


def arm_config(arm: str):
    p = import_pipeline()
    if arm == "sol":
        return p.H3SparseAttentionConfig.sol(STEPS)
    overrides = {}
    if arm == "spark10":
        pass
    elif arm == "spark20":
        overrides["sol_route_topk_ratio"] = 0.20
    elif arm == "spark30":
        overrides["sol_route_topk_ratio"] = 0.30
    elif arm == "spark10_no_reweight":
        overrides["sol_reweight_components"] = "none"
    elif arm == "spark10_fp32":
        overrides["sol_global_anchor_dtype"] = "float32"
    else:
        raise ValueError(arm)
    return p.H3SparseAttentionConfig.spark(STEPS, **overrides)


def migrate_conditioning() -> None:
    """Convert trusted local schema-v1 PipelineState caches to schema v2."""
    import torch

    p = import_pipeline()
    import _h3_inference_common as common

    selected = cases()
    legacy_manifest = read(LEGACY_CONDITIONING / "conditioning_manifest.json")
    by_index = {int(row["index"]): row for row in legacy_manifest["records"]}
    target_dir = ROOT / "conditioning_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = common._conditioning_model_fingerprint(MODEL)
    records = []
    for case in selected:
        old_meta = by_index[case["index"]]
        if (
            old_meta["sample_id"] != case["sample_id"]
            or old_meta["prompt_sha256"] != case["prompt_sha256"]
        ):
            raise RuntimeError(f"legacy conditioning mismatch for case {case['index']}")
        source = LEGACY_CONDITIONING / "conditioning_cache" / f"{case['index']:02d}.pt"
        state = torch.load(source, map_location="cpu", weights_only=False)
        target = common._conditioning_cache_path(target_dir, fingerprint, case["prompt"])
        if not target.exists():
            common._save_conditioning_state(
                state, target, model_fingerprint=fingerprint, prompt=case["prompt"]
            )
        loaded = common._load_conditioning_state(
            target, model_fingerprint=fingerprint, prompt=case["prompt"]
        )
        summary = common._conditioning_values_summary(
            loaded.values, prompt=case["prompt"], source=str(target)
        )
        records.append(
            {
                "index": case["index"],
                "sample_id": case["sample_id"],
                "prompt_sha256": case["prompt_sha256"],
                "source": str(source),
                "path": str(target),
                "sha256": sha256(target),
                "summary": summary,
            }
        )
    write(
        ROOT / "conditioning_manifest.json",
        {"status": "complete", "schema": 2, "model": str(MODEL), "records": records},
    )


def prepare() -> None:
    if not CLEAN_IMPL.is_dir():
        raise FileNotFoundError(CLEAN_IMPL)
    ROOT.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        for arm in ARMS:
            (ROOT / "records" / setting / arm).mkdir(parents=True, exist_ok=True)
            (ROOT / "latents" / setting / arm).mkdir(parents=True, exist_ok=True)
            (ROOT / "warmup" / setting / arm).mkdir(parents=True, exist_ok=True)
    migrate_conditioning()
    dirty_diff = command_output("git", "diff", "--binary", cwd=REPO)
    write(
        ROOT / "protocol.json",
        {
            "name": NAME,
            "status": "prepared",
            "purpose": (
                "Compare 5s768p and 10s480p as paired performance/latent-fidelity "
                "proxies for 10s768p across representative Sol/Spark ablations"
            ),
            "physical_gpus": list(GPUS),
            "gpu_name": command_output(
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
            ).splitlines()[GPUS[0]],
            "settings": SETTINGS,
            "arms": list(ARMS),
            "rank_arms": list(RANK_ARMS),
            "cases": cases(),
            "seed": SEED,
            "requested_steps": STEPS,
            "expected_transformer_evaluations": STEPS - 1,
            "warmup_steps": WARMUP_STEPS,
            "allocator": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "timing": "CUDA-synchronized denoiser call; model load, warmup, CPU copy/save excluded",
            "pairing": "one prompt per physical GPU; every prompt runs every setting and arm",
            "model": str(MODEL),
            "samples": str(SAMPLES),
            "implementation": {
                "path": str(CLEAN_IMPL),
                "revision": command_output("git", "rev-parse", "HEAD", cwd=CLEAN_IMPL),
                "status": command_output("git", "status", "--porcelain", cwd=CLEAN_IMPL),
                "reason": (
                    "detached clean HEAD used because the primary worktree's in-progress "
                    "SM80 int32 permutation change failed its CUDA parity test"
                ),
            },
            "primary_worktree": {
                "path": str(REPO),
                "revision": command_output("git", "rev-parse", "HEAD", cwd=REPO),
                "dirty_diff_sha256": hashlib.sha256(dirty_diff.encode()).hexdigest(),
            },
            "quality_scope": (
                "latent PSNR/relative-L2 against a setting-matched dense reference; "
                "decoded perceptual metrics are outside this pilot"
            ),
        },
    )


def target_paths(setting: str, arm: str, case: dict):
    stem = f"{case['index']:02d}_{case['sample_id']}"
    return (
        ROOT / "latents" / setting / arm / f"{stem}.pt",
        ROOT / "records" / setting / arm / f"{stem}.json",
        ROOT / "warmup" / setting / arm / f"{stem}.json",
    )


def save_latent(path: Path, payload) -> None:
    import torch

    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_generation(pipe, plugin, state, setting, steps: int):
    import torch

    p = import_pipeline()
    plugin.reset()
    inference_state = p.clone_state(state)
    inference_state.values["prompt_embeds"] = inference_state.values[
        "prompt_embeds"
    ].to("cuda")
    shape = SETTINGS[setting]
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = pipe(
        state=inference_state,
        num_frames=shape["frames"],
        height=shape["height"],
        width=shape["width"],
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        output=["latents", "audio_latents"],
    )
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def worker(gpu: int) -> None:
    import torch

    p = import_pipeline()
    selected = cases()
    rank = GPUS.index(gpu)
    case = selected[rank]
    args = SimpleNamespace(
        model=MODEL,
        output=ROOT,
        transformer_placement="resident",
        transformer_offload_dir=ROOT / "offload" / f"gpu{gpu}",
        resident_blocks=50,
    )
    workflow, states = p.configure_denoise_workflow(args, [case])
    pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    state = states[0]
    settings = list(SETTINGS)
    settings = settings[rank % len(settings) :] + settings[: rank % len(settings)]
    try:
        for setting_index, setting in enumerate(settings):
            arms = list(ARMS)
            shift = (rank + setting_index) % len(arms)
            arms = arms[shift:] + arms[:shift]
            for arm in arms:
                latent_path, record_path, warmup_path = target_paths(setting, arm, case)
                if latent_path.is_file() and record_path.is_file():
                    print(f"REUSE gpu={gpu} {setting} {arm} case={case['index']}", flush=True)
                    continue
                config = None if arm == "dense" else arm_config(arm)
                context = (
                    p.DensePlugin()
                    if config is None
                    else p.install_h3_sparse_attention(pipe.transformer, config)
                )
                with context as plugin:
                    # Warm every incomplete arm in the current process.  A
                    # persisted warmup marker cannot warm a fresh compiler
                    # cache after a worker restart.
                    torch.cuda.empty_cache()
                    warm, warm_seconds = run_generation(
                        pipe, plugin, state, setting, WARMUP_STEPS
                    )
                    del warm
                    write(
                        warmup_path,
                        {
                            "status": "complete",
                            "physical_gpu": gpu,
                            "case": case["index"],
                            "setting": setting,
                            "arm": arm,
                            "steps": WARMUP_STEPS,
                            "seconds": warm_seconds,
                        },
                    )
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    result, seconds = run_generation(pipe, plugin, state, setting, STEPS)
                    summary = plugin.summary()
                    if arm != "dense" and summary.get("completed_evaluations") != STEPS - 1:
                        raise RuntimeError(f"incomplete evaluations: {summary}")
                    payload = {
                        "metadata": {
                            "setting": setting,
                            "arm": arm,
                            "case": case["index"],
                            "sample_id": case["sample_id"],
                            "prompt_sha256": case["prompt_sha256"],
                            "seed": SEED,
                            "steps": STEPS,
                            **SETTINGS[setting],
                        },
                        "latents": result["latents"].detach().cpu(),
                        "audio_latents": result["audio_latents"].detach().cpu(),
                    }
                    del result
                    save_latent(latent_path, payload)
                    record = {
                        **payload["metadata"],
                        "status": "complete",
                        "physical_gpu": gpu,
                        "denoise_seconds": seconds,
                        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "latent_path": str(latent_path),
                        "latent_sha256": sha256(latent_path),
                        "transformer_placement": placement,
                        "torch_compile": True,
                        "compile_wrapped_blocks": len(
                            getattr(acceleration, "_forward_originals", ())
                        ),
                        "attention_config": None if config is None else dataclasses.asdict(config),
                        "attention_summary": summary,
                    }
                    write(record_path, record)
                    print(
                        f"DONE gpu={gpu} {setting} {arm} case={case['index']} "
                        f"seconds={seconds:.3f}",
                        flush=True,
                    )
    finally:
        acceleration.remove()
        del pipe, manager


def latent_metrics(candidate: Path, reference: Path):
    import torch

    a = torch.load(candidate, map_location="cpu", weights_only=True)["latents"].float()
    b = torch.load(reference, map_location="cpu", weights_only=True)["latents"].float()
    if a.shape != b.shape:
        raise ValueError(f"latent shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    mse = float(diff.square().mean())
    ref_power = float(b.square().mean())
    dynamic_range = float(b.max() - b.min())
    rel_l2 = float(diff.norm() / b.norm().clamp_min(1e-12))
    cosine = float(
        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    )
    psnr = float("inf") if mse == 0 else 20 * math.log10(dynamic_range / math.sqrt(mse))
    return {
        "mse": mse,
        "reference_power": ref_power,
        "relative_l2": rel_l2,
        "cosine": cosine,
        "latent_psnr_db": psnr,
        "shape": list(a.shape),
    }


def rank_correlation(values_a: dict[str, float], values_b: dict[str, float]):
    arms = [arm for arm in RANK_ARMS if arm in values_a and arm in values_b]
    a = [values_a[arm] for arm in arms]
    b = [values_b[arm] for arm in arms]

    def average_ranks(values):
        ranks = [0.0] * len(values)
        ordered = sorted(range(len(values)), key=values.__getitem__)
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
                end += 1
            rank = (start + 1 + end) / 2
            for index in ordered[start:end]:
                ranks[index] = rank
            start = end
        return ranks

    def pearson(x, y):
        x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
        x_centered = [value - x_mean for value in x]
        y_centered = [value - y_mean for value in y]
        numerator = sum(xv * yv for xv, yv in zip(x_centered, y_centered))
        denominator = math.sqrt(
            sum(value * value for value in x_centered)
            * sum(value * value for value in y_centered)
        )
        return numerator / denominator if denominator else float("nan")

    rho = pearson(average_ranks(a), average_ranks(b))
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_a)
        * (concordant + discordant + ties_b)
    )
    tau = (concordant - discordant) / denominator if denominator else float("nan")

    # Pairwise concordance deliberately ignores ties in either ordering.
    concordant = 0
    total = 0
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            total += 1
            concordant += (da > 0) == (db > 0)
    return {
        "arms": arms,
        "spearman_rho": rho,
        "kendall_tau": tau,
        "pairwise_concordance": concordant / total if total else None,
        "pairwise_comparisons": total,
    }


def bootstrap_correlations(per_case, metric: str, repeats: int = 2000):
    rng = random.Random(20261002)
    indices = sorted(per_case)
    result = {proxy: [] for proxy in PROXIES}
    for _ in range(repeats):
        sampled = [rng.choice(indices) for _ in indices]
        means = {}
        for setting in SETTINGS:
            means[setting] = {
                arm: statistics.fmean(
                    per_case[index][setting][arm][metric] for index in sampled
                )
                for arm in RANK_ARMS
            }
        for proxy in PROXIES:
            rho = rank_correlation(means[proxy], means[TARGET])["spearman_rho"]
            if math.isfinite(rho):
                result[proxy].append(rho)
    summarized = {}
    for proxy, values in result.items():
        values.sort()
        summarized[proxy] = {
            "samples": len(values),
            "mean": statistics.fmean(values),
            "p2p5": values[int(0.025 * (len(values) - 1))],
            "p50": values[int(0.5 * (len(values) - 1))],
            "p97p5": values[int(0.975 * (len(values) - 1))],
        }
    return summarized


def summarize() -> None:
    selected = cases()
    records = []
    per_case = {}
    for case in selected:
        per_case[case["index"]] = {}
        for setting in SETTINGS:
            per_case[case["index"]][setting] = {}
            dense_latent, _, _ = target_paths(setting, "dense", case)
            for arm in ARMS:
                latent, record_path, _ = target_paths(setting, arm, case)
                if not latent.is_file() or not record_path.is_file():
                    raise FileNotFoundError(f"incomplete: {setting}/{arm}/case{case['index']}")
                record = read(record_path)
                metrics = (
                    {
                        "mse": 0.0,
                        "reference_power": None,
                        "relative_l2": 0.0,
                        "cosine": 1.0,
                        "latent_psnr_db": None,
                        "shape": list(
                            __import__("torch").load(
                                latent, map_location="cpu", weights_only=True
                            )["latents"].shape
                        ),
                    }
                    if arm == "dense"
                    else latent_metrics(latent, dense_latent)
                )
                row = {**record, "latent_metrics_vs_dense": metrics}
                records.append(row)
                per_case[case["index"]][setting][arm] = {
                    "seconds": float(record["denoise_seconds"]),
                    "log_speedup_vs_sol": 0.0,
                    "latent_psnr_db": metrics["latent_psnr_db"],
                    "negative_relative_l2": -metrics["relative_l2"],
                }
            sol_seconds = per_case[case["index"]][setting]["sol"]["seconds"]
            for arm in RANK_ARMS:
                seconds = per_case[case["index"]][setting][arm]["seconds"]
                per_case[case["index"]][setting][arm]["log_speedup_vs_sol"] = math.log(
                    sol_seconds / seconds
                )

    means = {}
    for setting in SETTINGS:
        means[setting] = {}
        for arm in ARMS:
            rows = [r for r in records if r["setting"] == setting and r["arm"] == arm]
            means[setting][arm] = {
                "mean_seconds": statistics.fmean(r["denoise_seconds"] for r in rows),
                "median_seconds": statistics.median(r["denoise_seconds"] for r in rows),
                "mean_peak_cuda_allocated_gib": statistics.fmean(
                    r["peak_cuda_allocated_gib"] for r in rows
                ),
                "mean_latent_psnr_db": None
                if arm == "dense"
                else statistics.fmean(
                    r["latent_metrics_vs_dense"]["latent_psnr_db"] for r in rows
                ),
                "mean_relative_l2": statistics.fmean(
                    r["latent_metrics_vs_dense"]["relative_l2"] for r in rows
                ),
            }
        sol_seconds = means[setting]["sol"]["mean_seconds"]
        dense_seconds = means[setting]["dense"]["mean_seconds"]
        for arm in ARMS:
            seconds = means[setting][arm]["mean_seconds"]
            means[setting][arm]["speedup_vs_sol"] = sol_seconds / seconds
            means[setting][arm]["speedup_vs_dense"] = dense_seconds / seconds

    comparisons = {}
    for metric in ("log_speedup_vs_sol", "latent_psnr_db", "negative_relative_l2"):
        comparisons[metric] = {}
        target_values = {
            arm: statistics.fmean(
                per_case[index][TARGET][arm][metric] for index in per_case
            )
            for arm in RANK_ARMS
        }
        for proxy in PROXIES:
            proxy_values = {
                arm: statistics.fmean(
                    per_case[index][proxy][arm][metric] for index in per_case
                )
                for arm in RANK_ARMS
            }
            comparisons[metric][proxy] = {
                **rank_correlation(proxy_values, target_values),
                "proxy_values": proxy_values,
                "target_values": target_values,
            }
        comparisons[metric]["bootstrap"] = bootstrap_correlations(per_case, metric)

    result = {
        "status": "complete",
        "sample_count": len(selected),
        "settings": SETTINGS,
        "arms": list(ARMS),
        "means": means,
        "proxy_comparisons": comparisons,
        "per_case": per_case,
        "records": records,
    }
    write(ROOT / "results.json", result)
    protocol = read(ROOT / "protocol.json")
    protocol["status"] = "complete"
    write(ROOT / "protocol.json", protocol)
    # Keep manual/recovery summarization consistent with the full orchestrator.
    write(ROOT / "status.json", {"status": "complete"})

    lines = [
        "# Proxy-shape validation: 5s768p vs 10s480p for 10s768p",
        "",
        f"Four paired prompts, seed {SEED}, {STEPS} requested steps, physical GPUs 4--7.",
        "Timing is synchronized denoiser latency. Quality is latent-space fidelity to a setting-matched dense reference.",
        "",
        "## Mean results",
        "",
        "| Setting | Arm | Seconds | Speedup vs Sol | Latent PSNR vs dense | Rel-L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for setting in SETTINGS:
        for arm in ARMS:
            row = means[setting][arm]
            psnr = "—" if row["mean_latent_psnr_db"] is None else f"{row['mean_latent_psnr_db']:.3f}"
            lines.append(
                f"| {setting} | {arm} | {row['mean_seconds']:.3f} | "
                f"{row['speedup_vs_sol']:.4f}× | {psnr} | {row['mean_relative_l2']:.5f} |"
            )
    lines += ["", "## Proxy rank agreement with 10s768p", ""]
    for metric in ("log_speedup_vs_sol", "latent_psnr_db", "negative_relative_l2"):
        lines += [
            f"### {metric}",
            "",
            "| Proxy | Spearman | Kendall | Pairwise agreement | Bootstrap rho median [2.5%, 97.5%] |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for proxy in PROXIES:
            row = comparisons[metric][proxy]
            bootstrap = comparisons[metric]["bootstrap"][proxy]
            lines.append(
                f"| {proxy} | {row['spearman_rho']:.4f} | {row['kendall_tau']:.4f} | "
                f"{100 * row['pairwise_concordance']:.1f}% | {bootstrap['p50']:.4f} "
                f"[{bootstrap['p2p5']:.4f}, {bootstrap['p97p5']:.4f}] |"
            )
        lines.append("")
    spark10_5s = means["5s768p"]["spark10"]["mean_seconds"]
    spark10_480 = means["10s480p"]["spark10"]["mean_seconds"]
    spark10_target = means[TARGET]["spark10"]["mean_seconds"]
    lines += [
        "## Recommendation",
        "",
        "Use **5s768p as the default mixed speed/quality ablation proxy**. Both proxies "
        "match the target speed ordering equally well, while 5s768p has stronger and more "
        "stable latent-fidelity ordering, especially for relative-L2.",
        "",
        "Use **10s480p for speed-only sweeps** when throughput matters most: for Spark10 it "
        f"takes {spark10_480:.3f}s versus {spark10_5s:.3f}s at 5s768p and "
        f"{spark10_target:.3f}s at 10s768p. Thus 10s480p is "
        f"{spark10_5s / spark10_480:.2f}x faster than 5s768p, while 5s768p is "
        f"{spark10_target / spark10_5s:.2f}x faster than the target.",
        "",
        "This is a four-prompt, one-seed screening validation. Confirm finalists at "
        "10s768p, particularly when differences are as small as the Spark10, no-reweight, "
        "and fp32 variants.",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


def orchestrate() -> None:
    prepare()
    env_base = {
        **os.environ,
        "H3_IMPL_REPO": str(CLEAN_IMPL),
        "H3_DIFFUSERS_DIR": str(MODEL),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        # The 10s768p Spark path peaks within roughly 1 GiB of A800 capacity.
        # Expandable segments let PyTorch reuse the otherwise stranded reserve.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    processes = []
    for gpu in GPUS:
        log_path = ROOT / f"worker_gpu{gpu}.log"
        log = log_path.open("a")
        env = {**env_base, "CUDA_VISIBLE_DEVICES": str(gpu)}
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", "--gpu", str(gpu)],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log))
    failures = []
    for gpu, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failures.append((gpu, code))
    if failures:
        write(ROOT / "status.json", {"status": "failed", "workers": failures})
        raise RuntimeError(f"worker failures: {failures}")
    summarize()
    write(ROOT / "status.json", {"status": "complete"})


def status() -> None:
    rows = []
    for gpu in GPUS:
        log = ROOT / f"worker_gpu{gpu}.log"
        done = []
        if log.exists():
            done = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("DONE ")]
        rows.append({"gpu": gpu, "completed": len(done), "last": done[-1] if done else None})
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "summarize", "run", "status"))
    parser.add_argument("--gpu", type=int, choices=GPUS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        if args.gpu is None:
            parser.error("worker requires --gpu")
        worker(args.gpu)
    elif args.command == "summarize":
        summarize()
    elif args.command == "run":
        orchestrate()
    else:
        status()


if __name__ == "__main__":
    main()
