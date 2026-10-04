#!/usr/bin/env python3
"""Blog two-prompt DMAD/PDD x Dense/Spark-warm/Spark-nowarm matrix."""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
DMAD_REPO = REPO.parent / "DMAD"
DMAD_COMMIT = "a637fb080f48b516bb00d50140468b429140bb26"
sys.path.insert(0, str(DMAD_REPO))

NAME = "blog2_dmad_pdd_spark_matrix_20261003"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
MODEL = Path("/root/h3_local/h3_diffusers")
MODELS = Path("/autodl-fs/data/models")
DMAD_WEIGHT = MODELS / "ZhengmingYu-DMAD/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors"
PDD_WEIGHT = MODELS / "alibaba-pai-MiniMax-H3-Acc-LoRAs/MiniMax-H3-FL2VA-Acc-8Step.safetensors"
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
SAMPLES = BENCH / "vbench_core5_percent_subsets/blog_example_2prompts/samples.json"
CONDITIONING = Path(
    "/autodl-fs/data/h3_outputs/fasth3_vbench50_vsa_dense_20260915_082643/conditioning_cache_manifest.json"
)
FV_EVAL = REPO.parent / "FastVideo/examples/inference/eval"
LPIPS_PACKAGE = Path("/autodl-fs/data/h3_experiments/diag_fullcov_budget_20260912_v2/lpips_python")
LPIPS_TORCH_HOME = Path("/autodl-fs/data/h3_experiments/diag_fullcov_budget_20260912_v2/lpips_torch_cache")
GPUS = (0, 1, 2, 3)
MODELS_UNDER_TEST = ("dmad", "pdd")
VARIANTS = ("dense", "spark10_warm", "spark10_nowarm")
PROMPTS = ("0685", "0753")
JOBS = (("dmad", "0685"), ("dmad", "0753"), ("pdd", "0685"), ("pdd", "0753"))
HEIGHT, WIDTH, REQUESTED_FRAMES, ALIGNED_FRAMES, FPS, SEED = 768, 1344, 240, 243, 24, 42
ENV = {
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
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def sample_cases():
    return {
        row["sample_id"].split("_")[-1]: row
        for row in json.loads(SAMPLES.read_text())
    }


def load_condition(prompt_id):
    import torch

    case = sample_cases()[prompt_id]
    manifest = json.loads(CONDITIONING.read_text())
    key = hashlib.sha256(case["generation_prompt"].encode()).hexdigest()
    entry = manifest["entries"][key]
    payload = torch.load(entry["path"], map_location="cpu", weights_only=True)
    assert payload["prompt"] == case["generation_prompt"]
    assert torch.isfinite(payload["values"]["prompt_embeds"]).all()
    return case, payload["values"], Path(entry["path"])


def sparse_config(model, variant):
    from h3_sparse_attention import H3SparseAttentionConfig

    evaluations = 4 if model == "dmad" else 8
    grid_points = evaluations + 1
    warm = variant == "spark10_warm"
    cfg = H3SparseAttentionConfig.spark(
        grid_points,
        warmup_percent=20.0 if warm else 0.0,
        sol_dense_layers=1 if warm else 0,
        sol_route_topk_ratio=0.10,
        sol_log_density=False,
    )
    assert cfg.total_evaluations == evaluations
    assert cfg.dense_evaluations == (1 if model == "dmad" and warm else 2 if warm else 0)
    return cfg


@contextlib.contextmanager
def weight_load_lock():
    path = Path("/tmp/minimax_h3_blog2_weight_load.lock")
    with path.open("w") as handle:
        print("waiting for weight-load lock", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print("acquired weight-load lock", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fuse_pairs(transformer, pairs, scale):
    """Fuse one Diffusers-format LoRA exactly once using FP32 accumulation."""
    import torch

    plan = []
    for name, pair in sorted(pairs.items()):
        module = transformer.get_submodule(name)
        a, b = pair["A"], pair["B"]
        assert tuple(module.weight.shape) == (b.shape[0], a.shape[1]), name
        plan.append((name, module, a, b))
    for index, (name, module, a, b) in enumerate(plan):
        merged = module.weight.detach().to("cuda", torch.float32, copy=True)
        merged.addmm_(b.to("cuda", torch.float32), a.to("cuda", torch.float32), alpha=scale)
        module.weight.data = merged.to("cpu", module.weight.dtype)
        del merged
        if index % 50 == 0:
            print("fused", index, "/", len(plan), name, flush=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"target_weights": len(plan), "scale": scale, "arithmetic": "FP32 B@A + base"}


def fuse_pdd_wrappers(transformer, pdd_module):
    """Fuse PDD's trunk LoRA wrappers while retaining its parallel output heads."""
    import torch

    targets = [
        (name, module) for name, module in transformer.named_modules()
        if isinstance(module, pdd_module.LoRALinear)
    ]
    for index, (name, wrapper) in enumerate(targets):
        base = wrapper.base
        merged = base.weight.detach().to("cuda", torch.float32, copy=True)
        merged.addmm_(
            wrapper.lora_up.detach().to("cuda", torch.float32),
            wrapper.lora_down.detach().to("cuda", torch.float32),
            alpha=wrapper.scaling,
        )
        base.weight.data = merged.to("cpu", base.weight.dtype)
        parent_name, _, attribute = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        setattr(parent, attribute, base)
        del merged
        if index % 50 == 0:
            print("fused PDD trunk", index, "/", len(targets), name, flush=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"target_weights": len(targets), "arithmetic": "FP32 B@A + base"}


def load_student(model):
    import torch
    from dmad_h3 import load_transformer, read_lora_file
    from dmad_h3.lora import lora_rank_alpha

    transformer = load_transformer(MODEL, torch.bfloat16)
    if model == "dmad":
        pairs, metadata = read_lora_file(DMAD_WEIGHT)
        rank, alpha = lora_rank_alpha(pairs, metadata)
        fusion = fuse_pairs(transformer, pairs, alpha / rank)
        nfe, video_shift, audio_shift, renoise = 4, 12.0, 2.0, True
    else:
        sys.path.insert(0, str(REPO / "scripts"))
        import minimax_h3_pdd as pdd

        nfe = pdd.apply_pdd_lora(transformer, str(PDD_WEIGHT), 12.0, 3.0)
        assert nfe == 8
        fusion = fuse_pdd_wrappers(transformer, pdd)
        video_shift, audio_shift, renoise = 12.0, 3.0, False
    transformer.requires_grad_(False).eval().to("cuda")
    torch.cuda.synchronize()
    return transformer, fusion, nfe, video_shift, audio_shift, renoise


def run_once(transformer, values, layout, shape, geometry, *, model, variant, seed):
    import torch
    from dmad_h3 import rollout, sigma_grid
    from h3_sparse_attention import install_h3_sparse_attention
    from h3_sparse_attention.sol_numerator_virtual_q import fused_compile_cache_stats

    nfe = 4 if model == "dmad" else 8
    video_shift = 12.0
    audio_shift = 2.0 if model == "dmad" else 3.0
    renoise = model == "dmad"
    config = None if variant == "dense" else sparse_config(model, variant)
    context = (
        contextlib.nullcontext(None)
        if config is None else install_h3_sparse_attention(transformer, config)
    )
    before = fused_compile_cache_stats()
    torch.cuda.reset_peak_memory_stats()
    with context as plugin, torch.inference_mode():
        if plugin is not None:
            plugin.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        video_rows, audio_rows = rollout(
            transformer,
            values["prompt_embeds"].to("cuda", torch.bfloat16),
            layout,
            shape,
            geometry,
            sigma_grid(nfe, video_shift),
            sigma_grid(nfe, audio_shift),
            seed,
            torch.device("cuda"),
            renoise=renoise,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = None if plugin is None else plugin.summary()
    after = fused_compile_cache_stats()
    assert torch.isfinite(video_rows).all() and torch.isfinite(audio_rows).all()
    if summary is not None:
        assert summary["completed_evaluations"] == nfe, summary
        expected_dense = 1 if model == "dmad" and variant == "spark10_warm" else 2 if variant == "spark10_warm" else 0
        assert summary["dense_evaluations"] == expected_dense, summary
    return video_rows, audio_rows, {
        "seconds": seconds,
        "attention_summary": summary,
        "peak_denoise_mb": torch.cuda.max_memory_allocated() / 2**20,
        "compile_before": before,
        "compile_after": after,
        "new_compile_calls": after["compile_calls"] - before["compile_calls"],
        "new_compile_seconds": after["compile_seconds"] - before["compile_seconds"],
    }


def inspect_archive(path):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True))
    videos = [s for s in probe["streams"] if s["codec_type"] == "video"]
    audios = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    assert len(videos) == 1 and audios
    assert int(videos[0]["width"]) == WIDTH and int(videos[0]["height"]) == HEIGHT
    frame_count = videos[0].get("nb_read_frames") or videos[0].get("nb_frames")
    assert frame_count not in (None, "N/A")
    assert int(frame_count) == REQUESTED_FRAMES
    return {"frames": REQUESTED_FRAMES, "width": WIDTH, "height": HEIGHT,
            "audio_streams": len(audios), "duration": float(probe["format"]["duration"])}


def worker(rank):
    import torch
    from dmad_h3 import H3Geometry, build_packed_sequence, decode_rows, latent_shape_of
    from h3_sparse_attention import H3AccelerationConfig, install_h3_acceleration

    rank = int(rank)
    gpu = GPUS[rank]
    model, prompt_id = JOBS[rank]
    torch.set_num_threads(4)
    case, values, cache_path = load_condition(prompt_id)
    geometry = H3Geometry.from_model_dir(MODEL)
    shape = latent_shape_of(HEIGHT, WIDTH, ALIGNED_FRAMES, geometry)
    layout = build_packed_sequence(
        values["text_token_tags"], shape["latent_frames"], shape["latent_height"],
        shape["latent_width"], shape["audio_latents"], geometry.patch_size,
    ).to("cuda")
    status_path = ROOT / f"worker_gpu{gpu}.json"
    try:
        write(status_path, {"status": "loading", "model": model, "prompt": prompt_id})
        with weight_load_lock():
            transformer, fusion, nfe, video_shift, audio_shift, renoise = load_student(model)
        acceleration = install_h3_acceleration(
            SimpleNamespace(transformer=transformer),
            H3AccelerationConfig(torch_compile=True, vae_fp16=False),
        )
        acceleration.__enter__()
        payloads = {}
        order = VARIANTS if rank % 2 == 0 else tuple(reversed(VARIANTS))
        try:
            for variant in order:
                write(status_path, {"status": "warmup", "model": model,
                                   "prompt": prompt_id, "variant": variant})
                warm_video, warm_audio, warm = run_once(
                    transformer, values, layout, shape, geometry,
                    model=model, variant=variant, seed=41,
                )
                del warm_video, warm_audio
                write(ROOT / "records" / f"{model}_{prompt_id}_{variant}_warmup.json", {
                    "phase": "discarded_warmup", "gpu": gpu, "model": model,
                    "prompt": prompt_id, "variant": variant, **warm,
                })
                write(status_path, {"status": "measured", "model": model,
                                   "prompt": prompt_id, "variant": variant})
                video_rows, audio_rows, measured = run_once(
                    transformer, values, layout, shape, geometry,
                    model=model, variant=variant, seed=SEED,
                )
                payloads[variant] = (video_rows, audio_rows)
                record = {
                    "phase": "measured", "gpu": gpu, "model": model,
                    "prompt": prompt_id, "sample_id": case["sample_id"],
                    "variant": variant, "seed": SEED, "nfe": nfe,
                    "video_shift": video_shift, "audio_shift": audio_shift,
                    "sampling_rule": "DMAD re-noise" if renoise else "PDD block Euler",
                    "conditioning_cache": str(cache_path), "conditioning_sha256": sha(cache_path),
                    "fusion": fusion, **measured,
                }
                write(ROOT / "records" / f"{model}_{prompt_id}_{variant}.json", record)
                print("MEASURED", model, prompt_id, variant, round(measured["seconds"], 3), flush=True)
        finally:
            acceleration.remove()
        del transformer, acceleration, layout
        gc.collect()
        torch.cuda.empty_cache()

        from dmad_h3 import load_vaes
        sys.path.insert(0, str(FV_EVAL))
        from fasth3_vbench_archive import archive

        with weight_load_lock():
            vae, audio_vae = load_vaes(MODEL, torch.device("cuda"))
        for variant, (video_rows, audio_rows) in payloads.items():
            write(status_path, {"status": "decoding", "model": model,
                               "prompt": prompt_id, "variant": variant})
            with torch.inference_mode():
                frames, audio = decode_rows(
                    vae, audio_vae, video_rows, audio_rows, shape, geometry
                )
            frames = frames[:REQUESTED_FRAMES]
            audio = audio[: round(REQUESTED_FRAMES / FPS * 32000)]
            assert frames.shape == (REQUESTED_FRAMES, HEIGHT, WIDTH, 3)
            target = OUT / model / variant
            target.mkdir(parents=True, exist_ok=True)
            video_path = target / f"{prompt_id}_{case['sample_id']}.mkv"
            evidence = archive(frames, audio, 32000, video_path, fps=FPS, threads=16)
            record_path = ROOT / "records" / f"{model}_{prompt_id}_{variant}.json"
            record = json.loads(record_path.read_text())
            record.update(video_path=str(video_path), video_sha256=sha(video_path),
                          archive=evidence, inspected=inspect_archive(video_path))
            write(record_path, record)
            print("DECODED", model, prompt_id, variant, flush=True)
        del vae, audio_vae, payloads
        gc.collect()
        torch.cuda.empty_cache()
        write(status_path, {"status": "complete", "model": model, "prompt": prompt_id})
    except BaseException:
        write(status_path, {"status": "failed", "model": model, "prompt": prompt_id,
                           "traceback": traceback.format_exc()})
        raise


def quality_pair(model, prompt, variant, gpu):
    import torch
    import torch.nn.functional as functional
    from decord import VideoReader

    sys.path.insert(0, str(LPIPS_PACKAGE))
    os.environ["TORCH_HOME"] = str(LPIPS_TORCH_HOME)
    import lpips

    ref = json.loads((ROOT / "records" / f"{model}_{prompt}_dense.json").read_text())
    cand = json.loads((ROOT / "records" / f"{model}_{prompt}_{variant}.json").read_text())
    a_reader, b_reader = VideoReader(ref["video_path"], num_threads=16), VideoReader(cand["video_path"], num_threads=16)
    assert len(a_reader) == len(b_reader) == REQUESTED_FRAMES
    metric = lpips.LPIPS(net="alex", version="0.1", lpips=True, pnet_rand=False,
                         eval_mode=True, verbose=False).cuda().eval()
    coordinates = torch.arange(11, device="cuda", dtype=torch.float32) - 5
    one = torch.exp(-coordinates.square() / 4.5)
    one /= one.sum()
    kernel = (one[:, None] * one[None, :]).expand(3, 1, 11, 11).contiguous()
    total_sse, total_values, ssims, distances = 0.0, 0, [], []
    with torch.inference_mode():
        for start in range(0, REQUESTED_FRAMES, 8):
            ids = list(range(start, min(start + 8, REQUESTED_FRAMES)))
            a = torch.from_numpy(a_reader.get_batch(ids).asnumpy()).cuda().float().permute(0, 3, 1, 2).div_(255)
            b = torch.from_numpy(b_reader.get_batch(ids).asnumpy()).cuda().float().permute(0, 3, 1, 2).div_(255)
            error = b - a
            total_sse += float(error.square().sum())
            total_values += error.numel()
            mux, muy = functional.conv2d(a, kernel, groups=3), functional.conv2d(b, kernel, groups=3)
            mux2, muy2, muxy = mux.square(), muy.square(), mux * muy
            sx2 = functional.conv2d(a.square(), kernel, groups=3) - mux2
            sy2 = functional.conv2d(b.square(), kernel, groups=3) - muy2
            sxy = functional.conv2d(a * b, kernel, groups=3) - muxy
            score = ((2 * muxy + 0.0001) * (2 * sxy + 0.0009) /
                     ((mux2 + muy2 + 0.0001) * (sx2 + sy2 + 0.0009)))
            ssims.extend(float(x) for x in score.flatten(1).mean(1).cpu())
            distances.extend(float(x) for x in metric(b.mul(2).sub(1), a.mul(2).sub(1),
                                                        normalize=False).flatten().cpu())
    mse = total_sse / total_values
    return {"model": model, "prompt": prompt, "variant": variant, "gpu": gpu,
            "reference": ref["video_path"], "candidate": cand["video_path"],
            "psnr_db": -10 * math.log10(mse), "mean_frame_ssim": statistics.mean(ssims),
            "mean_frame_lpips_alex_v0_1": statistics.mean(distances)}


def quality_worker(rank):
    tasks = [(m, p, v) for m in MODELS_UNDER_TEST for p in PROMPTS
             for v in VARIANTS if v != "dense"]
    for model, prompt, variant in tasks[int(rank)::len(GPUS)]:
        row = quality_pair(model, prompt, variant, GPUS[int(rank)])
        write(ROOT / "quality" / f"{model}_{prompt}_{variant}.json", row)
        print("QUALITY", model, prompt, variant, round(row["psnr_db"], 3), flush=True)


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    (ROOT / "records").mkdir(parents=True)
    (ROOT / "quality").mkdir()
    OUT.mkdir(parents=True)
    assert git(DMAD_REPO, "rev-parse", "HEAD") == DMAD_COMMIT
    protocol = {
        "status": "running", "name": NAME, "matrix": "2 prompts x 2 models x 3 variants",
        "jobs": JOBS, "models": MODELS_UNDER_TEST, "variants": VARIANTS,
        "samples": str(SAMPLES), "height": HEIGHT, "width": WIDTH,
        "requested_frames": REQUESTED_FRAMES, "aligned_frames": ALIGNED_FRAMES,
        "fps": FPS, "seed": SEED,
        "models_protocol": {
            "dmad": {"nfe": 4, "shifts": [12, 2], "rule": "official re-noise",
                     "training_length_note": "official checkpoint trained at 124 frames; 240-frame test is extrapolation"},
            "pdd": {"nfe": 8, "shifts": [12, 3], "rule": "official PDD block Euler"},
        },
        "variants_protocol": {
            "dense": "dense attention",
            "spark10_warm": "Top-K 10%; first 20% evaluations dense; transformer layer 0 always dense",
            "spark10_nowarm": "Top-K 10% from evaluation 0; no always-dense layer",
        },
        "runtime_warmup": "one full same-prompt pass per variant, discarded",
        "timing": "CUDA-synchronized student rollout only; loading, warmup, decode and archive excluded",
        "quality": "PSNR/SSIM/LPIPS against same-model same-prompt Dense output",
        "weights": {"dmad": {"path": str(DMAD_WEIGHT), "sha256": sha(DMAD_WEIGHT)},
                    "pdd": {"path": str(PDD_WEIGHT), "sha256": sha(PDD_WEIGHT)}},
        "git_revision": git(REPO, "rev-parse", "HEAD"),
        "dmad_source_revision": DMAD_COMMIT,
        "runner_sha256": sha(Path(__file__).resolve()),
        "environment": ENV,
    }
    write(ROOT / "protocol.json", protocol)
    shutil.copy2(__file__, ROOT / "runner_source.py")


def summarize():
    records = [json.loads(p.read_text()) for p in (ROOT / "records").glob("*.json")
               if "warmup" not in p.name]
    quality = [json.loads(p.read_text()) for p in (ROOT / "quality").glob("*.json")]
    if len(records) != 12 or len(quality) != 8:
        raise RuntimeError(f"incomplete records={len(records)} quality={len(quality)}")
    result = {"status": "complete", "outputs": 12, "timing": {}, "quality": {}}
    for model in MODELS_UNDER_TEST:
        result["timing"][model] = {}
        dense_mean = statistics.mean(r["seconds"] for r in records
                                     if r["model"] == model and r["variant"] == "dense")
        for variant in VARIANTS:
            selected = sorted((r for r in records if r["model"] == model and r["variant"] == variant),
                              key=lambda row: row["prompt"])
            mean = statistics.mean(r["seconds"] for r in selected)
            result["timing"][model][variant] = {
                "seconds": [r["seconds"] for r in selected], "mean_seconds": mean,
                "speedup_vs_dense": dense_mean / mean,
            }
        result["quality"][model] = {}
        for variant in VARIANTS[1:]:
            selected = sorted((r for r in quality if r["model"] == model and r["variant"] == variant),
                              key=lambda row: row["prompt"])
            result["quality"][model][variant] = {
                "psnr_db": statistics.mean(r["psnr_db"] for r in selected),
                "ssim": statistics.mean(r["mean_frame_ssim"] for r in selected),
                "lpips": statistics.mean(r["mean_frame_lpips_alex_v0_1"] for r in selected),
                "per_prompt": selected,
            }
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
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker", str(rank)],
            env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu),
                 "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"generation workers failed: {codes}")
    quality_jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"quality_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "quality", str(rank)],
            env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        quality_jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in quality_jobs]
    for _, _, log in quality_jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"quality workers failed: {codes}")
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        launch()
    elif command == "worker":
        worker(sys.argv[2])
    elif command == "quality":
        quality_worker(sys.argv[2])
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
