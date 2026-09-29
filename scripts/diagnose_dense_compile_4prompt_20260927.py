#!/usr/bin/env python3
"""Isolate compile sensitivity of dense H3 denoising on four matched prompts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import diagnose_spark_psnr_regression_4prompt_20260927 as prior

ROOT = Path("/autodl-fs/data/h3_experiments/diagnose_dense_compile_4prompt_20260927")
OUT = Path("/autodl-fs/data/h3_outputs/diagnose_dense_compile_4prompt_20260927")
CASES = (2, 4, 6, 8)
METHODS = ("dense_eager", "dense_compile")
GPUS = (0, 1, 2, 3)


def configure():
    prior.ROOT, prior.OUT, prior.CASES = ROOT, OUT, CASES
    prior.configure()


def args_for(command, *, compile_enabled: bool):
    argv = [command, "--samples", str(prior.route.SAMPLES),
            "--output", str(OUT), "--method", "dense",
            "--steps", "20", "--frames", "240", "--height", "768",
            "--width", "1344", "--workers", "1"]
    if not compile_enabled:
        argv.append("--no-torch-compile")
    return prior.route.base.pipeline().build_parser().parse_args(argv)


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("diagnostic output already exists")
    selected = prior.route.cases()
    for method in METHODS:
        (ROOT / "records" / method).mkdir(parents=True, exist_ok=True)
        (OUT / method / "latents").mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(prior.COND / name, target_is_directory=name == "conditioning_cache")
    prior.route.base.write(ROOT / "protocol.json", dict(
        purpose="compare dense compile-on versus eager, without any sparse plugin",
        cases=selected, methods=METHODS, gpus=GPUS,
        seed=42, steps=20, evaluations=19, frames=240,
        height=768, width=1344, conditioning_source=str(prior.COND),
        dense_reference=str(prior.DENSE)))
    prior.route.base.pipeline().configure_denoise_workflow(
        args_for("denoise", compile_enabled=False), selected)


def worker(rank, only_method=None):
    import torch

    rank = int(rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = prior.route.base.pipeline()
    case = prior.route.cases()[rank]
    methods = (only_method,) if only_method is not None else (
        METHODS if rank % 2 == 0 else METHODS[::-1])
    for method in methods:
        if method not in METHODS:
            raise ValueError(method)
        record_path = ROOT / "records" / method / f"case_{case['index']:02}.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if Path(record["latent_path"]).is_file():
                print("REUSE", rank, method, case["index"], flush=True)
                continue
        compile_enabled = method == "dense_compile"
        args = args_for("denoise", compile_enabled=compile_enabled)
        workflow, states = p.configure_denoise_workflow(args, [case])
        pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
        try:
            state = p.clone_state(states[0])
            state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
            with torch.inference_mode():
                result = pipe(
                    state=state, num_frames=240, height=768, width=1344,
                    num_inference_steps=20,
                    generator=torch.Generator(device="cpu").manual_seed(42),
                    output=["latents", "audio_latents"],
                )
            payload = {key: result[key].detach().cpu().contiguous()
                       for key in ("latents", "audio_latents")}
            if not all(torch.isfinite(value).all() for value in payload.values()):
                raise FloatingPointError(f"non-finite {method} case {case['index']}")
            path = OUT / method / "latents" / f"{case['index']:02}_{case['sample_id']}.pt"
            p.atomic_torch_save(payload, path)
            prior.route.base.write(ROOT / "records" / method / f"case_{case['index']:02}.json", dict(
                method=method, case=case["index"], sample_id=case["sample_id"],
                prompt_sha256=case["prompt_sha256"], torch_compile=compile_enabled,
                latent_path=str(path), latent_sha256=prior.route.base.sha(path),
                seed=42, steps=20, frames=240, height=768, width=1344,
                placement=placement))
            print("DONE", rank, method, case["index"], flush=True)
            del result, payload, state
        finally:
            acceleration.remove()
            del pipe, manager
            p.release_cpu_arenas()
            torch.cuda.empty_cache()


def run():
    prepare()
    resume()


def resume():
    """Run only missing arms; each arm gets a fresh process and CUDA context."""
    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)
    cases = prior.route.cases()
    for method in METHODS:
        jobs = []
        for rank, gpu in enumerate(GPUS):
            case = cases[rank]
            record_path = ROOT / "records" / method / f"case_{case['index']:02}.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text())
                if Path(record["latent_path"]).is_file():
                    continue
            log = (ROOT / f"resume_gpu{gpu}_{method}.log").open("a")
            process = subprocess.Popen(
                [str(prior.route.base.PYTHON), str(Path(__file__).resolve()),
                 "worker", str(rank), method],
                env={**os.environ, **prior.route.base.ENV,
                     "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log, stderr=subprocess.STDOUT)
            jobs.append((gpu, process, log))
        codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
        for _, _, log in jobs:
            log.close()
        if any(code for _, code in codes):
            raise RuntimeError(f"dense diagnostic resume failed for {method}: {codes}")


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run()
    elif command == "resume":
        resume()
    elif command == "worker":
        worker(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        raise ValueError(command)
