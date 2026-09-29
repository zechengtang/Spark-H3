#!/usr/bin/env python3
"""Capture one real post-RoPE head before the first Spark sparse attention call."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(os.environ.get(
    "H3_CAPTURE_ROOT",
    "/autodl-fs/data/h3_experiments/blog50_spark_intermediate_20260927",
))
HEAD_SPEC = os.environ.get("H3_CAPTURE_HEADS", "0")
HEADS = None if HEAD_SPEC == "all" else tuple(int(value) for value in HEAD_SPEC.split(","))
HIST_BASE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
BENCH = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark")
SAMPLES = BENCH / "vbench_core5_percent_subsets/20pct/samples.json"


class Captured(RuntimeError):
    pass


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("capture is assigned exclusively to GPU0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(BENCH / "scripts"))
    import _impl_bootstrap  # noqa: F401
    import minimax_h3_vbench_4gpu_pipeline as pipeline
    from h3_sparse_attention import H3SparseAttentionConfig, install_h3_sparse_attention
    import h3_sparse_attention.spark_integration as integration

    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    output = ROOT / "workflow"
    output.mkdir()
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        source = HIST_BASE / name
        (output / name).symlink_to(source, target_is_directory=source.is_dir())

    args = pipeline.build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(output),
        "--method", "dense", "--case-indices", "5", "--steps", "20",
        "--frames", "240", "--height", "768", "--width", "1344",
        "--workers", "1",
    ])
    cases = pipeline.load_cases(SAMPLES, [5], expected_indices=tuple(range(1, 51)))
    workflow, states = pipeline.configure_denoise_workflow(args, cases)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)
    if len(acceleration._forward_originals) != 50:
        raise RuntimeError("torch.compile did not wrap 50 transformer blocks")
    config = H3SparseAttentionConfig.spark(
        20, sol_tau=1.0, sol_thresh_type="diag", sol_kv_splits=1,
        warmup_percent=20, sol_dense_layers=1, sol_log_density=False,
        sol_force_local_blocks=False, sol_tail_granularity="query",
        sol_video_tail_mode="dense", sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore", sol_reweight_logmass_key="stored",
        sol_reweight_components="full", sol_route_topk_execution="threshold",
    )
    original = integration.spark_attention
    captured_meta = {}

    def capture(controller, q, k, v, layout, layer, *, return_bthd=False):
        if controller.evaluation_index != 4 or layer != 1:
            return original(controller, q, k, v, layout, layer, return_bthd=return_bthd)
        selected = tuple(range(q.shape[1])) if HEADS is None else HEADS
        values = {
            "query": q[:, selected].permute(0, 2, 1, 3).contiguous().cpu(),
            "key": k[:, selected].permute(0, 2, 1, 3).contiguous().cpu(),
            "value": v[:, selected].permute(0, 2, 1, 3).contiguous().cpu(),
            "video_tokens": layout.video_tokens,
            "sequence_length": layout.sequence_length,
            "grid": layout.grid,
            "evaluation": controller.evaluation_index,
            "layer": layer,
            "heads": selected,
            "case": 5,
            "seed": 42,
            "compile_wrapped_blocks": 50,
        }
        captured_meta.update(heads=selected, shape=list(values["query"].shape))
        target = ROOT / "post_rope_head0_eval04_layer01.pt"
        torch.save(values, target)
        print("CAPTURED", target, [(name, digest(values[name])) for name in ("query", "key", "value")], flush=True)
        raise Captured()

    integration.spark_attention = capture
    try:
        state = pipeline.clone_state(states[0])
        state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
        with install_h3_sparse_attention(pipe.transformer, config), torch.inference_mode():
            pipe(
                state=state, num_frames=240, height=768, width=1344,
                num_inference_steps=20,
                generator=torch.Generator(device="cpu").manual_seed(42),
                output=["latents", "audio_latents"],
            )
    except Captured:
        pass
    else:
        raise RuntimeError("Spark capture hook was not reached")
    finally:
        integration.spark_attention = original
        acceleration.remove()
    (ROOT / "capture_meta.json").write_text(json.dumps({
        "case": 5, "evaluation": 4, "layer": 1, "heads": captured_meta["heads"],
        "shape": captured_meta["shape"], "placement": placement,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
