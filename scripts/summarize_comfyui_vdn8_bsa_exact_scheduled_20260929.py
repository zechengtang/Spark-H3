"""Summarize rerun quality, speed and score changes from the original SDPA reference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import comfyui_vdn8_four_model_72_run_20260929 as prior
import comfyui_vdn8_bsa_all_exact_dense_rerun_20260929 as rerun


def main():
    new = rerun.ROOT
    result = prior.read(new / "results.json")
    rows = result["rows"]
    assert result["status"] == "complete" and len(rows) == 72
    assert len(list(rerun.OUT.glob("*/*.mp4"))) == 12
    assert all(prior.read(new / model / "quality_complete.json")["status"] == "complete"
               for model in prior.MODELS)
    methods = prior.METHODS[1:]
    deltas = []
    for row in rows:
        if row["method"] == "old_dense":
            continue
        case_index = {"5s": 1, "10s": 2, "14p4s": 3}[row["duration"]]
        old = prior.read(prior.ROOT / row["model"] / "quality" / row["method"] /
                         f"{row['method']}_{case_index:02}.json")
        deltas.append({"model": row["model"], "duration": row["duration"],
                       "method": row["method"],
                       "old_psnr_db": old["psnr_db"], "new_psnr_db": row["psnr_db"],
                       "delta_psnr_db": row["psnr_db"] - old["psnr_db"],
                       "old_ssim": old["ssim"], "new_ssim": row["ssim"],
                       "delta_ssim": row["ssim"] - old["ssim"],
                       "old_lpips": old["lpips"], "new_lpips": row["lpips"],
                       "delta_lpips": row["lpips"] - old["lpips"]})
    assert len(deltas) == 60
    with (new / "old_vs_new_reference_deltas.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    lines = [
        "# VDN8: scheduled BSA full-sink dense reference",
        "",
        "Completed 2026-09-29 on GPUs 1–3. Twelve new reference videos: four models",
        "times three prompt durations. Seed 42, original model/LoRA weights, frame",
        "counts and sampler steps. The 60 existing sparse videos were not regenerated.",
        "Video PSNR/SSIM/LPIPS were rescored against the new reference. The",
        "12 original SDPA dense videos were also scored against it.",
        "",
        "Schedule: the first four of 20 model evaluations (base H3), or first two",
        "of eight (each LoRA), use the original SDPA block. Transformer layer 0",
        "also always uses SDPA. Later eligible layers use BSA with",
        "`sink_blocks=[0, ceil(tokens/64)]`, `tail=False`, `topk_ratio=0`,",
        "and no extra tokens. All H3 sequences here exceed `min_tokens=12288`.",
        "The original Sol and Spark candidate videos retain their original",
        "method-specific schedules; only the new reference uses this control.",
        "",
        "## Mean scores over three durations",
        "",
        "| Model | Candidate | New PSNR dB ↑ | ΔPSNR vs old ref | New SSIM ↑ | New LPIPS ↓ | Mean speedup vs new reference |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in prior.MODELS:
        for method in (*methods, "old_dense"):
            group = [r for r in rows if r["model"] == model and r["method"] == method]
            assert len(group) == 3
            dgroup = [d for d in deltas if d["model"] == model and d["method"] == method]
            delta = statistics.mean(d["delta_psnr_db"] for d in dgroup) if dgroup else None
            lines.append(
                f"| {model} | {method} | {statistics.mean(r['psnr_db'] for r in group):.3f} | "
                f"{delta:+.3f} dB" if delta is not None else
                f"| {model} | {method} | {statistics.mean(r['psnr_db'] for r in group):.3f} | —"
            )
            lines[-1] += (
                f" | {statistics.mean(r['ssim'] for r in group):.4f}"
                f" | {statistics.mean(r['lpips'] for r in group):.4f}"
                f" | {statistics.mean(r['speedup'] for r in group):.3f}× |"
            )
    improved = sum(d["delta_psnr_db"] > 0 for d in deltas)
    worsened = sum(d["delta_psnr_db"] < 0 for d in deltas)
    ssim_improved = sum(d["delta_ssim"] > 0 for d in deltas)
    lpips_improved = sum(d["delta_lpips"] < 0 for d in deltas)
    lines += [
        "", "## Answer to the reference question", "",
        f"Across all 60 sparse pairs, {improved} PSNR values increased and {worsened} decreased",
        f"relative to the original SDPA dense reference; mean ΔPSNR = "
        f"{statistics.mean(d['delta_psnr_db'] for d in deltas):+.3f} dB.",
        f"SSIM increased in {ssim_improved}/60 pairs; LPIPS decreased (improved) in {lpips_improved}/60.",
        "There is no general PSNR increase from changing the reference.",
        "The full per-pair score deltas are in `old_vs_new_reference_deltas.csv`.",
        "", "## Reference sampler times", "",
        "| Model | Duration | Old SDPA dense s | Scheduled BSA reference s | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in prior.MODELS:
        for duration in ("5s", "10s", "14p4s"):
            a = prior.read(prior.ROOT / model / "records" / f"{duration}_dense.json")["sampler_seconds"]
            b = prior.read(new / model / "records" / f"{duration}_bsa_all_exact.json")["sampler_seconds"]
            lines.append(f"| {model} | {duration} | {a:.2f} | {b:.2f} | {a/b:.3f}× |")
    lines += [
        "", "## Artifacts", "",
        "- `results.csv`: 72 pair scores and sampler timings (60 sparse plus 12 old dense).",
        "- `results.json`: machine-readable full results and video paths.",
        "- `old_vs_new_reference_deltas.csv`: each sparse pair's old/new metrics.",
        "- `protocol.json`: weights, prompts, seed, GPU assignment and source hashes.",
        "- Per-model `graphs/`, `records/`, `videos/` and `quality/` directories:",
        "  exact generation graph, latent hashes, video hashes and per-case scores.",
        "- Original 72-video experiment remains in the separate",
        "  `comfyui_vdn8_four_model_72_20260929` directory.",
    ]
    (new / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", new / "report.md", "improved", improved, "worsened", worsened)


if __name__ == "__main__":
    main()
