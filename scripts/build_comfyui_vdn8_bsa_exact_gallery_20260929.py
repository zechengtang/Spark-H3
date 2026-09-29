"""Build a 6008 subpage for the scheduled BSA reference and existing sparse videos."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import build_comfyui_vdn8_gallery_20260929 as old_gallery
import comfyui_vdn8_four_model_72_run_20260929 as prior
import comfyui_vdn8_bsa_all_exact_dense_rerun_20260929 as rerun


def main():
    root = rerun.ROOT
    protocol = prior.read(root / "protocol.json")
    scored = prior.read(root / "results.json")
    scores = {(r["model"], r["duration"], r["method"]): r for r in scored["rows"]}
    cases = {c["label"]: c for c in protocol["cases"]}
    data = {"seed": protocol["seed"], "cases": cases, "results": {}}
    for model in prior.MODELS:
        data["results"][model] = {}
        for case in prior.cases():
            duration = case["label"]
            entries = {}
            for method in prior.METHODS:
                source = (rerun.OUT / model / f"{duration}_bsa_all_exact.mp4"
                          if method == "dense" else
                          prior.OUT / model / f"{duration}_{method}.mp4")
                target = root / "videos" / model / f"{duration}_{method}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != source.resolve():
                        raise RuntimeError(f"wrong video link: {target}")
                elif target.exists():
                    raise RuntimeError(f"video link occupied: {target}")
                else:
                    target.symlink_to(source)
                poster = root / "posters" / model / f"{duration}_{method}.jpg"
                poster.parent.mkdir(parents=True, exist_ok=True)
                if not poster.exists():
                    archived = old_gallery.ROOT / "posters" / model / poster.name
                    if method != "dense" and archived.is_file():
                        poster.symlink_to(archived)
                    else:
                        subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.5",
                                        "-i", str(source), "-frames:v", "1",
                                        "-vf", "scale=768:-2", "-q:v", "4", "-y",
                                        str(poster)], check=True)
                if method == "dense":
                    seconds = prior.read(root / model / "records" /
                                         f"{duration}_bsa_all_exact.json")["sampler_seconds"]
                    row = None
                else:
                    row = scores[(model, duration, method)]
                    seconds = row["candidate_sampler_seconds"]
                entries[method] = {
                    "src": f"videos/{model}/{duration}_{method}.mp4",
                    "poster": f"posters/{model}/{duration}_{method}.jpg",
                    "seconds": seconds,
                    "speedup": None if row is None else row["speedup"],
                    "psnr": None if row is None else row["psnr_db"],
                    "ssim": None if row is None else row["ssim"],
                    "lpips": None if row is None else row["lpips"],
                    "timing_note": (
                        "首个稀疏步含明显初始化开销"
                        if model in ("lightx2v", "comfyui") and duration == "14p4s"
                        and method == "spark_10pct" else None),
                }
            data["results"][model][duration] = entries
    script_data = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = old_gallery.TEMPLATE.replace("__DATA__", script_data)
    page = page.replace("<title>Prompt 08 · ComfyUI H3 视频对照</title>",
                        "<title>Prompt 08 · BSA 全量 Exact 参考</title>")
    page = page.replace("<h1>同一 Prompt，六种注意力方案</h1>",
                        "<h1>以 BSA 全量 Exact 为参考</h1>")
    page = page.replace(
        "每组保持相同的 prompt、seed、模型和采样步数；画质指标以该组 Dense 视频为参照。",
        "每组保持相同的 prompt、seed、模型和采样步数；前 20% 采样调用及第 0 层保留 SDPA，"
        "其余层的参考视频使用全范围 sink blocks 的 BSA exact。画质指标以该参考视频为准。")
    page = page.replace("['dense','Dense']", "['dense','BSA exact 参考']")
    page = page.replace("'Dense 基准'", "'BSA exact 基准'")
    page = page.replace("相对于 Dense", "相对于 BSA exact")
    page = page.replace(
        "页面显示原实验的单次去噪计时。每模型只做过 5 秒 Dense 两步预热，稀疏模式与各时长没有逐一预热；",
        "页面显示新 BSA 参考和原稀疏变体各自的单次去噪计时。新参考每模型先做两步预热；"
        "原稀疏模式与各时长没有逐一预热；")
    (root / "index.html").write_text(page)
    print(root / "index.html")


if __name__ == "__main__":
    main()
