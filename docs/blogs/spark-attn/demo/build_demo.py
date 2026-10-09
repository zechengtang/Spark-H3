#!/usr/bin/env python3
"""Build the Spark-H3 visual-comparison demo reel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
INPUT_ROOT = Path(
    os.environ.get("SPARK_DEMO_INPUT_ROOT", REPO / ".demo-inputs")
).expanduser()
GALLERY = Path(
    os.environ.get("SPARK_DEMO_GALLERY", INPUT_ROOT / "gallery")
).expanduser()
LIGHTX2V_GALLERY = Path(
    os.environ.get("SPARK_DEMO_LIGHTX2V_GALLERY", INPUT_ROOT / "lightx2v")
).expanduser()
REF2VA_MANIFEST = REPO / "docs" / "blogs" / "spark-attn" / "integration" / "ref2va.json"
REF2VA_GALLERY = (
    REPO / "docs" / "blogs" / "spark-attn" / ".preview"
    / "integration" / "ref2va" / "media"
)
WORDMARK = REPO / "assets" / "spark-h3-wordmark.png"
PKU_LOGO = REPO / "assets" / "school_logos" / "PKU.png"
NJU_LOGO = REPO / "assets" / "school_logos" / "NJU.jpg"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

FPS = 24
WIDTH, HEIGHT = 1920, 1080
COMPARISON_SECONDS = 10
SLOW_FACTOR = 4.0
DETAIL_GREEN = "#5f8f73"
DETAIL_NEUTRAL = "#69727e"
DETAIL_ORANGE = "#ff6b00"
DETAIL_GOLD = "#c69214"
CROSSFADE_SECONDS = 0.6
FINAL_CROSSFADE_SECONDS = 1.2
CASE_HOLD_SECONDS = 0.75
REF2VA_POST_PLAY_HOLD_SECONDS = 0.6
OUTRO_HOLD_SECONDS = 6.2
BGM_START_OFFSET_SECONDS = 4.0
BGM_MUTE_RAMP_SECONDS = 0.3
ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "slow", "-crf", "14",
    "-maxrate", "24M", "-bufsize", "48M",
]

# Display factors from the corrected 14.4 s VDN-protocol retest:
# 9 sigma points / 8 evaluations, first 2 evaluations and layer 0 Dense.
FEW_STEP_SPEEDUP = 6.13
VDN_WARMUP_DIT_SPEEDUP = 1.78
COMBINED_FEW_STEP_SPEEDUP = FEW_STEP_SPEEDUP * VDN_WARMUP_DIT_SPEEDUP

# Blog-wide mean DiT-latency speedups from the current benchmark table.
BLOG_DIT_SPEEDUPS = ("Baseline", "1.64× Speedup", "1.77× Speedup")

# (display number, gallery key, highlighted source ranges, opening hold,
# optional detail region shown during the opening hold)
# A highlighted range is (start, end, region).
CASES = [
    (
        "1", "train47",
        ((1.5, 3.5, "bottom_right_1_5x"),),
        CASE_HOLD_SECONDS, None,
    ),
    (
        "2", "v0685",
        ((0.0, 1.0, "bottom_right"),), CASE_HOLD_SECONDS, None,
    ),
]

LIGHTX2V_CASES = [
    ("3", "vdn_0001_japanese_woman_city_closeup.mp4", 262.2),
    ("4", "vdn_0010_2d_anime_eye_pullout.mp4", 265.74),
]

def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def drawtext(
    text: str,
    x: str,
    y: str,
    size: int,
    color: str = "#15181e",
    *,
    bold: bool = False,
) -> str:
    escaped = text.replace("\\", r"\\").replace("'", r"\'").replace(":", r"\:")
    font = FONT_BOLD if bold else FONT
    return (
        f"drawtext=fontfile='{font}':text='{escaped}':x={x}:y={y}:"
        f"fontsize={size}:fontcolor={color}:expansion=none"
    )


def source_video(case_key: str, method: str) -> Path:
    candidate = GALLERY / f"{case_key}_{method}.mp4"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def render_intro(output: Path) -> None:
    decorations = [
        drawtext(
            "Adaptive Block-Sparse Attention for MiniMax-H3",
            "(w-text_w)/2", "553", 36, "#252a31", bold=True,
        ),
        "drawbox=x=715:y=633:w=490:h=4:color=#ff6b00:t=fill",
        drawtext(
            "REBLOCK  ·  REWEIGHT  ·  HIGH SPARSITY",
            "(w-text_w)/2", "678", 25, "#d95300", bold=True,
        ),
    ]
    filters = (
        "[1:v]crop=1950:625:0:130,scale=1080:-2[wordmark];"
        "[2:v]scale=124:124[pku];"
        "[3:v]scale=-2:132[nju];"
        "[0:v][wordmark]overlay=(W-w)/2:203[stage1];"
        "[stage1][pku]overlay=815:838[stage2];"
        "[stage2][nju]overlay=1005:834[stage3];"
        "[stage3]" + ",".join(decorations)
        + ",fade=t=in:st=0:d=0.4,format=yuv420p[out]"
    )
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=white:s={WIDTH}x{HEIGHT}:r={FPS}:d=3",
            "-loop", "1", "-framerate", str(FPS), "-i", str(WORDMARK),
            "-loop", "1", "-framerate", str(FPS), "-i", str(PKU_LOGO),
            "-loop", "1", "-framerate", str(FPS), "-i", str(NJU_LOGO),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-filter_complex", filters, "-map", "[out]", "-map", "4:a",
            "-t", "3", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )


def render_lora_transition(output: Path) -> None:
    filters = [
        drawtext(
            "Spark-H3 + Community LoRAs", "(w-text_w)/2", "229", 25,
            "#d95300", bold=True,
        ),
        drawtext(
            "Spark-H3 can be integrated", "(w-text_w)/2", "319", 56,
            "#15181e", bold=True,
        ),
        drawtext(
            "with community LoRAs.", "(w-text_w)/2", "391", 56,
            "#15181e", bold=True,
        ),
        "drawbox=x=710:y=489:w=500:h=4:color=#ff6b00:t=fill",
        drawtext(
            "For example, few-step LoRAs enable further speedup.",
            "(w-text_w)/2", "539", 25, "#5b626c",
        ),
        drawtext(
            "Here, we use the 8-step LoRA from LightX2V.",
            "(w-text_w)/2", "584", 25, "#5b626c",
        ),
        drawtext(
            "LightX2V  ·  Larryvrh  ·  Acc-LoRA  ·  DMAD-LoRA",
            "(w-text_w)/2", "649", 22, "#d95300", bold=True,
        ),
        "format=yuv420p",
    ]
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=#f7f7f4:s={WIDTH}x{HEIGHT}:r={FPS}:d=3.2",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-vf", ",".join(filters), "-map", "0:v", "-map", "1:a",
            "-t", "3.2", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )


def render_ref2va_transition(output: Path) -> None:
    filters = [
        drawtext(
            "Spark-H3 + MiniMax-H3 Ref2VA", "(w-text_w)/2", "229", 25,
            "#d95300", bold=True,
        ),
        drawtext(
            "Spark-H3 can also be used", "(w-text_w)/2", "319", 56,
            "#15181e", bold=True,
        ),
        drawtext(
            "with the MiniMax-H3 Ref2VA variant.", "(w-text_w)/2", "391", 56,
            "#15181e", bold=True,
        ),
        "drawbox=x=710:y=489:w=500:h=4:color=#ff6b00:t=fill",
        drawtext(
            "Spark-H3 accelerates both reference and target tokens.",
            "(w-text_w)/2", "539", 25, "#5b626c",
        ),
        drawtext(
            "Compact Cond provides an additional speedup.",
            "(w-text_w)/2", "584", 25, "#5b626c",
        ),
        "format=yuv420p",
    ]
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=#f7f7f4:s={WIDTH}x{HEIGHT}:r={FPS}:d=3.2",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-vf", ",".join(filters), "-map", "0:v", "-map", "1:a",
            "-t", "3.2", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )


def retimed_video_filter(
    input_index: int,
    label: str,
    slow_ranges: tuple[tuple[float, float, str], ...],
    width: int,
    height: int,
    hold_seconds: float,
) -> str:
    if not slow_ranges:
        return (
            f"[{input_index}:v]trim=duration={COMPARISON_SECONDS},setpts=PTS-STARTPTS,"
            f"tpad=start_mode=clone:start_duration={hold_seconds:g},"
            f"fps={FPS},scale={width}:{height}[{label}];"
        )

    segments: list[tuple[float, float, float]] = []
    cursor = 0.0
    for start, end, _ in slow_ranges:
        if cursor < start:
            segments.append((cursor, start, 1.0))
        segments.append((start, end, SLOW_FACTOR))
        cursor = end
    if cursor < COMPARISON_SECONDS:
        segments.append((cursor, COMPARISON_SECONDS, 1.0))

    filters: list[str] = []
    segment_labels: list[str] = []
    for index, (start, end, factor) in enumerate(segments):
        segment_label = f"{label}_segment_{index}"
        segment_labels.append(f"[{segment_label}]")
        filters.append(
            f"[{input_index}:v]trim=start={start}:end={end},"
            f"setpts={factor:g}*(PTS-STARTPTS)[{segment_label}]"
        )
    filters.append(
        "".join(segment_labels)
        + f"concat=n={len(segments)}:v=1:a=0,"
        f"tpad=start_mode=clone:start_duration={hold_seconds:g},"
        f"fps={FPS},scale={width}:{height}[{label}]"
    )
    return ";".join(filters) + ";"


def slow_motion_output_ranges(
    slow_ranges: tuple[tuple[float, float, str], ...],
    hold_seconds: float,
) -> list[tuple[float, float, str]]:
    result: list[tuple[float, float, str]] = []
    accumulated_extra = 0.0
    for start, end, region in slow_ranges:
        output_start = hold_seconds + start + accumulated_extra
        output_end = output_start + SLOW_FACTOR * (end - start)
        result.append((output_start, output_end, region))
        accumulated_extra += (SLOW_FACTOR - 1) * (end - start)
    return result


def region_box(
    panel_x: int,
    panel_y: int,
    panel_width: int,
    panel_height: int,
    region: str,
    expression: str,
    color: str,
) -> str:
    if region == "bottom_right_1_5x":
        box_width = panel_width * 2 // 3
        box_height = panel_height * 2 // 3
    else:
        box_width = panel_width // 2
        box_height = panel_height // 2
    if region in ("bottom_right", "bottom_right_up", "bottom_right_1_5x"):
        x = panel_x + panel_width - box_width
        y = panel_y + panel_height - box_height
        if region == "bottom_right_1_5x":
            x -= panel_width // 8
        if region == "bottom_right_up":
            y -= box_height // 5
    elif region in ("center", "center_left"):
        x = panel_x + (panel_width - box_width) // 2
        y = panel_y + (panel_height - box_height) // 2
        if region == "center_left":
            x -= panel_width // 8
    else:
        raise ValueError(f"Unknown highlight region: {region}")
    return (
        f"drawbox=x={x}:y={y}:w={box_width}:h={box_height}:"
        f"color={color}@0.98:t=6:enable='{expression}'"
    )


def crop_geometry(region: str, panel_width: int, panel_height: int) -> tuple[int, int, int, int]:
    if region == "bottom_right_1_5x":
        crop_width = panel_width * 2 // 3
        crop_height = panel_height * 2 // 3
    else:
        crop_width = panel_width // 2
        crop_height = panel_height // 2
    if region in ("bottom_right", "bottom_right_up", "bottom_right_1_5x"):
        x = panel_width - crop_width
        y = panel_height - crop_height
        if region == "bottom_right_1_5x":
            x -= panel_width // 8
        if region == "bottom_right_up":
            y -= crop_height // 5
        return crop_width, crop_height, x, y
    if region in ("center", "center_left"):
        x = (panel_width - crop_width) // 2
        if region == "center_left":
            x -= panel_width // 8
        return crop_width, crop_height, x, (panel_height - crop_height) // 2
    raise ValueError(f"Unknown highlight region: {region}")


def render_comparison_clip(
    display_number: str,
    case_key: str,
    slow_ranges: tuple[tuple[float, float, str], ...],
    hold_seconds: float,
    hold_detail_region: str | None,
    position: int,
    output: Path,
) -> None:
    methods = ("dense", "sol", "lite_fused")
    inputs = [source_video(case_key, method) for method in methods]
    panel_width, panel_height = 600, 342
    panel_y = 220
    detail_y = 660
    xs = [40, 660, 1280]
    labels = ["Dense", "Sol-H3", "Spark-H3"]
    metrics = BLOG_DIT_SPEEDUPS
    output_duration = hold_seconds + COMPARISON_SECONDS + sum(
        (SLOW_FACTOR - 1) * (end - start) for start, end, _ in slow_ranges
    )

    decorations = [
        drawtext(f"Case {display_number}", "40", "48", 42, "#15181e", bold=True),
        drawtext(
            "GENERATION SETTINGS", "1880-text_w", "30", 13,
            "#d95300", bold=True,
        ),
        drawtext(
            "10 s  ·  1344 × 768  ·  19 steps",
            "1880-text_w", "56", 21, "#3f4650", bold=True,
        ),
    ]
    for index, (x, label, metric) in enumerate(zip(xs, labels, metrics)):
        label_color = "#d95300" if index == 2 else "#15181e"
        metric_color = "#d95300" if index == 2 else "#5b626c"
        decorations.append(
            drawtext(label, str(x), "170", 29, label_color, bold=True)
        )
        if metric:
            decorations.append(
                drawtext(
                    metric, f"{x}+{panel_width}-text_w", "176", 22,
                    metric_color, bold=True,
                )
            )
    slow_enabled_ranges = slow_motion_output_ranges(slow_ranges, hold_seconds)
    detail_ranges = list(slow_enabled_ranges)
    if hold_detail_region is not None:
        detail_ranges.insert(0, (0.0, hold_seconds, hold_detail_region))
    slow_expression = "+".join(
        fr"between(t\,{start:g}\,{end:g})" for start, end, _ in slow_enabled_ranges
    )
    detail_expression = "+".join(
        fr"between(t\,{start:g}\,{end:g})" for start, end, _ in detail_ranges
    )
    if slow_enabled_ranges:
        decorations.extend(
            [
                f"drawbox=x=789:y=116:w=342:h=40:color=#ff6b00@0.94:t=fill:enable='{slow_expression}'",
                drawtext("4× SLOW MOTION", "(w-text_w)/2", "123", 22, "white", bold=True)
                + f":enable='{slow_expression}'",
            ]
        )
    if detail_ranges:
        detail_label = (
            "1.5× Detail"
            if all(region == "bottom_right_1_5x" for _, _, region in detail_ranges)
            else "2× Detail"
        )
        decorations.append(
            drawtext(detail_label, "(w-text_w)/2", "610", 22, "#d95300", bold=True)
            + f":enable='{detail_expression}'"
        )
    for start, end, region in detail_ranges:
        region_expression = fr"between(t\,{start:g}\,{end:g})"
        for method, x in zip(("dense", "sol", "spark"), xs):
            if method == "sol":
                detail_color = DETAIL_NEUTRAL
            elif method == "spark":
                detail_color = DETAIL_ORANGE
            else:
                detail_color = DETAIL_GREEN
            decorations.extend(
                [
                    region_box(
                        x, panel_y, panel_width, panel_height, region, region_expression,
                        detail_color,
                    ),
                    f"drawbox=x={x-5}:y={detail_y-5}:w={panel_width+10}:h={panel_height+10}:"
                    f"color={detail_color}:t=4:enable='{region_expression}'",
                ]
            )

    base_filters = (
        retimed_video_filter(0, "dense_base", slow_ranges, panel_width, panel_height, hold_seconds)
        + retimed_video_filter(1, "sol_base", slow_ranges, panel_width, panel_height, hold_seconds)
        + retimed_video_filter(2, "spark_base", slow_ranges, panel_width, panel_height, hold_seconds)
    )
    detail_filters: list[str] = []
    for method in ("dense", "sol", "spark"):
        if not detail_ranges:
            detail_filters.append(f"[{method}_base]null[{method}]")
            continue
        split_outputs = f"[{method}]" + "".join(
            f"[{method}_detail_source_{index}]" for index in range(len(detail_ranges))
        )
        detail_filters.append(
            f"[{method}_base]split={len(detail_ranges) + 1}{split_outputs}"
        )
        for index, (_, _, region) in enumerate(detail_ranges):
            crop_width, crop_height, crop_x, crop_y = crop_geometry(
                region, panel_width, panel_height,
            )
            detail_filters.append(
                f"[{method}_detail_source_{index}]crop={crop_width}:{crop_height}:"
                f"{crop_x}:{crop_y},scale={panel_width}:{panel_height}"
                f"[{method}_detail_{index}]"
            )

    filters = (
        base_filters
        + ";".join(detail_filters) + ";"
        + f"color=c=#f7f7f4:s={WIDTH}x{HEIGHT}:r={FPS}:d={output_duration}[bg];"
        f"[bg][dense]overlay={xs[0]}:{panel_y}[tmp1];"
        f"[tmp1][sol]overlay={xs[1]}:{panel_y}[tmp2];"
        f"[tmp2][spark]overlay={xs[2]}:{panel_y}[panels]"
    )
    previous = "panels"
    detail_index = 0
    for range_index, (start, end, _) in enumerate(detail_ranges):
        region_expression = fr"between(t\,{start:g}\,{end:g})"
        for method, x in zip(("dense", "sol", "spark"), xs):
            next_label = f"with_detail_{detail_index}"
            filters += (
                f";[{previous}][{method}_detail_{range_index}]overlay={x}:{detail_y}:"
                f"enable='{region_expression}'[{next_label}]"
            )
            previous = next_label
            detail_index += 1
    filters += (
        f";[{previous}]" + ",".join(decorations) + ",format=yuv420p[out]"
    )
    command = ["ffmpeg", "-y"]
    for source in inputs:
        command.extend(["-i", str(source)])
    command.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000"])
    command.extend(
        [
            "-filter_complex", filters, "-map", "[out]", "-map", "3:a",
            "-t", f"{output_duration:g}", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )
    run(command)


def render_lightx2v_clip(
    case_number: str,
    filename: str,
    latency: float,
    output: Path,
) -> None:
    source = LIGHTX2V_GALLERY / filename
    duration = probe_duration(source)
    output_duration = duration + CASE_HOLD_SECONDS
    audio_delay_ms = round(CASE_HOLD_SECONDS * 1000)
    video_width, video_height = 1428, 816
    video_x, video_y = 246, 146
    decorations = [
        drawtext(f"Case {case_number}", "56", "48", 40, "#15181e", bold=True),
        drawtext("LightX2V + Spark-H3", "(w-text_w)/2", "34", 30, "#252a31", bold=True),
        drawtext(
            (
                f"49 → 8 steps ({FEW_STEP_SPEEDUP:.2f}×)  ×  "
                f"90% sparse ({VDN_WARMUP_DIT_SPEEDUP:.2f}×)  =  "
                f"{COMBINED_FEW_STEP_SPEEDUP:.2f}× Speedup"
            ),
            "(w-text_w)/2", "78", 20, "#d95300", bold=True,
        ),
        drawtext(
            "GENERATION SETTINGS", "1864-text_w", "30", 13,
            "#d95300", bold=True,
        ),
        drawtext(
            "14.4 s  ·  1344 × 768  ·  8 steps", "1864-text_w", "56", 21,
            "#3f4650", bold=True,
        ),
        f"drawbox=x={video_x-5}:y={video_y-5}:w={video_width+10}:h={video_height+10}:color=#ff6b00:t=4",
    ]
    filters = (
        f"[0:v]fps={FPS},scale={video_width}:{video_height},"
        f"tpad=start_mode=clone:start_duration={CASE_HOLD_SECONDS:g}[video];"
        f"[0:a]adelay={audio_delay_ms}|{audio_delay_ms}[audio];"
        f"color=c=#f7f7f4:s={WIDTH}x{HEIGHT}:r={FPS}:d={output_duration}[bg];"
        f"[bg][video]overlay={video_x}:{video_y}[stage];"
        "[stage]" + ",".join(decorations) + ",format=yuv420p[out]"
    )
    run(
        [
            "ffmpeg", "-y", "-i", str(source), "-filter_complex", filters,
            "-map", "[out]", "-map", "[audio]", "-t", f"{output_duration:g}",
            "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", "32000", *ENCODE_ARGS,
            "-movflags", "+faststart", str(output),
        ]
    )


def render_ref2va_clip(output: Path) -> None:
    manifest = json.loads(REF2VA_MANIFEST.read_text())
    prompt = manifest["case"]["prompt"]
    prompt_first, prompt_remainder = prompt.split(", animate ", 1)
    prompt_second, prompt_third = prompt_remainder.split(", retain ", 1)
    prompt_lines = (
        f"{prompt_first},",
        f"animate {prompt_second},",
        f"retain {prompt_third}",
    )
    records = manifest["variants"]
    dense_seconds = records[0]["denoise_seconds"]
    sources = [REF2VA_GALLERY / Path(record["preview"]).name for record in records]
    duration = min(probe_duration(source) for source in sources)
    ref2va_tail_seconds = (
        REF2VA_POST_PLAY_HOLD_SECONDS + FINAL_CROSSFADE_SECONDS
    )
    output_duration = duration + CASE_HOLD_SECONDS + ref2va_tail_seconds
    audio_delay_ms = round(CASE_HOLD_SECONDS * 1000)

    panel_width, panel_height = 600, 342
    panel_y = 220
    xs = [40, 660, 1280]
    labels = ["Dense", "Spark-H3", "Spark-H3 + Compact Cond"]
    metrics = [
        "BASELINE",
        f"{dense_seconds / records[1]['denoise_seconds']:.2f}× Speedup",
        f"{dense_seconds / records[2]['denoise_seconds']:.2f}× Speedup",
    ]
    colors = [DETAIL_GREEN, DETAIL_ORANGE, DETAIL_GOLD]

    decorations = [
        drawtext("Official Ref2VA Case", "40", "48", 40, "#15181e", bold=True),
        drawtext(
            "GENERATION SETTINGS", "1880-text_w", "30", 13,
            "#d95300", bold=True,
        ),
        drawtext(
            "5 s  ·  1344 × 768  ·  19 steps",
            "1880-text_w", "56", 21, "#3f4650", bold=True,
        ),
    ]
    for index, (x, label, metric, color) in enumerate(zip(xs, labels, metrics, colors)):
        label_color = "#15181e" if index == 0 else color
        decorations.extend(
            [
                drawtext(label, str(x), "177", 25, label_color, bold=True),
                drawtext(
                    metric, f"{x}+{panel_width}-text_w", "183", 18,
                    color, bold=True,
                ),
                (
                    f"drawbox=x={x-4}:y={panel_y-4}:w={panel_width+8}:"
                    f"h={panel_height+8}:color={color}:t=4"
                ),
            ]
        )
    decorations.append(
        drawtext("PROMPT", "(w-text_w)/2", "622", 13, "#d95300", bold=True)
    )
    for line_index, line in enumerate(prompt_lines):
        decorations.append(
            drawtext(
                line, "(w-text_w)/2", str(652 + line_index * 34),
                20, "#3f4650",
            )
        )

    filters = (
        f"[0:v]fps={FPS},scale={panel_width}:{panel_height},"
        f"tpad=start_mode=clone:start_duration={CASE_HOLD_SECONDS:g}:"
        f"stop_mode=clone:stop_duration={ref2va_tail_seconds:g}[dense];"
        f"[1:v]fps={FPS},scale={panel_width}:{panel_height},"
        f"tpad=start_mode=clone:start_duration={CASE_HOLD_SECONDS:g}:"
        f"stop_mode=clone:stop_duration={ref2va_tail_seconds:g}[spark];"
        f"[2:v]fps={FPS},scale={panel_width}:{panel_height},"
        f"tpad=start_mode=clone:start_duration={CASE_HOLD_SECONDS:g}:"
        f"stop_mode=clone:stop_duration={ref2va_tail_seconds:g}[compressed];"
        f"[0:a]adelay={audio_delay_ms}|{audio_delay_ms},"
        f"apad=pad_dur={ref2va_tail_seconds:g}[audio];"
        f"color=c=#f7f7f4:s={WIDTH}x{HEIGHT}:r={FPS}:d={output_duration}[bg];"
        f"[bg][dense]overlay={xs[0]}:{panel_y}[stage1];"
        f"[stage1][spark]overlay={xs[1]}:{panel_y}[stage2];"
        f"[stage2][compressed]overlay={xs[2]}:{panel_y}[stage3];"
        "[stage3]" + ",".join(decorations) + ",format=yuv420p[out]"
    )
    command = ["ffmpeg", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex", filters, "-map", "[out]", "-map", "[audio]",
            "-t", f"{output_duration:g}", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", "32000",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )
    run(command)


def render_outro(output: Path) -> None:
    # The first FINAL_CROSSFADE_SECONDS overlap the preceding clip, so extend
    # the source to preserve the requested full hold after the transition.
    output_duration = OUTRO_HOLD_SECONDS + FINAL_CROSSFADE_SECONDS
    decorations = [
        drawtext("THANKS FOR WATCHING.", "(w-text_w)/2", "555", 54, "#15181e", bold=True),
        "drawbox=x=760:y=640:w=400:h=4:color=#ff6b00:t=fill",
        drawtext(
            "STAY TUNED!",
            "(w-text_w)/2", "690", 29, "#d95300", bold=True,
        ),
        drawtext(
            "github.com/zechengtang/Spark-H3",
            "(w-text_w)/2", "760", 22, "#5b626c",
        ),
    ]
    filters = (
        "[1:v]crop=1950:625:0:130,scale=650:-2[wordmark];"
        "[0:v][wordmark]overlay=(W-w)/2:220[stage];"
        "[stage]" + ",".join(decorations)
        + f",fade=t=out:st={output_duration - 0.5:g}:d=0.5,format=yuv420p[out]"
    )
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=white:s={WIDTH}x{HEIGHT}:r={FPS}:d={output_duration:g}",
            "-loop", "1", "-framerate", str(FPS), "-i", str(WORDMARK),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-filter_complex", filters, "-map", "[out]", "-map", "2:a",
            "-t", f"{output_duration:g}", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )


def concatenate_with_transitions(clips: list[Path], output: Path) -> tuple[list[float], list[float]]:
    durations = [probe_duration(clip) for clip in clips]
    starts = [0.0]
    for index, duration in enumerate(durations[:-1]):
        transition_duration = (
            FINAL_CROSSFADE_SECONDS
            if index == len(durations) - 2
            else CROSSFADE_SECONDS
        )
        starts.append(starts[-1] + duration - transition_duration)

    filters: list[str] = []
    for index in range(len(clips)):
        filters.extend(
            [
                f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[v{index}]",
                f"[{index}:a]aresample=32000,asetpts=PTS-STARTPTS[a{index}]",
            ]
        )

    video_label = "v0"
    audio_label = "a0"
    timeline_duration = durations[0]
    for index in range(1, len(clips)):
        video_out = f"vx{index}"
        audio_out = f"ax{index}"
        transition_duration = (
            FINAL_CROSSFADE_SECONDS
            if index == len(clips) - 1
            else CROSSFADE_SECONDS
        )
        offset = timeline_duration - transition_duration
        filters.extend(
            [
                f"[{video_label}][v{index}]xfade=transition=fade:"
                f"duration={transition_duration:g}:offset={offset:.6f}[{video_out}]",
                f"[{audio_label}][a{index}]acrossfade=d={transition_duration:g}:"
                f"c1=tri:c2=tri[{audio_out}]",
            ]
        )
        video_label = video_out
        audio_label = audio_out
        timeline_duration += durations[index] - transition_duration

    filters.append(f"[{video_label}]format=yuv420p[vfinal]")
    video_label = "vfinal"

    command = ["ffmpeg", "-y"]
    for clip in clips:
        command.extend(["-i", str(clip)])
    command.extend(
        [
            "-filter_complex", ";".join(filters),
            "-map", f"[{video_label}]", "-map", f"[{audio_label}]",
            "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "32000",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )
    run(command)
    return durations, starts


def add_background_music(
    video: Path,
    music: Path,
    output: Path,
    mute_ranges: list[tuple[float, float]],
    music_end_time: float,
    music_resume_time: float,
) -> None:
    duration = probe_duration(video)
    fade_out_start = max(0.0, duration - 2.0)
    mute_envelopes = []
    for mute_start, mute_end in mute_ranges:
        fade_start = max(0.0, mute_start - BGM_MUTE_RAMP_SECONDS)
        fade_end = mute_end + BGM_MUTE_RAMP_SECONDS
        mute_envelopes.append(
            f"if(lt(t,{fade_start:.6f}),1,"
            f"if(lt(t,{mute_start:.6f}),"
            f"({mute_start:.6f}-t)/{BGM_MUTE_RAMP_SECONDS:g},"
            f"if(lt(t,{mute_end:.6f}),0,"
            f"if(lt(t,{fade_end:.6f}),"
            f"(t-{mute_end:.6f})/{BGM_MUTE_RAMP_SECONDS:g},1))))"
        )
    combined_envelope = "1"
    for envelope in mute_envelopes:
        combined_envelope = f"min({combined_envelope},{envelope})"
    music_fade_start = max(0.0, music_end_time - BGM_MUTE_RAMP_SECONDS)
    chapter_envelope = (
        f"if(lt(t,{music_fade_start:.6f}),1,"
        f"if(lt(t,{music_end_time:.6f}),"
        f"({music_end_time:.6f}-t)/{BGM_MUTE_RAMP_SECONDS:g},"
        f"if(lt(t,{music_resume_time:.6f}),0,"
        f"if(lt(t,{music_resume_time + FINAL_CROSSFADE_SECONDS:.6f}),"
        f"(t-{music_resume_time:.6f})/{FINAL_CROSSFADE_SECONDS:g},1))))"
    )
    combined_envelope = f"min({combined_envelope},{chapter_envelope})"
    music_volume = f"0.18*({combined_envelope})"
    audio_filter = (
        f"[0:a]volume=1[original];"
        f"[1:a]atrim=start={BGM_START_OFFSET_SECONDS:g}:duration={duration:.6f},"
        "asetpts=PTS-STARTPTS,"
        f"afade=t=out:st={fade_out_start:.6f}:d=2,"
        f"volume='{music_volume}':"
        "eval=frame[music];"
        "[original][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95[aout]"
    )
    run(
        [
            "ffmpeg", "-y", "-i", str(video), "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", audio_filter, "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "32000",
            "-t", f"{duration:.6f}", "-movflags", "+faststart", str(output),
        ]
    )


def validate(music: Path | None = None) -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required")
    asset_paths = [FONT, FONT_BOLD, WORDMARK, PKU_LOGO, NJU_LOGO, REF2VA_MANIFEST]
    if music is not None:
        asset_paths.append(music)
    for path in asset_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    for _, case_key, _, _, _ in CASES:
        for method in ("dense", "sol", "lite_fused"):
            source_video(case_key, method)
    for _, filename, _ in LIGHTX2V_CASES:
        source = LIGHTX2V_GALLERY / filename
        if not source.exists():
            raise FileNotFoundError(source)
    ref2va = json.loads(REF2VA_MANIFEST.read_text())
    for record in ref2va["variants"]:
        source = REF2VA_GALLERY / Path(record["preview"]).name
        if not source.exists():
            raise FileNotFoundError(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "assets" / "spark-h3-demo.mp4",
    )
    parser.add_argument(
        "--music", type=Path,
        help="optional licensed music file; media is intentionally not stored in Git",
    )
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    validate(args.music)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="spark-h3-demo-", dir=args.output.parent))
    try:
        clips: list[Path] = []
        intro = work / "00-intro.mp4"
        render_intro(intro)
        clips.append(intro)
        for position, (
            display_number, case_key, slow_ranges, hold_seconds,
            hold_detail_region,
        ) in enumerate(CASES, 1):
            clip = work / f"{position:02d}-case-{display_number}.mp4"
            render_comparison_clip(
                display_number, case_key, slow_ranges,
                hold_seconds, hold_detail_region, position, clip,
            )
            clips.append(clip)
        transition = work / f"{len(clips):02d}-lora-transition.mp4"
        render_lora_transition(transition)
        clips.append(transition)
        first_lora_case_index = len(clips)
        mute_clip_indices: list[int] = []
        for offset, (case_number, filename, latency) in enumerate(LIGHTX2V_CASES, len(clips)):
            clip = work / f"{offset:02d}-lightx2v-{case_number}.mp4"
            render_lightx2v_clip(case_number, filename, latency, clip)
            clips.append(clip)
            mute_clip_indices.append(len(clips) - 1)
        ref2va_transition = work / f"{len(clips):02d}-ref2va-transition.mp4"
        render_ref2va_transition(ref2va_transition)
        clips.append(ref2va_transition)
        ref2va_index = len(clips)
        ref2va_clip = work / f"{ref2va_index:02d}-ref2va-official-case.mp4"
        render_ref2va_clip(ref2va_clip)
        clips.append(ref2va_clip)
        mute_clip_indices.append(ref2va_index)
        outro = work / f"{len(clips):02d}-outro.mp4"
        render_outro(outro)
        clips.append(outro)
        durations, starts = concatenate_with_transitions(clips, args.output)
        mute_ranges = [
            (
                starts[index],
                starts[index] + durations[index] - (
                    FINAL_CROSSFADE_SECONDS
                    if index == len(clips) - 2
                    else CROSSFADE_SECONDS
                ),
            )
            for index in mute_clip_indices
        ]
        if args.music is not None:
            music = args.music.expanduser().resolve()
            suffix = "bgm"
            variant = args.output.with_name(
                f"{args.output.stem}-{suffix}{args.output.suffix}"
            )
            add_background_music(
                args.output, music, variant, mute_ranges,
                starts[first_lora_case_index], starts[-1],
            )
            print(f"Wrote {variant}")
        print(f"Wrote {args.output}")
    finally:
        if args.keep_work:
            print(f"Kept work directory: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
