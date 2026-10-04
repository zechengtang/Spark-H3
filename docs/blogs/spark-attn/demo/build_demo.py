#!/usr/bin/env python3
"""Build the Spark-H3 visual-comparison demo reel."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


GALLERY = Path("/autodl-fs/data/h3_outputs/video_gallery_formal_20260913")
LIGHTX2V_GALLERY = Path(
    "/autodl-fs/data/h3_experiments/vdn10_lightx2v_showcase_20260922/media"
)
REPO = Path(__file__).resolve().parents[4]
WORDMARK = REPO / "assets" / "spark-h3-wordmark.png"
PKU_LOGO = REPO / "assets" / "school_logos" / "PKU.png"
NJU_LOGO = REPO / "assets" / "school_logos" / "NJU.jpg"
MUSIC_DIR = REPO / "docs" / "blogs" / "spark-attn" / "demo" / "music_candidates"
BGM_VARIANTS = [
    ("bgm-08-its-love", MUSIC_DIR / "08-its-love-michael-ramir-c.mp3"),
]
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

FPS = 24
WIDTH, HEIGHT = 1920, 1080
COMPARISON_SECONDS = 10
SLOW_FACTOR = 4.0
DETAIL_GREEN = "#5f8f73"
DETAIL_RED = "#c95b52"
CROSSFADE_SECONDS = 0.5
CASE_HOLD_SECONDS = 0.75
ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "slow", "-crf", "14",
    "-maxrate", "24M", "-bufsize", "48M",
]

# (display number, sample id, gallery number, latencies, highlighted source ranges,
# opening hold, optional detail region shown during the opening hold)
# A highlighted range is (start, end, region).
CASES = [
    (
        "1", "0808", "21", (576.6, 342.3, 370.0),
        ((6.1, 7.1, "bottom_right"),), CASE_HOLD_SECONDS, None,
    ),
    (
        "2", "0285", "06", (576.9, 342.4, 362.4),
        ((1.0, 2.0, "bottom_right_up"), (5.1, 7.1, "center_left")),
        CASE_HOLD_SECONDS, None,
    ),
    (
        "3", "0685", "32", (580.0, 344.2, 363.1),
        ((0.0, 1.0, "bottom_right"),), CASE_HOLD_SECONDS, None,
    ),
]

LIGHTX2V_CASES = [
    ("4", "vdn_0001_japanese_woman_city_closeup.mp4", 262.2),
    ("5", "vdn_0010_2d_anime_eye_pullout.mp4", 265.74),
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


def source_video(method: str, number: str, sample_id: str) -> Path:
    candidate = GALLERY / method / "videos" / f"{number}_vbench_all_{sample_id}.mp4"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def render_intro(output: Path) -> None:
    decorations = [
        drawtext(
            "Adaptive Block-Sparse Attention for MiniMax-H3",
            "(w-text_w)/2", "575", 34, "#5b626c",
        ),
        "drawbox=x=715:y=650:w=490:h=4:color=#ff6b00:t=fill",
        drawtext(
            "REBLOCK  ·  REWEIGHT  ·  HIGH SPARSITY",
            "(w-text_w)/2", "695", 25, "#d95300", bold=True,
        ),
    ]
    filters = (
        "[1:v]crop=1950:625:0:130,scale=1080:-2[wordmark];"
        "[2:v]scale=124:124[pku];"
        "[3:v]scale=-2:132[nju];"
        "[0:v][wordmark]overlay=(W-w)/2:190[stage1];"
        "[stage1][pku]overlay=815:825[stage2];"
        "[stage2][nju]overlay=1005:821[stage3];"
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
            "SPARK-H3 + COMMUNITY LoRAs", "(w-text_w)/2", "330", 25,
            "#d95300", bold=True,
        ),
        drawtext(
            "Spark-H3 can be integrated", "(w-text_w)/2", "420", 56,
            "#15181e", bold=True,
        ),
        drawtext(
            "with community LoRAs.", "(w-text_w)/2", "492", 56,
            "#15181e", bold=True,
        ),
        "drawbox=x=710:y=590:w=500:h=4:color=#ff6b00:t=fill",
        drawtext(
            "Here, we use the 8-step LoRA from LightX2V.",
            "(w-text_w)/2", "640", 29, "#5b626c",
        ),
        drawtext(
            "Thanks for the awesome community work!",
            "(w-text_w)/2", "725", 24, "#d95300", bold=True,
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
    box_width = panel_width // 2
    box_height = panel_height // 2
    if region in ("bottom_right", "bottom_right_up"):
        x = panel_x + panel_width - box_width
        y = panel_y + panel_height - box_height
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
    crop_width = panel_width // 2
    crop_height = panel_height // 2
    if region in ("bottom_right", "bottom_right_up"):
        y = panel_height - crop_height
        if region == "bottom_right_up":
            y -= crop_height // 5
        return crop_width, crop_height, panel_width - crop_width, y
    if region in ("center", "center_left"):
        x = (panel_width - crop_width) // 2
        if region == "center_left":
            x -= panel_width // 8
        return crop_width, crop_height, x, (panel_height - crop_height) // 2
    raise ValueError(f"Unknown highlight region: {region}")


def render_comparison_clip(
    display_number: str,
    sample_id: str,
    number: str,
    latencies: tuple[float, float, float],
    slow_ranges: tuple[tuple[float, float, str], ...],
    hold_seconds: float,
    hold_detail_region: str | None,
    position: int,
    output: Path,
) -> None:
    methods = ("dense", "spark_h3_10pct", "sol")
    inputs = [source_video(method, number, sample_id) for method in methods]
    panel_width, panel_height = 600, 342
    panel_y = 220
    detail_y = 680
    xs = [40, 660, 1280]
    labels = ["Dense", "Spark-H3", "Sol-H3"]
    dense_latency, spark_latency, sol_latency = latencies
    metrics = [
        "1.00×",
        f"{dense_latency / spark_latency:.2f}×",
        f"{dense_latency / sol_latency:.2f}×",
    ]
    output_duration = hold_seconds + COMPARISON_SECONDS + sum(
        (SLOW_FACTOR - 1) * (end - start) for start, end, _ in slow_ranges
    )

    decorations = [
        drawtext("SPARK-H3  /  VISUAL COMPARISON", "40", "32", 20, "#6b7078", bold=True),
        drawtext(f"Case {display_number}", "40", "70", 42, "#15181e", bold=True),
        drawtext(
            "10 s  ·  1344 × 768  ·  19 steps  ·  matched seed",
            "1880-text_w", "52", 21, "#5b626c",
        ),
    ]
    for index, (x, label, metric) in enumerate(zip(xs, labels, metrics)):
        label_color = "#d95300" if index == 1 else "#15181e"
        metric_color = "#d95300" if index == 1 else "#5b626c"
        decorations.extend(
            [
                drawtext(label, str(x), "170", 29, label_color, bold=True),
                drawtext(
                    metric, f"{x}+{panel_width}-text_w", "176", 22,
                    metric_color, bold=True,
                ),
            ]
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
        decorations.append(
            drawtext("2× DETAIL", "(w-text_w)/2", "630", 22, "#d95300", bold=True)
            + f":enable='{detail_expression}'"
        )
    for start, end, region in detail_ranges:
        region_expression = fr"between(t\,{start:g}\,{end:g})"
        for method_index, x in enumerate(xs):
            detail_color = DETAIL_RED if method_index == 2 else DETAIL_GREEN
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
        + retimed_video_filter(1, "spark_base", slow_ranges, panel_width, panel_height, hold_seconds)
        + retimed_video_filter(2, "sol_base", slow_ranges, panel_width, panel_height, hold_seconds)
    )
    detail_filters: list[str] = []
    for method in ("dense", "spark", "sol"):
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
        f"[tmp1][spark]overlay={xs[1]}:{panel_y}[tmp2];"
        f"[tmp2][sol]overlay={xs[2]}:{panel_y}[panels]"
    )
    previous = "panels"
    detail_index = 0
    for range_index, (start, end, _) in enumerate(detail_ranges):
        region_expression = fr"between(t\,{start:g}\,{end:g})"
        for method, x in zip(("dense", "spark", "sol"), xs):
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
        drawtext("8 steps  ·  90% sparse", "(w-text_w)/2", "78", 20, "#d95300", bold=True),
        drawtext("14.4 s  ·  1344 × 768", "1864-text_w", "55", 21, "#5b626c"),
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


def render_outro(output: Path) -> None:
    decorations = [
        drawtext("THANKS FOR WATCHING", "(w-text_w)/2", "555", 54, "#15181e", bold=True),
        "drawbox=x=760:y=640:w=400:h=4:color=#ff6b00:t=fill",
        drawtext(
            "Stay tuned for more Spark-H3 updates.",
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
        + ",fade=t=out:st=2.7:d=0.5,format=yuv420p[out]"
    )
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=white:s={WIDTH}x{HEIGHT}:r={FPS}:d=3.2",
            "-loop", "1", "-framerate", str(FPS), "-i", str(WORDMARK),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-filter_complex", filters, "-map", "[out]", "-map", "2:a",
            "-t", "3.2", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
            *ENCODE_ARGS, "-movflags", "+faststart", str(output),
        ]
    )


def concatenate_with_transitions(clips: list[Path], output: Path) -> tuple[list[float], list[float]]:
    durations = [probe_duration(clip) for clip in clips]
    starts = [0.0]
    for duration in durations[:-1]:
        starts.append(starts[-1] + duration - CROSSFADE_SECONDS)

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
        offset = timeline_duration - CROSSFADE_SECONDS
        filters.extend(
            [
                f"[{video_label}][v{index}]xfade=transition=fade:"
                f"duration={CROSSFADE_SECONDS:g}:offset={offset:.6f}[{video_out}]",
                f"[{audio_label}][a{index}]acrossfade=d={CROSSFADE_SECONDS:g}:"
                f"c1=tri:c2=tri[{audio_out}]",
            ]
        )
        video_label = video_out
        audio_label = audio_out
        timeline_duration += durations[index] - CROSSFADE_SECONDS

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
    duck_start: float,
    duck_end: float,
) -> None:
    duration = probe_duration(video)
    fade_out_start = max(0.0, duration - 2.0)
    audio_filter = (
        f"[0:a]volume=1[original];"
        f"[1:a]atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,"
        "afade=t=in:st=0:d=1.2,"
        f"afade=t=out:st={fade_out_start:.6f}:d=2,volume=0.18,"
        f"volume=0.28:enable='between(t,{duck_start:.6f},{duck_end:.6f})'[music];"
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


def validate() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is required")
    asset_paths = [FONT, FONT_BOLD, WORDMARK, PKU_LOGO, NJU_LOGO]
    asset_paths.extend(music for _, music in BGM_VARIANTS)
    for path in asset_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    for _, sample_id, number, _, _, _, _ in CASES:
        for method in ("dense", "sol", "spark_h3_10pct"):
            source_video(method, number, sample_id)
    for _, filename, _ in LIGHTX2V_CASES:
        source = LIGHTX2V_GALLERY / filename
        if not source.exists():
            raise FileNotFoundError(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/autodl-fs/data/h3_experiments/spark-h3-demo-20260923/spark-h3-demo.mp4"
        ),
    )
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="spark-h3-demo-", dir=args.output.parent))
    try:
        clips: list[Path] = []
        intro = work / "00-intro.mp4"
        render_intro(intro)
        clips.append(intro)
        for position, (
            display_number, sample_id, number, latencies, slow_ranges, hold_seconds,
            hold_detail_region,
        ) in enumerate(CASES, 1):
            clip = work / f"{position:02d}-case-{display_number}.mp4"
            render_comparison_clip(
                display_number, sample_id, number, latencies, slow_ranges,
                hold_seconds, hold_detail_region, position, clip,
            )
            clips.append(clip)
        transition = work / "04-lora-transition.mp4"
        render_lora_transition(transition)
        clips.append(transition)
        lightx2v_start_index = len(clips)
        for offset, (case_number, filename, latency) in enumerate(LIGHTX2V_CASES, len(clips)):
            clip = work / f"{offset:02d}-lightx2v-{case_number}.mp4"
            render_lightx2v_clip(case_number, filename, latency, clip)
            clips.append(clip)
        outro = work / "07-outro.mp4"
        render_outro(outro)
        clips.append(outro)
        durations, starts = concatenate_with_transitions(clips, args.output)
        duck_start = starts[lightx2v_start_index]
        last_lightx2v_index = lightx2v_start_index + len(LIGHTX2V_CASES) - 1
        duck_end = starts[last_lightx2v_index] + durations[last_lightx2v_index]
        for suffix, music in BGM_VARIANTS:
            variant = args.output.with_name(
                f"{args.output.stem}-{suffix}{args.output.suffix}"
            )
            add_background_music(args.output, music, variant, duck_start, duck_end)
            print(f"Wrote {variant}")
        print(f"Wrote {args.output}")
    finally:
        if args.keep_work:
            print(f"Kept work directory: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
