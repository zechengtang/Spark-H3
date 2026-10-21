#!/usr/bin/env python3
"""Run the architecture-specific LightX2V Dense/Fidelity/Speed benchmark."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import requests
import websocket

from config import ARCH, CAPABILITY, ELF_ARCH, NODE, ACTIVATION, RESULTS, WHEEL, PORT as CONFIG_PORT, STEPS, LORA_NAME, PLATFORM, EXTENSION_SUFFIX


KIT = Path(__file__).resolve().parent
COMFY = Path(os.environ["COMFYUI_ROOT"]).resolve()
PYTHON = Path(sys.executable).absolute()
WORKFLOW = KIT / "lightx2v_workflow.json"
ROOT = RESULTS
OUTPUTS = ROOT / "outputs"
COMFY_OUTPUT = ROOT / "comfy_output"
RUNTIME_SITE = ROOT / "runtime_site"
RUNTIME_BASE = ROOT / "runtime_base"
PLUGIN_SOURCE = KIT / "ComfyUI-Spark-H3"
SOURCE_WHEEL = WHEEL
PORT = CONFIG_PORT
SAMPLER_NODE = "125"
SAVE_NODE = "92"
if SOURCE_WHEEL is None:
    raise RuntimeError("SPARK_WHEEL is missing; start this runner through validate.py")
_VERSION_MATCH = re.search(r"comfy_kitchen-(.+?)-cp\d+", SOURCE_WHEEL.name)
if _VERSION_MATCH is None:
    raise RuntimeError(f"cannot parse comfy-kitchen version from {SOURCE_WHEEL.name}")
EXPECTED_BACKEND = _VERSION_MATCH.group(1)
FIDELITY_WARMUP_STEPS = min(STEPS - 1, math.ceil(STEPS * 0.2))
TIERS = (
    {"name": "dense", "mode": "dense", "warmup_steps": 0, "warmup_evaluations": 1},
    {"name": "fidelity", "mode": "fidelity", "warmup_steps": FIDELITY_WARMUP_STEPS,
     "warmup_evaluations": FIDELITY_WARMUP_STEPS + 1},
    {"name": "speed", "mode": "speed", "warmup_steps": 0, "warmup_evaluations": 1},
)


def runtime_environment() -> dict[str, str]:
    current = os.environ.get("PYTHONPATH", "")
    entries = [str(RUNTIME_SITE)]
    if current:
        entries.append(current)
    pythonpath = os.pathsep.join(entries)
    return {**os.environ, "PYTHONPATH": pythonpath}


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        args, cwd=cwd, text=True, encoding="utf-8", errors="replace",
        stderr=subprocess.STDOUT,
        env=runtime_environment(),
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def workflow_to_prompt(path: Path) -> dict:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        if node.get("mode", 0) == 4 or node["type"] == "MarkdownNote":
            continue
        inputs = copy.deepcopy(node.get("widgets_values_named", {}))
        for node_input in node.get("inputs", []):
            link_id = node_input.get("link")
            if link_id is None:
                continue
            link = links[link_id]
            origin_id, origin_slot = link[1], link[2]
            origin = nodes[origin_id]
            if origin.get("mode", 0) == 4 or origin["type"] == "MarkdownNote":
                continue
            inputs[node_input["name"]] = [str(origin_id), origin_slot]
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt


BASE_PROMPT = workflow_to_prompt(WORKFLOW)


def formal_prompt(tier: dict, run_token: str) -> dict:
    prompt = copy.deepcopy(BASE_PROMPT)
    prompt["141"]["class_type"] = NODE
    prompt["127"]["inputs"].update(
        unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        weight_dtype="default",
    )
    prompt["142"]["inputs"].update(
        lora_name=LORA_NAME,
        strength_model=1.0,
    )
    prompt["119"]["inputs"]["vae_name"] = "minimax_h3_video_vae_fp16.safetensors"
    prompt["120"]["inputs"]["vae_name"] = "minimax_h3_audio_vae_fp32.safetensors"
    prompt["128"]["inputs"].update(
        clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        type="minimax",
        device="default",
    )
    prompt["131"]["inputs"].update(width=1344, height=768, length=345)
    prompt["129"]["inputs"].update(noise_seed=42, fixed="fixed")
    prompt["123"]["inputs"]["sampler_name"] = "res_multistep"
    prompt["124"]["inputs"].update(scheduler="simple", steps=STEPS, denoise=1.0)
    prompt["130"]["inputs"]["fps"] = 24
    prompt["141"]["inputs"].update(
        mode=tier["mode"],
        steps=STEPS,
        warmup_mode="warmup_steps",
        warmup_ratio=0.2 if tier["mode"] == "fidelity" else 0.0,
        warmup_steps=tier["warmup_steps"],
        topk_mode="topk_ratio",
        topk_ratio=0.20,
        topk_blocks=228,
        dense_layers=0,
        min_tokens=12288,
        strict=True,
        dense_backend="inherit",
        tail_granularity="query",
        reblock_layout="q_reuse_k",
    )
    prompt["141"]["inputs"].pop("enabled", None)
    prompt[SAVE_NODE]["inputs"]["filename_prefix"] = (
        f"{ARCH}_lightx2v_three_variants/{run_token}_{tier['name']}"
    )
    return prompt


def warmup_prompt(formal: dict, tier: dict, run_token: str) -> dict:
    prompt = copy.deepcopy(formal)
    evaluations = tier["warmup_evaluations"]
    prompt["124"]["inputs"]["steps"] = evaluations
    prompt["141"]["inputs"]["steps"] = evaluations
    # Deliberately retain the measured variant's Spark warmup_steps.
    prompt["141"]["inputs"]["warmup_steps"] = tier["warmup_steps"]
    prompt[SAVE_NODE]["inputs"]["filename_prefix"] = (
        f"excluded_warmup/{run_token}_{tier['name']}"
    )
    return prompt


def read_process_mib(pid: int) -> int:
    try:
        output = command(
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    maximum = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == pid:
            try:
                maximum = max(maximum, int(fields[1]))
            except ValueError:
                pass
    return maximum


class PeakMonitor:
    def __init__(self, pid: int):
        self.pid = pid
        self.peak_mib = 0
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak_mib = max(self.peak_mib, read_process_mib(self.pid))
            self.samples += 1
            self.stop_event.wait(0.10)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.peak_mib = max(self.peak_mib, read_process_mib(self.pid))


def wait_server(server=None) -> dict:
    deadline = time.monotonic() + 600
    last_error = None
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup with code {server.returncode}")
        try:
            response = requests.get(f"http://127.0.0.1:{PORT}/system_stats", timeout=3)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"ComfyUI did not start: {last_error}")


def history_record(prompt_id: str) -> dict:
    response = requests.get(f"http://127.0.0.1:{PORT}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()[prompt_id]


def output_files(record: dict) -> list[Path]:
    result = []
    for output in record.get("outputs", {}).values():
        for collection in output.values():
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                candidate = COMFY_OUTPUT / item.get("subfolder", "") / item["filename"]
                if candidate.suffix.lower() == ".mp4":
                    result.append(candidate)
    return result


def parse_execution_summary(log_text: str) -> dict | None:
    marker = "[Spark-H3] execution summary: "
    lines = [line[line.index(marker) + len(marker):] for line in log_text.splitlines() if marker in line]
    if not lines:
        return None
    line = lines[-1]
    prefix, fallback_text = line.rsplit(" unexpected_fallbacks=", 1)
    prefix, counts_text = prefix.rsplit(" counts=", 1)
    fields = dict(part.split("=", 1) for part in prefix.split())
    return {
        "evaluations": int(fields["evaluations"]),
        "warmup_evaluations": int(fields["warmup_evaluations"]),
        "counts": ast.literal_eval(counts_text),
        "unexpected_fallbacks": ast.literal_eval(fallback_text),
    }


def run_prompt(
    server: subprocess.Popen,
    prompt: dict,
    label: str,
    log_path: Path,
    *,
    interrupt_after_sampler: bool,
) -> dict:
    client_id = str(uuid.uuid4())
    ws = websocket.create_connection(
        f"ws://127.0.0.1:{PORT}/ws?clientId={client_id}",
        timeout=45,
        http_proxy_host=None,
    )
    log_offset = log_path.stat().st_size
    with PeakMonitor(server.pid) as memory:
        response = requests.post(
            f"http://127.0.0.1:{PORT}/prompt",
            json={"prompt": prompt, "client_id": client_id},
            timeout=30,
        )
        response.raise_for_status()
        submitted = response.json()
        if submitted.get("node_errors"):
            raise RuntimeError(f"node errors for {label}: {submitted['node_errors']}")
        prompt_id = submitted["prompt_id"]
        started = time.monotonic()
        sampler_started = None
        sampler_seconds = None
        terminal = None
        error_data = None
        interrupt_sent = False
        while terminal is None:
            if server.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with {server.returncode} during {label}")
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                print(f"HEARTBEAT {label} elapsed={time.monotonic() - started:.1f}s", flush=True)
                continue
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            data = event.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            event_type = event.get("type")
            if event_type == "executing":
                node = data.get("node")
                if sampler_started is not None and node != SAMPLER_NODE and sampler_seconds is None:
                    sampler_seconds = time.monotonic() - sampler_started
                    print(f"SAMPLER_END {label} seconds={sampler_seconds:.6f}", flush=True)
                    if interrupt_after_sampler:
                        interrupt = requests.post(
                            f"http://127.0.0.1:{PORT}/interrupt",
                            json={"prompt_id": prompt_id},
                            timeout=30,
                        )
                        interrupt.raise_for_status()
                        interrupt_sent = True
                if node == SAMPLER_NODE and sampler_started is None:
                    sampler_started = time.monotonic()
                    print(f"SAMPLER_START {label}", flush=True)
            elif event_type in ("execution_success", "execution_error", "execution_interrupted"):
                terminal = event_type
                if event_type == "execution_error":
                    error_data = data
        workflow_wall_seconds = time.monotonic() - started
    ws.close()

    record = history_record(prompt_id)
    status = record.get("status", {})
    expected_interrupt = (
        interrupt_after_sampler and interrupt_sent and terminal == "execution_interrupted"
    )
    if not expected_interrupt and (
        terminal != "execution_success" or status.get("status_str") != "success"
    ):
        raise RuntimeError(
            f"{label} failed: terminal={terminal} status={status} error={error_data}"
        )
    if sampler_seconds is None:
        raise RuntimeError(f"{label} never completed sampler node {SAMPLER_NODE}")

    timestamps = {
        kind: value.get("timestamp")
        for kind, value in status.get("messages", [])
        if isinstance(value, dict)
    }
    terminal_timestamp = timestamps.get("execution_success") or timestamps.get(
        "execution_interrupted"
    )
    history_seconds = None
    if timestamps.get("execution_start") is not None and terminal_timestamp is not None:
        history_seconds = (terminal_timestamp - timestamps["execution_start"]) / 1000.0

    # Give the file logger a moment to flush the sampler cleanup summary.
    time.sleep(0.2)
    with log_path.open("rb") as handle:
        handle.seek(log_offset)
        log_segment = handle.read().decode("utf-8", errors="replace")
    summary = parse_execution_summary(log_segment)
    return {
        "prompt_id": prompt_id,
        "sampler_seconds": sampler_seconds,
        "workflow_seconds": history_seconds if history_seconds is not None else workflow_wall_seconds,
        "workflow_wall_seconds": workflow_wall_seconds,
        "peak_cuda_memory_mib": memory.peak_mib,
        "peak_memory_measurement": "nvidia-smi per-process used_gpu_memory sampled every 100 ms",
        "terminal": terminal,
        "execution_summary": summary,
        "log_segment": log_segment,
        "output_files": [str(path) for path in output_files(record)],
    }


def inspect_video(path: Path) -> dict:
    raw = command(
        os.environ.get("FFPROBE", "ffprobe"), "-v", "error", "-count_frames", "-show_entries",
        "stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    )
    streams = json.loads(raw)["streams"]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise RuntimeError(f"expected one video stream in {path}, got {len(videos)}")
    video = videos[0]
    result = {
        "frames": int(video["nb_read_frames"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video.get("avg_frame_rate") or video["r_frame_rate"],
        "audio_streams": len(audio),
    }
    if result != {"frames": 345, "width": 1344, "height": 768, "fps": "24/1", "audio_streams": 1}:
        raise RuntimeError(f"unexpected media properties for {path}: {result}")
    return result


def make_comparison_video(videos: dict[str, Path]) -> dict:
    """Place the three formal outputs side by side, using Dense audio."""
    labels = ("Dense", "Fidelity", "Speed")
    names = ("dense", "fidelity", "speed")
    filters = [
        f"[{index}:v]drawtext=text='{label}':fontcolor=white:fontsize=42:"
        f"box=1:boxcolor=black@0.65:boxborderw=12:x=24:y=24[{name}]"
        for index, (name, label) in enumerate(zip(names, labels))
    ]
    filters.append("[dense][fidelity][speed]hstack=inputs=3[comparison]")
    destination = OUTPUTS / "dense_fidelity_speed_side_by_side.mp4"
    # Encode on local temporary storage before copying. Some mounted output
    # filesystems have yielded a truncated MP4 index during faststart remuxing.
    with tempfile.TemporaryDirectory(prefix="spark-h3-comparison-") as directory:
        temporary = Path(directory) / destination.name
        subprocess.run(
            [os.environ.get("FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error",
             "-nostdin", "-y",
             *(argument for name in names for argument in ("-i", str(videos[name]))),
             "-filter_complex", ";".join(filters),
             "-map", "[comparison]", "-map", "0:a:0",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
             str(temporary)],
            check=True,
        )
        media = inspect_comparison_video(temporary)
        shutil.copy2(temporary, destination)
    if inspect_comparison_video(destination) != media:
        raise RuntimeError(f"comparison video changed while copying: {destination}")
    result = {"path": str(destination), "sha256": sha256(destination), **media}
    print(f"COMPARISON_DONE {destination}", flush=True)
    return result


def inspect_comparison_video(path: Path) -> dict:
    raw = command(
        os.environ.get("FFPROBE", "ffprobe"), "-v", "error", "-count_frames",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    )
    streams = json.loads(raw)["streams"]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise RuntimeError(f"expected one comparison video stream in {path}")
    video = videos[0]
    media = {
        "frames": int(video["nb_read_frames"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": video["avg_frame_rate"],
        "audio_streams": len(audio),
        "layout": ["dense", "fidelity", "speed"],
    }
    expected = {
        "frames": 345, "width": 4032, "height": 768,
        "fps": "24/1", "audio_streams": 1,
        "layout": ["dense", "fidelity", "speed"],
    }
    if media != expected:
        raise RuntimeError(f"unexpected comparison media properties for {path}: {media}")
    return media


def environment(system_stats: dict, backend_state: dict) -> dict:
    wheel = next((ROOT / "wheelhouse").glob(f"comfy_kitchen-{EXPECTED_BACKEND}-*.whl"))
    extension = find_native_extension()
    gpu_fields = command(
        "nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ).split(",")
    return {
        "gpu": {
            "name": gpu_fields[0].strip(),
            "compute_capability": gpu_fields[1].strip(),
            "driver_version": gpu_fields[2].strip(),
            "memory_total_mib": int(gpu_fields[3].strip()),
        },
        "python": command(str(PYTHON), "--version"),
        "torch": command(str(PYTHON), "-c", "import torch; print(torch.__version__)"),
        "torch_cuda": command(str(PYTHON), "-c", "import torch; print(torch.version.cuda)"),
        "torch_capability": json.loads(command(
            str(PYTHON), "-c", "import torch,json; print(json.dumps(torch.cuda.get_device_capability()))"
        )),
        "nvcc": command("nvcc", "--version"),
        "comfyui": {
            "path": str(COMFY),
            "version": (COMFY / "comfyui_version.py").read_text(encoding="utf-8").split('"')[1],
            "system_stats": system_stats,
        },
        "comfy_kitchen": {
            "version": EXPECTED_BACKEND,
            "module": command(str(PYTHON), "-c", "import comfy_kitchen; print(comfy_kitchen.__file__)"),
            "backend_state": backend_state,
            "native_spark_attn": True,
            "extension": str(extension),
            "wheel": str(wheel),
            "wheel_sha256": sha256(wheel),
            "cuobjdump_list_elf": command("cuobjdump", "--list-elf", str(extension)).splitlines(),
        },
        "workflow": {
            "path": str(WORKFLOW),
            "sha256": sha256(WORKFLOW),
        },
        "assets": {
            "models_directory": str(COMFY / "models"),
            "model": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "lora": f"loras/{LORA_NAME}",
            "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
            "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        },
    }


def backend_state() -> dict:
    source = (
        "import json,importlib.metadata as m,comfy_kitchen as ck; "
        "from comfy_kitchen.backends import cuda; "
        "print(json.dumps({'version':m.version('comfy-kitchen'),"
        "'has_spark_attn':hasattr(cuda._C,'spark_attn'),'backends':ck.list_backends()}))"
    )
    return json.loads(command(str(PYTHON), "-c", source))


def validate_static(state: dict) -> None:
    if state["version"] != EXPECTED_BACKEND:
        raise RuntimeError(f"wrong comfy-kitchen identity: {state['version']}")
    if not state["has_spark_attn"]:
        raise RuntimeError("wheel lacks native _C.spark_attn")
    cuda = state["backends"].get("cuda", {})
    if not cuda.get("available") or cuda.get("disabled"):
        raise RuntimeError(f"CUDA backend is not active: {cuda}")
    if command(str(PYTHON), "-c", "import torch; print(torch.version.cuda)") != "13.0":
        raise RuntimeError("torch.version.cuda is not 13.0")
    capability = command(
        str(PYTHON), "-c", "import torch; print(torch.cuda.get_device_capability())"
    )
    if capability != str(CAPABILITY):
        raise RuntimeError(f"GPU capability is not {CAPABILITY}: {capability}")


def find_native_extension() -> Path:
    candidates = list((RUNTIME_SITE / "comfy_kitchen/backends/cuda").glob(f"_C*{EXTENSION_SUFFIX}"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {PLATFORM} native extension, found {candidates}")
    return candidates[0]


def prepare_isolated_runtime() -> dict:
    if not SOURCE_WHEEL.is_file():
        raise FileNotFoundError(f"missing {ARCH} wheel: {SOURCE_WHEEL}")
    if not PLUGIN_SOURCE.is_dir():
        raise FileNotFoundError(f"missing {ARCH} plugin: {PLUGIN_SOURCE}")

    shutil.rmtree(RUNTIME_SITE, ignore_errors=True)
    shutil.rmtree(RUNTIME_BASE, ignore_errors=True)
    RUNTIME_SITE.mkdir(parents=True)
    plugin_target = RUNTIME_BASE / "custom_nodes/ComfyUI-Spark-H3"
    plugin_target.parent.mkdir(parents=True)
    shutil.copytree(PLUGIN_SOURCE, plugin_target)
    with zipfile.ZipFile(SOURCE_WHEEL) as archive:
        archive.extractall(RUNTIME_SITE)

    wheelhouse = ROOT / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    wheel_copy = wheelhouse / SOURCE_WHEEL.name
    shutil.copy2(SOURCE_WHEEL, wheel_copy)

    extension = find_native_extension()
    elf_lines = command("cuobjdump", "--list-elf", str(extension)).splitlines()
    architectures = sorted({arch for line in elf_lines for arch in re.findall(r"sm_\d+", line)})
    if architectures != [ELF_ARCH]:
        raise RuntimeError(f"expected only {ELF_ARCH} cubins, found {architectures}")
    return {
        "wheel": str(wheel_copy),
        "wheel_sha256": sha256(wheel_copy),
        "extension": str(extension),
        "cubin_count": sum("ELF file" in line for line in elf_lines),
        "architectures": architectures,
    }


def protocol(run_token: str) -> dict:
    spark_common = {
        "strict": True,
        "topk_mode": "topk_ratio",
        "topk_ratio": 0.20,
        "dense_layers": 0,
        "min_tokens": 12288,
        "tail_granularity": "query",
        "reblock_layout": "q_reuse_k",
    }
    return {
        "status": "running",
        "run_token": run_token,
        "workflow": str(WORKFLOW),
        "execution_order": ["dense", "fidelity", "speed"],
        "comfyui_processes": 1,
        "process_policy": "reuse_one_server_with_short_warmup_per_variant",
        "comparison_video": {
            "layout": ["dense", "fidelity", "speed"],
            "filename": "outputs/dense_fidelity_speed_side_by_side.mp4",
            "audio_source": "dense",
        },
        "fixed": {
            "width": 1344, "height": 768, "frames": 345, "fps": 24,
            "steps": STEPS, "lora": LORA_NAME,
            "sampler": "res_multistep", "scheduler": "simple", "seed": 42,
            "prompt_sha256": hashlib.sha256(
                BASE_PROMPT["131"]["inputs"]["prompt"].encode()
            ).hexdigest(),
        },
        "variants": {
            "dense": {"mode": "dense", "steps": STEPS},
            "fidelity": {**spark_common, "mode": "fidelity", "warmup_mode": "warmup_steps", "warmup_ratio": 0.2, "warmup_steps": FIDELITY_WARMUP_STEPS, "steps": STEPS},
            "speed": {**spark_common, "mode": "speed", "warmup_mode": "warmup_steps", "warmup_ratio": 0.0, "warmup_steps": 0, "steps": STEPS},
        },
        "benchmark_warmup": {
            "excluded_from_measurement": True,
            "dense": {"evaluations": 1, "dense_evaluations": 1, "sparse_evaluations": 0},
            "fidelity": {"evaluations": FIDELITY_WARMUP_STEPS + 1,
                         "dense_evaluations": FIDELITY_WARMUP_STEPS,
                         "sparse_evaluations": 1,
                         "formal_warmup_steps_retained": FIDELITY_WARMUP_STEPS},
            "speed": {"evaluations": 1, "dense_evaluations": 0, "sparse_evaluations": 1, "formal_warmup_steps_retained": 0},
        },
        "timing": {
            "primary": "ComfyUI websocket SamplerCustomAdvanced node wall interval",
            "sampler_node": SAMPLER_NODE,
            "full_workflow": "ComfyUI history execution_start to terminal timestamp",
            "warmup_excluded": True,
        },
    }


def launch_server(env: dict[str, str], log_handle):
    extra = shlex.split(os.environ.get("SPARK_EXTRA_COMFY_ARGS", "--reserve-vram 1.5"))
    model_paths = COMFY / "extra_model_paths.yaml"
    command_line = [
        str(PYTHON), "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
        "--disable-auto-launch", "--preview-method", "none",
        "--base-directory", str(RUNTIME_BASE),
        "--models-directory", str(COMFY / "models"),
        "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-Spark-H3",
        "--output-directory", str(COMFY_OUTPUT),
        "--temp-directory", str(ROOT / "temp"),
        "--input-directory", str(ROOT / "input"),
        "--user-directory", str(ROOT / "user"),
        *(["--extra-model-paths-config", str(model_paths)] if model_paths.is_file() else []),
        *extra,
    ]
    return subprocess.Popen(
        command_line, cwd=COMFY, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
    )


def stop_server(server) -> None:
    if server is None or server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for directory in (COMFY_OUTPUT, ROOT / "temp", ROOT / "input", ROOT / "user"):
        directory.mkdir(parents=True, exist_ok=True)

    wheel_audit = prepare_isolated_runtime()
    write_json(ROOT / "wheel_audit.json", wheel_audit)
    state = backend_state()
    validate_static(state)
    run_token = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    protocol_data = protocol(run_token)
    write_json(ROOT / "protocol.json", protocol_data)

    log_path = ROOT / "comfyui.log"
    log_handle = log_path.open("w")
    env = {
        **runtime_environment(),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    server = launch_server(env, log_handle)
    records = {}
    try:
        stats = wait_server(server)
        print(f"SERVER_READY pid={server.pid} port={PORT}", flush=True)
        write_json(ROOT / "environment.json", environment(stats, state))

        for tier in TIERS:
            name = tier["name"]
            formal = formal_prompt(tier, run_token)
            warmup = warmup_prompt(formal, tier, run_token)
            if (warmup["124"]["inputs"]["steps"] != tier["warmup_evaluations"]
                    or warmup["141"]["inputs"]["steps"] != tier["warmup_evaluations"]):
                raise RuntimeError(f"{name} short warmup step count does not match protocol")
            print(f"WARMUP_START {name}", flush=True)
            warmup_result = run_prompt(
                server, warmup, f"{name}_benchmark_warmup", log_path,
                interrupt_after_sampler=True,
            )
            print(f"FORMAL_START {name}", flush=True)
            formal_result = run_prompt(
                server, formal, f"{name}_formal", log_path,
                interrupt_after_sampler=False,
            )

            if name == "dense":
                kernel_calls = 0
                formal_summary = None
            else:
                formal_summary = formal_result["execution_summary"]
                if formal_summary is None:
                    raise RuntimeError(f"missing Spark execution summary for {name}")
                expected_sparse_evaluations = STEPS - tier["warmup_steps"]
                expected_calls = expected_sparse_evaluations * 50
                kernel_calls = int(formal_summary["counts"].get("comfy_kitchen_calls", 0))
                native_calls = int(formal_summary["counts"].get("sparse:spark_comfy_native", 0))
                if formal_summary["evaluations"] != STEPS:
                    raise RuntimeError(f"{name} executed {formal_summary['evaluations']} evaluations")
                if formal_summary["warmup_evaluations"] != tier["warmup_steps"]:
                    raise RuntimeError(f"{name} warmup gate mismatch: {formal_summary}")
                if kernel_calls != expected_calls or native_calls != expected_calls:
                    raise RuntimeError(
                        f"{name} kernel calls {kernel_calls}/{native_calls}, expected {expected_calls}"
                    )
                if formal_summary["unexpected_fallbacks"]:
                    raise RuntimeError(f"{name} unexpected fallback: {formal_summary}")
                bad = [
                    text for text in ("dense after Spark failure", "[Spark-H3] dense fallback:")
                    if text in formal_result["log_segment"]
                ]
                if bad:
                    raise RuntimeError(f"{name} fallback markers: {bad}")
                if ACTIVATION not in formal_result["log_segment"]:
                    raise RuntimeError(f"{name} did not log {ARCH} activation")

            warmup_summary = warmup_result["execution_summary"]
            expected_warmup_sparse_calls = 0 if name == "dense" else 50
            actual_warmup_sparse_calls = 0
            if warmup_summary is not None:
                actual_warmup_sparse_calls = int(
                    warmup_summary["counts"].get("comfy_kitchen_calls", 0)
                )
            if actual_warmup_sparse_calls != expected_warmup_sparse_calls:
                raise RuntimeError(
                    f"{name} warmup sparse calls {actual_warmup_sparse_calls}, "
                    f"expected {expected_warmup_sparse_calls}"
                )

            candidates = [Path(path) for path in formal_result["output_files"]]
            if not candidates:
                candidates = sorted(
                    (COMFY_OUTPUT / f"{ARCH}_lightx2v_three_variants").glob(
                        f"{run_token}_{name}*.mp4"
                    )
                )
            candidates = [path for path in candidates if path.is_file()]
            if len(candidates) != 1:
                raise RuntimeError(f"expected one {name} MP4, got {candidates}")
            final_video = OUTPUTS / f"{name}.mp4"
            shutil.copy2(candidates[0], final_video)
            media = inspect_video(final_video)

            unexpected = []
            for segment in (warmup_result["log_segment"], formal_result["log_segment"]):
                for marker in ("dense after Spark failure", "[Spark-H3] dense fallback:"):
                    if marker in segment and marker not in unexpected:
                        unexpected.append(marker)
            record = {
                "valid": not unexpected,
                "variant": name,
                "comfyui_pid": server.pid,
                "formal_configuration": formal["141"]["inputs"],
                "warmup": {
                    "excluded_from_measurement": True,
                    "method": f"same-shape {tier['warmup_evaluations']}-evaluation sampler call",
                    "evaluations": tier["warmup_evaluations"],
                    "dense_evaluations": (
                        tier["warmup_evaluations"] - (0 if name == "dense" else 1)
                    ),
                    "sparse_evaluations": 0 if name == "dense" else 1,
                    "sampler_seconds": warmup_result["sampler_seconds"],
                    "workflow_seconds": warmup_result["workflow_seconds"],
                    "kernel_calls": actual_warmup_sparse_calls,
                    "execution_summary": warmup_summary,
                },
                "formal": {
                    "sampler_seconds": formal_result["sampler_seconds"],
                    "workflow_seconds": formal_result["workflow_seconds"],
                    "peak_cuda_memory_mib": formal_result["peak_cuda_memory_mib"],
                    "peak_memory_measurement": formal_result["peak_memory_measurement"],
                    "kernel_calls": kernel_calls,
                    "execution_summary": formal_summary,
                    "prompt_id": formal_result["prompt_id"],
                },
                "backend": state,
                "spark_activation_logged": (
                    name == "dense" or ACTIVATION in formal_result["log_segment"]
                ),
                "unexpected_fallbacks": unexpected,
                "output": {
                    "path": str(final_video),
                    "sha256": sha256(final_video),
                    **media,
                },
                "log_summary": [
                    line for line in formal_result["log_segment"].splitlines()
                    if "Spark-H3" in line or "comfy-kitchen" in line
                ],
            }
            write_json(ROOT / f"{name}.json", record)
            records[name] = record
            print(
                f"FORMAL_DONE {name} sampler={record['formal']['sampler_seconds']:.6f}s "
                f"workflow={record['formal']['workflow_seconds']:.6f}s "
                f"kernel_calls={kernel_calls} peak_mib={record['formal']['peak_cuda_memory_mib']}",
                flush=True,
            )
        full_log = log_path.read_text(encoding="utf-8", errors="replace")
        required_logs = (ACTIVATION,)
        missing_logs = [marker for marker in required_logs if marker not in full_log]
        unexpected_fallbacks = []
        for marker in ("dense after Spark failure", "[Spark-H3] dense fallback:"):
            if marker in full_log:
                unexpected_fallbacks.append(marker)
        comparison = make_comparison_video({
            name: Path(records[name]["output"]["path"])
            for name in ("dense", "fidelity", "speed")
        })
        cuda = state["backends"]["cuda"]
        dense = records["dense"]["formal"]["sampler_seconds"]
        fidelity = records["fidelity"]["formal"]["sampler_seconds"]
        speed = records["speed"]["formal"]["sampler_seconds"]
        valid = (
            not missing_logs
            and not unexpected_fallbacks
            and cuda["available"]
            and not cuda["disabled"]
            and records["fidelity"]["formal"]["kernel_calls"] == (STEPS - FIDELITY_WARMUP_STEPS) * 50
            and records["speed"]["formal"]["kernel_calls"] == STEPS * 50
        )
        summary = {
            "valid": valid,
            "gpu": command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").strip(),
            "torch_cuda": command(str(PYTHON), "-c", "import torch; print(torch.version.cuda)"),
            "comfyui_pid": server.pid,
            "process_policy": "reuse_one_server_with_short_warmup_per_variant",
            "architecture": ARCH,
            "backend_version": EXPECTED_BACKEND,
            "steps": STEPS,
            "lora": LORA_NAME,
            "wheel_audit": wheel_audit,
            "dense_seconds": dense,
            "fidelity_seconds": fidelity,
            "speed_seconds": speed,
            "fidelity_speedup": dense / fidelity,
            "speed_speedup": dense / speed,
            "fidelity_saved_percent": (1.0 - fidelity / dense) * 100.0,
            "speed_saved_percent": (1.0 - speed / dense) * 100.0,
            "unexpected_fallbacks": unexpected_fallbacks,
            "missing_required_logs": missing_logs,
            "cuda_backend": cuda,
            "kernel_calls": {
                name: records[name]["formal"]["kernel_calls"] for name in records
            },
            "outputs": {
                name: records[name]["output"] for name in records
            },
            "comparison_video": comparison,
        }
        write_json(ROOT / "summary.json", summary)
        speedup_report = {
            "valid": valid,
            "basis": "formal_sampler_seconds",
            "warmup_excluded": True,
            "dense_seconds": dense,
            "fidelity": {
                "seconds": fidelity,
                "speedup_vs_dense": dense / fidelity,
                "time_saved_percent": (1.0 - fidelity / dense) * 100.0,
            },
            "speed": {
                "seconds": speed,
                "speedup_vs_dense": dense / speed,
                "time_saved_percent": (1.0 - speed / dense) * 100.0,
            },
        }
        write_json(ROOT / "speedup_report.json", speedup_report)
        protocol_data["status"] = "complete" if valid else "invalid"
        protocol_data["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(ROOT / "protocol.json", protocol_data)
        if not valid:
            raise RuntimeError(f"benchmark validation failed: {summary}")
        print(
            f"SPEEDUP_VS_DENSE fidelity={dense / fidelity:.3f}x "
            f"speed={dense / speed:.3f}x "
            f"basis=formal_sampler_seconds warmup_excluded=true "
            f"report={ROOT / 'speedup_report.json'}",
            flush=True,
        )
        print("ALL_VALID", flush=True)
    except BaseException as error:
        write_json(ROOT / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        raise
    finally:
        stop_server(server)
        log_handle.close()


if __name__ == "__main__":
    main()
