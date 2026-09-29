"""Real ComfyUI QKV single sparse evaluation smoke on GPU2."""

import json
import os
from pathlib import Path
import subprocess

import comfyui_latest_spark_4prompt_20260926 as trial


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_bsa_materializer_smoke_20260928")
ARCHIVED = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
trial.source.cases = lambda: trial.read(ARCHIVED)["cases"]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("input", "temp", "user", "output"):
        (ROOT / name).mkdir(exist_ok=True)
    port = 8532
    log = (ROOT / "server.log").open("a")
    command = [trial.base.PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(port),
               "--disable-auto-launch", "--disable-cuda-malloc", "--preview-method", "none",
               "--cache-classic"]
    for name in ("input", "temp", "user", "output"):
        command.extend([f"--{name}-directory", str(ROOT / name)])
    verify = ROOT / "real_qkv_parity.json"
    proc = subprocess.Popen(command, cwd=trial.old.COMFY, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "2",
                                 "H3_SPARK_BSA_MATERIALIZER": "1",
                                 "H3_SPARK_BSA_VERIFY_PATH": str(verify),
                                 "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                                 "OMP_NUM_THREADS": "4", "NO_PROXY": "127.0.0.1,localhost",
                                 "no_proxy": "127.0.0.1,localhost"})
    try:
        trial.base.wait_server(port, timeout=600)
        case = trial.cases()[0]
        output = ROOT / "warmup.safetensors"
        record = trial.base.queue_and_wait(
            port, trial.graph(10, "spark_block", case, output, 5), "11")
        record.pop("history", None)
        (ROOT / "result.json").write_text(json.dumps({
            "record": record, "case": case["index"],
            "verify_exists": verify.exists(),
            "verify": json.loads(verify.read_text()) if verify.exists() else None,
        }, indent=2))
        if not verify.exists():
            raise RuntimeError("Sparse attention was not exercised in five-step smoke")
        print(verify.read_text(), flush=True)
    finally:
        trial.base.stop_servers([("gpu2-smoke", proc, log)])


if __name__ == "__main__":
    main()
