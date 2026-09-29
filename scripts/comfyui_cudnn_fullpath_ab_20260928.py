"""Isolated ComfyUI H3 dense-SDPA backend A/B on a Spark full sampler path.

The runner builds a separate ComfyUI source copy and patches only its SDPA
priority for the exact H3 10s packed attention shape. Production files remain
untouched. Requires CUDA_VISIBLE_DEVICES=0 at server launch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import comfyui_latest_spark_4prompt_20260926 as trial


ROOT = Path('/autodl-fs/data/h3_experiments/comfyui_cudnn_verified_20260928')
COMFY = Path('/autodl-fs/data/h3_repos/ComfyUI')
COPY = ROOT / 'ComfyUI_experimental_copy'
MODES = ('flash', 'cudnn')
ARCHIVED = Path('/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json')
trial.source.cases = lambda: trial.read(ARCHIVED)['cases']
PAIR = int(os.environ.get('H3_CUDNN_PAIR', '0'))
if PAIR not in (0, 1):
    raise ValueError('H3_CUDNN_PAIR must be 0 or 1')
PAIR_MODES = MODES if PAIR == 0 else MODES[::-1]
PAIR_ROOT = ROOT / ('fullpath_gpu0_10s' if PAIR == 0 else 'fullpath_gpu0_10s_reverse')


def prepare_copy():
    if COPY.exists():
        return
    shutil.copytree(COMFY, COPY, symlinks=True)
    ops = COPY / 'comfy/ops.py'
    source = ops.read_text()
    old = '                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):'
    new = '''                # Experiment only: force cuDNN for H3 10s packed dense attention.
                import os
                h3_cudnn = (os.environ.get('H3_EXPERIMENT_DENSE_BACKEND') == 'cudnn'
                            and q.shape == (1, 56, 73573, 128)
                            and q.dtype == torch.bfloat16 and attn_mask is None)
                if h3_cudnn and not globals().get('_H3_CUDNN_BRANCH_LOGGED', False):
                    logging.warning('H3 experiment cuDNN branch hit: shape=%s dtype=%s strides=%s',
                                    tuple(q.shape), q.dtype, tuple(q.stride()))
                    globals()['_H3_CUDNN_BRANCH_LOGGED'] = True
                backend_priority = [SDPBackend.CUDNN_ATTENTION] if h3_cudnn else SDPA_BACKEND_PRIORITY
                with sdpa_kernel(backend_priority, set_priority=True):'''
    if source.count(old) != 1:
        raise RuntimeError('ComfyUI ops.py source differs from expected patch point')
    ops.write_text(source.replace(old, new))
    (ROOT / 'ops_patch.diff').write_text('Replaced one SDPA priority line in experiment copy; see comfy/ops.py.\n')


def run_mode(mode):
    path = PAIR_ROOT / mode
    path.mkdir(parents=True, exist_ok=True)
    port = 8490
    dirs = {label: path / label for label in ('input', 'temp', 'user', 'output')}
    for folder in dirs.values():
        folder.mkdir(exist_ok=True)
    log = (path / 'server.log').open('a')
    cmd = [trial.base.PYTHON, 'main.py', '--listen', '127.0.0.1', '--port', str(port),
           '--disable-auto-launch', '--disable-cuda-malloc', '--preview-method', 'none',
           '--cache-classic']
    for label, folder in dirs.items():
        cmd.extend([f'--{label}-directory', str(folder)])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'H3_EXPERIMENT_DENSE_BACKEND': mode,
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
           'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost'}
    process = subprocess.Popen(cmd, cwd=COPY, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        trial.base.wait_server(port, timeout=600)
        case = trial.cases()[0]
        # Different number of steps invalidates ComfyUI's sampler cache.
        trial.base.queue_and_wait(port, trial.graph(10, 'spark_block', case, path / 'warmup.safetensors', 5), '11')
        trial.base.queue_and_wait(port, trial.graph(10, 'spark_block', case, path / 'pre_measure.safetensors', 5), '11')
        result = trial.base.queue_and_wait(port, trial.graph(10, 'spark_block', case, path / 'measured.safetensors'), '11')
        result.pop('history', None)
        if not result.get('sampler_seconds') or result['sampler_seconds'] < 100:
            raise RuntimeError(f'cached or missing full sampler: {result}')
        trial.base.write_json(path / 'timing.json', dict(result, mode=mode, gpu=0, seconds=10,
                                                       case=case['index']))
        print(mode, result['sampler_seconds'], flush=True)
    finally:
        trial.base.stop_servers([(mode, process, log)])


def main():
    ROOT.mkdir(exist_ok=True)
    prepare_copy()
    protocol = ROOT / ('fullpath_protocol.json' if PAIR == 0 else 'fullpath_protocol_reverse.json')
    if not protocol.exists():
        trial.base.write_json(protocol, dict(
            source_comfy=str(COMFY), experimental_copy=str(COPY),
            source_ops_sha256=trial.base.sha256(COMFY / 'comfy/ops.py'),
            patched_ops_sha256=trial.base.sha256(COPY / 'comfy/ops.py'),
            spark_plugin_sha256=trial.base.sha256(trial.REPO / 'comfyui_nodes.py'),
            case=trial.cases()[0], settings=dict(duration_s=10, width=1344, height=768,
            steps=20, seed=42, method='spark_block', gpu=0, modes=PAIR_MODES,
            backend_override='only exact H3 10s packed dense attention shape')))
    for mode in PAIR_MODES:
        if not (PAIR_ROOT / mode / 'timing.json').exists():
            run_mode(mode)


if __name__ == '__main__':
    main()
