"""GPU3 full sampler A/B of baseline Spark and experimental BSA producer path.

Each mode gets a fresh ComfyUI process. The candidate switch is supplied via
--experiment-env and is recorded verbatim in the protocol. This runner does
not modify production code or the comfy-kitchen binary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import traceback

import comfyui_latest_spark_4prompt_20260926 as trial


ROOT = Path('/autodl-fs/data/h3_experiments/comfyui_bsa_spark_full_ab_20260928')
ARCHIVED = Path('/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json')
GPU = 3
PORT = 8493
SOURCES = (
    trial.REPO / 'comfyui_backend.py',
    trial.REPO / 'comfyui_nodes.py',
    trial.REPO / 'comfyui_reblock_plan.py',
    trial.old.KITCHEN / 'comfy_kitchen/backends/cuda/sage_attention/sol_attn_producer.cu',
    trial.old.KITCHEN / 'comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu',
    trial.old.KITCHEN / 'comfy_kitchen/backends/cuda/dlpack_bindings.cpp',
    trial.old.KITCHEN / 'comfy_kitchen/backends/cuda/_C.abi3.so',
)
trial.source.cases = lambda: trial.read(ARCHIVED)['cases']


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(mode: str, experiment_env: dict[str, str], marker: str) -> None:
    if mode not in ('baseline', 'candidate'):
        raise ValueError(mode)
    if mode == 'baseline' and experiment_env:
        raise ValueError('baseline must not have experimental environment variables')
    path = ROOT / f'gpu{GPU}' / mode
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'record.json').exists():
        raise FileExistsError(f'Existing measurement will not be overwritten: {path}')
    dirs = {name: path / name for name in ('input', 'temp', 'user', 'output')}
    for directory in dirs.values():
        directory.mkdir(exist_ok=True)
    hashes = {str(source): hash_file(source) for source in SOURCES}
    case = trial.cases()[0]
    protocol = {
        'mode': mode, 'gpu': GPU, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'case': case,
        'experiment_env': experiment_env, 'required_log_marker': marker,
        'settings': {'seconds': 10, 'width': 1344, 'height': 768,
                     'steps': 20, 'seed': 42, 'method': 'spark_block',
                     'topk_ratio': 0.1, 'ablation_mode': 'full',
                     'tail_granularity': 'block', 'video_tail_mode': 'dense',
                     'global_anchor_dtype': 'float32',
                     'warmup': 'excluded 5-step sampler',
                     'scope': 'SamplerCustomAdvanced full denoise; excludes load, decode and latent save'},
        'source_sha256': hashes,
    }
    trial.base.write_json(path / 'protocol.json', protocol)
    cmd = [trial.base.PYTHON, 'main.py', '--listen', '127.0.0.1', '--port', str(PORT),
           '--disable-auto-launch', '--disable-cuda-malloc', '--preview-method', 'none',
           '--cache-classic']
    for name, directory in dirs.items():
        cmd.extend([f'--{name}-directory', str(directory)])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPU),
           'H3_SPARK_DIRECT_OUTPUT': '1', 'HF_HUB_OFFLINE': '1',
           'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
           'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost',
           **experiment_env}
    for switch in ('H3_SPARK_BSA_MATERIALIZER', 'H3_SPARK_BSA_ORIGINAL_PRODUCER'):
        if switch not in experiment_env:
            env.pop(switch, None)
    log_path = path / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(cmd, cwd=trial.old.COMFY, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        trial.base.write_json(path / 'pid.json', {'pid': process.pid, 'port': PORT})
        try:
            trial.base.wait_server(PORT, timeout=600)
            warmup = path / 'warmup.safetensors'
            trial.base.queue_and_wait(PORT, trial.graph(10, 'spark_block', case, warmup, 5), '11')
            target = path / 'measured.safetensors'
            result = trial.base.queue_and_wait(PORT, trial.graph(10, 'spark_block', case, target), '11')
            result.pop('history', None)
            if not result.get('sampler_seconds') or result['sampler_seconds'] < 100:
                raise RuntimeError(f'cached or missing 20-evaluation sampler: {result}')
            end_hashes = {str(source): hash_file(source) for source in SOURCES}
            if end_hashes != hashes:
                raise RuntimeError('Implementation changed during sampler; result invalid')
            log.flush()
            marker_count = log_path.read_text(errors='replace').count(marker) if marker else None
            if marker and marker_count == 0:
                raise RuntimeError(f'Candidate branch marker absent: {marker}')
            trial.base.write_json(path / 'record.json', {
                **result, 'gpu': GPU, 'mode': mode,
                'sampler_seconds': result['sampler_seconds'],
                'latent_path': str(target), 'latent_sha256': hash_file(target),
                'branch_marker_count': marker_count,
                'source_sha256': hashes,
            })
            print('DONE', mode, result['sampler_seconds'], hash_file(target), flush=True)
        except Exception:
            trial.base.write_json(path / 'error.json', {'traceback': traceback.format_exc()})
            raise
        finally:
            trial.base.stop_servers([(mode, process, log)])


def summarize() -> None:
    import torch
    from safetensors.torch import load_file

    paths = {mode: ROOT / f'gpu{GPU}' / mode for mode in ('baseline', 'candidate')}
    records = {mode: trial.read(path / 'record.json') for mode, path in paths.items()}
    protocols = {mode: trial.read(path / 'protocol.json') for mode, path in paths.items()}
    for key in ('source_sha256', 'case', 'settings'):
        if protocols['baseline'][key] != protocols['candidate'][key]:
            raise RuntimeError(f'Paired runs differ in {key}')
    marker = protocols['candidate']['required_log_marker']
    if marker and (paths['baseline'] / 'server.log').read_text(errors='replace').count(marker):
        raise RuntimeError('Candidate branch marker found in baseline log')
    seconds = {mode: record['sampler_seconds'] for mode, record in records.items()}
    latent = {mode: load_file(str(path / 'measured.safetensors'), device='cpu')
              for mode, path in paths.items()}
    if latent['baseline'].keys() != latent['candidate'].keys():
        raise RuntimeError('Latent keys differ')
    errors = {}
    for key in latent['baseline']:
        a, b = latent['baseline'][key], latent['candidate'][key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f'Latent tensor contract differs: {key}')
        different = torch.count_nonzero(a != b).item()
        if a.is_floating_point():
            delta = (a.float() - b.float()).abs()
            errors[key] = {
                'shape': list(a.shape), 'dtype': str(a.dtype),
                'different_elements': different, 'elements': a.numel(),
                'max_abs': delta.max().item(), 'mean_abs': delta.mean().item(),
            }
        else:
            errors[key] = {
                'shape': list(a.shape), 'dtype': str(a.dtype),
                'different_elements': different, 'elements': a.numel(),
            }
    order = sorted(paths, key=lambda mode: (paths[mode] / 'protocol.json').stat().st_mtime_ns)
    trial.base.write_json(ROOT / f'gpu{GPU}' / 'summary.json', {
        'status': 'complete', 'gpu': GPU, 'order': order,
        'sampler_seconds': seconds,
        'delta_seconds_candidate_minus_baseline': seconds['candidate'] - seconds['baseline'],
        'speedup_baseline_over_candidate': seconds['baseline'] / seconds['candidate'],
        'latent_sha256': {mode: record['latent_sha256'] for mode, record in records.items()},
        'latent_error': errors,
        'candidate_branch_marker_count': records['candidate']['branch_marker_count'],
        'source_sha256': {mode: record['source_sha256'] for mode, record in records.items()},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'candidate', 'summarize'))
    parser.add_argument('--experiment-env', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--required-log-marker', default='')
    args = parser.parse_args()
    if args.mode == 'summarize':
        summarize()
        return
    experiment_env = dict(item.split('=', 1) for item in args.experiment_env)
    if args.mode == 'candidate' and not experiment_env:
        parser.error('candidate requires at least one --experiment-env')
    run(args.mode, experiment_env, args.required_log_marker)


if __name__ == '__main__':
    main()
