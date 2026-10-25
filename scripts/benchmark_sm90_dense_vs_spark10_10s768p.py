#!/usr/bin/env python3
"""Warmed H20 DiT benchmark using the same profile as the performance matrix.

The benchmark uses the production packed_external_no_route_qk policy.
Conditioning is prepared once outside timing and reused by both methods.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

import benchmark_sm80_dense_vs_spark10_10s768p as base
from h3_sparse_attention import H3SparseAttentionConfig, install_h3_sparse_attention
from h3_sparse_attention.acceleration import H3AccelerationConfig, install_h3_acceleration
from h3_sparse_attention.sol_numerator_virtual_q import fused_compile_cache_stats


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def optional_git_commit(path):
    """Return a source checkout commit without rejecting wheel installations."""
    result = subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare_conditioning(model, destination, prompt):
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager

    workflow = ModularPipeline.from_pretrained(str(model), local_files_only=True).blocks.get_workflow('t2va')
    pipe = workflow.sub_blocks['text_encoder'].init_pipeline(
        str(model), components_manager=ComponentsManager(),
    )
    pipe.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path={'default': str(model)}, local_files_only=True,
    )
    pipe.text_encoder.to('cuda')
    with torch.inference_mode():
        state = pipe(prompt=prompt)
    for key, value in list(state.values.items()):
        if isinstance(value, torch.Tensor):
            state.values[key] = value.detach().cpu()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    save(destination.with_suffix('.json'), {'prompt': prompt, 'model': str(model)})
    del state, pipe, workflow
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[3] / 'models/diffusers/MiniMax-H3')
    parser.add_argument('--conditioning', type=Path, required=True)
    parser.add_argument('--prepare-conditioning', action='store_true')
    parser.add_argument('--prompt', default='A cinematic shot of a small wooden fishing boat drifting across a calm lake at sunrise. Golden sunlight illuminates the mist above the water, mountains in the distance, gentle ripples and birdsong.')
    parser.add_argument(
        '--route-execution',
        choices=('packed_external_no_route_qk',),
        default='packed_external_no_route_qk',
    )
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError('H20/SM90 required')
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    if args.prepare_conditioning:
        if args.conditioning.exists():
            raise FileExistsError(args.conditioning)
        prepare_conditioning(args.model, args.conditioning, args.prompt)
        return
    base.MODEL, base.CONDITIONING = args.model, args.conditioning
    loaded = time.perf_counter()
    pipe = base.load_pipeline()
    load_seconds = time.perf_counter() - loaded
    if not args.conditioning.is_file():
        raise FileNotFoundError(args.conditioning)
    result = {
        'status': 'running', 'gpu': torch.cuda.get_device_name(),
        'capability': torch.cuda.get_device_capability(),
        'model': str(args.model), 'conditioning': str(args.conditioning),
        'resolution': [1344, 768], 'requested_frames': 243,
        'requested_steps': 20, 'transformer_evaluations': 19,
        'topk_ratio': 0.10, 'dense_warmup_evaluations': 4, 'dense_layers': 1,
        'landmark_direction': 'legacy', 'anchor_policy': 'global',
        'route_execution': args.route_execution, 'torch_compile': True,
        'compile_reproducibility_fix': False,
        'load_seconds': load_seconds, 'seed': 42,
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'triton', 'nvidia-cutlass-dsl', 'transformers', 'diffusers')},
        'diffusers_commit': optional_git_commit(Path(__import__('diffusers').__file__).parents[2]),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sm120_reference': {'dense_seconds': 583.9, 'spark_seconds': 338.7, 'speedup_x': 583.9 / 338.7, 'route_execution': 'threshold', 'source': 'docs/kernel_performance_matrix.md'},
        'runs': [],
    }
    save(args.output, result)
    try:
        with install_h3_acceleration(pipe, H3AccelerationConfig(compile_reproducibility_fix=False)):
            config = H3SparseAttentionConfig.spark(
                20, warmup_percent=20.0, sol_dense_layers=1,
                sol_route_topk_ratio=0.10, sol_route_topk_execution=args.route_execution,
                sol_log_density=False,
            )
            with install_h3_sparse_attention(pipe.transformer, config) as plugin:
                def spark(measured, steps=20):
                    before = fused_compile_cache_stats()
                    row = base.run(pipe, 'spark10', measured=measured,
                                   inference_steps=steps, installed_plugin=plugin,
                                   frames=243, route_execution=args.route_execution)
                    after = fused_compile_cache_stats()
                    row['new_fused_compile_calls'] = after['compile_calls'] - before['compile_calls']
                    assert row['finite'], row
                    result['runs'].append(row)
                    save(args.output, result)
                spark(False, 6)
                for _ in range(args.repeats):
                    base.reset_measurement_statistics(plugin)
                    spark(True)
            # Dense graphs have already executed during Spark's warmup. Record
            # one discarded dense pass as well to stabilize method transitions.
            for measured in [False] + [True] * args.repeats:
                row = base.run(pipe, 'dense', measured=measured,
                               inference_steps=6 if not measured else 20,
                               frames=243, route_execution=args.route_execution)
                assert row['finite'], row
                result['runs'].append(row)
                save(args.output, result)
        means = {method: statistics.mean(row['seconds'] for row in result['runs']
                   if row['method'] == method and row['phase'] == 'measured')
                 for method in ('dense', 'spark10')}
        result.update(status='complete', dense_seconds=means['dense'],
                      spark_seconds=means['spark10'], speedup_x=means['dense']/means['spark10'])
        result['sm120_speedup_ratio'] = result['speedup_x'] / result['sm120_reference']['speedup_x']
        save(args.output, result)
        print(json.dumps(result, indent=2, default=str), flush=True)
    except Exception as error:
        result.update(status='failed', error=repr(error))
        save(args.output, result)
        raise


if __name__ == '__main__':
    main()
