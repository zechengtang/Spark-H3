"""Per-timestep dense-importance ablation over the full trajectory: 5s/480p, GPUs 0-3.

Each arm `dense_eval_XX` (X in 4..18) keeps the uniform 10% Top-K baseline (22
blocks/head, exact per-call total 1,232) at every sparse evaluation except
evaluation XX, which additionally runs through the true production dense path
(config.sol_extra_dense_evaluations). Reverse arms `sparse_eval_0X` (X in 0..3)
instead REMOVE dense from warmup evaluation X (warmup_percent=0, the other
three early evaluations passed as sol_extra_dense_evaluations), measuring the
dense gain of the production warmup evaluations. This is an importance probe:
arms are not budget-matched to uniform. The uniform baseline and the dense
references are reused from the ramp_budget experiment after config and SHA256
verification. Stages: prepare/smoke/worker/finish/report/run.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, '/autodl-fs/data/h3_repos/Spark-H3-relative-difficulty/scripts')
import relative_difficulty_ablation as base

REPO = Path(__file__).resolve().parents[1]
MODEL = Path('/autodl-fs/data/models/MiniMax-H3')
EXP = Path('/autodl-fs/data/h3_experiments/dense_step_ablation_5s_480p_20260920')
OUT = Path('/autodl-fs/data/h3_outputs') / EXP.name
BASELINE = Path('/autodl-fs/data/h3_experiments/ramp_budget_5s_480p_20260920')
EXTRA_DENSE_EVALS = tuple(range(4, 19))
ARMS = tuple(f'dense_eval_{e:02}' for e in EXTRA_DENSE_EVALS)
REVERSE_EVALS = tuple(range(4))
REVERSE_ARMS = tuple(f'sparse_eval_{e:02}' for e in REVERSE_EVALS)
WORKER_ARMS = ('uniform',) + ARMS + REVERSE_ARMS
CASES = base.CASES
FRAMES, HEIGHT, WIDTH = 120, 480, 832
HEADS, BLOCKS = 56, 225
CALL_TOTAL = HEADS * 22  # 1,232 blocks per sparse (evaluation, layer)
SPARSE_LAYERS = 49       # layers 1..49
GPUS = (0, 1, 2, 3)
read, write, sha = base.read, base.write, base.sha


def configure():
    base.REPO = REPO
    base.MODEL = MODEL
    base.EXP, base.OUT = EXP, OUT
    base.SHAPES = {480: (832, 225, 'head_sparsity_all50_20260919')}
    base.ARMS = ('dense',) + WORKER_ARMS


class UniformAllocator:
    """Exact 22 blocks per head at every sparse (evaluation, layer)."""

    def __init__(self):
        self.records = []

    def __call__(self, layer, evaluation, values, video_tokens):
        heads = values.shape[1]
        blocks = video_tokens // 64
        if heads != HEADS or blocks != BLOCKS:
            raise ValueError('unexpected head count or video-block resolution')
        if not 0 <= evaluation <= 18 or not 1 <= layer <= 49:
            raise ValueError('allocator called outside the sparse evaluations/layers')
        budget = np.full(heads, 22, dtype=np.int64)
        self.records.append(dict(layer=layer, evaluation=evaluation, blocks=blocks,
                                 total=CALL_TOTAL, budgets=budget.tolist()))
        import torch
        return torch.as_tensor(budget, device=values.device, dtype=torch.int64)


def arm_evaluation(arm):
    return int(arm.split('_eval_')[1])


def arm_sparse_evals(arm):
    if arm == 'uniform':
        return set(range(4, 19))
    if arm.startswith('dense_eval_'):
        return set(range(4, 19)) - {arm_evaluation(arm)}
    return set(range(4, 19)) | {arm_evaluation(arm)}


def arm_extra_dense_evals(arm):
    if arm == 'uniform':
        return ()
    if arm.startswith('dense_eval_'):
        return (arm_evaluation(arm),)
    return tuple(sorted(set(REVERSE_EVALS) - {arm_evaluation(arm)}))


def expected_calls(arm):
    return len(arm_sparse_evals(arm)) * SPARSE_LAYERS


def prepare():
    EXP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    required = [MODEL / 'audio_vae/diffusion_pytorch_model.safetensors']
    for folder in ('transformer', 'vae'):
        index = read(MODEL / folder / 'diffusion_pytorch_model.safetensors.index.json')
        required.extend(MODEL / folder / name for name in set(index['weight_map'].values()))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = read(base.CONDITIONING / 'conditioning_manifest.json')
    cases = [r for r in manifest['records'] if r['index'] in CASES]
    assert len(cases) == len(CASES)
    # Dense references: reuse as in the ramp-budget experiment.
    for case in CASES:
        stem = f'480_dense_{case:02}'
        for folder in ('records', 'decoded'):
            record = read(BASELINE / folder / f'{stem}.json')
            if folder == 'decoded':
                assert Path(record['video_path']).is_file() and sha(record['video_path']) == record['sha256']
            write(EXP / folder / f'{stem}.json', record)
    # Uniform baseline: reuse from the ramp-budget experiment only if the
    # configuration, latents, videos, and quality records all verify.
    sys.path.insert(0, str(REPO))
    from h3_sparse_attention import H3SparseAttentionConfig
    import json
    expected_config = json.loads(json.dumps(dataclasses.asdict(
        H3SparseAttentionConfig.spark(20, sol_log_density=False))))
    reused = True
    for case in CASES:
        stem = f'480_uniform_{case:02}'
        try:
            r = read(BASELINE / 'records' / f'{stem}.json')
            q = read(BASELINE / 'quality' / f'{stem}.json')
            d = read(BASELINE / 'decoded' / f'{stem}.json')
            got_config = dict(r['config'])
            got_config.setdefault('sol_extra_dense_evaluations', [])  # predates the config field
            checks = [
                got_config == expected_config,
                (r['frames'], r['height'], r['width'], r['steps'], r['seed']) == (FRAMES, HEIGHT, WIDTH, 20, 42),
                Path(r['latent_path']).is_file() and sha(r['latent_path']) == r['latent_sha256'],
                Path(d['video_path']).is_file() and sha(d['video_path']) == d['sha256'],
                len(r['calls']) == 735 and all(c['total'] == CALL_TOTAL for c in r['calls']),
                q['arm'] == 'uniform' and q['case'] == case,
            ]
            assert all(checks), checks
        except (AssertionError, KeyError, FileNotFoundError) as exc:
            print('UNIFORM REUSE REJECTED', stem, repr(exc), flush=True)
            reused = False
            break
        for folder, data in (('records', r), ('decoded', d), ('quality', q)):
            write(EXP / folder / f'{stem}.json', data)
        print('UNIFORM REUSED', stem, flush=True)
    write(EXP / 'protocol.json', dict(cases=cases, arms=['dense', *WORKER_ARMS],
        frames=FRAMES, fps=24, height=HEIGHT, width=WIDTH, steps=20, seed=42, gpus=[0, 1],
        heads=HEADS, blocks=BLOCKS, uniform_per_head=22, exact_total=CALL_TOTAL,
        sparse_evaluations='4..18', sparse_layers='1..49', dense_evaluations=4, dense_layers=1,
        extra_dense_evals=list(EXTRA_DENSE_EVALS),
        calls_per_generation=dict(uniform=735, dense_eval_arms=686),
        model=str(MODEL), environment=base.ENV,
        model_index_sha256={folder: sha(MODEL / folder / 'diffusion_pytorch_model.safetensors.index.json')
                            for folder in ('transformer', 'vae')},
        probe=('Each dense_eval_XX arm adds exactly one dense evaluation (true production dense path, '
               'not Top-K with full budget) to the uniform 22-blocks/head baseline. Arms therefore spend '
               'MORE total compute than uniform: this is a per-timestep importance probe, not a '
               'budget-matched reallocation.'),
        reverse_probe=('Each sparse_eval_0X arm (X in 0..3) REMOVES dense from warmup evaluation X: built '
               'with warmup_percent=0 and the other three early evaluations in '
               'sol_extra_dense_evaluations, so X runs the uniform sparse path (16 sparse evaluations, '
               '784 allocator calls, 150 extra-dense layer calls). Dense gain of warmup eval X = '
               'uniform - sparse_eval_0X.'),
        uniform_baseline=dict(source=str(BASELINE), reused=reused,
            verification='Config equality, frames/height/width/steps/seed, latent SHA256, video SHA256, 735 calls at 1232 blocks'),
        references=dict(source=str(BASELINE), reused='Dense records and lossless videos, SHA256 verified'),
        timing='Denoising includes allocation and audit overhead; rotated arm order per case; no isolated kernel speed claims.'))
    write(EXP / 'source_manifest.json', {str(p): sha(p) for p in
        [Path(__file__), base.__file__, *REPO.glob('h3_sparse_attention/*.py')]})


def generate(slot, case_positions, arms):
    import torch
    torch.set_num_threads(4)
    h3, p = base.imports()
    assert Path(h3.__file__).is_relative_to(REPO)
    args = p.build_parser().parse_args(['denoise', '--model', str(base.MODEL), '--output', str(OUT),
        '--method', 'sol', '--frames', str(FRAMES), '--steps', '20', '--no-torch-compile'])
    workflow = p.get_workflow(base.MODEL)
    for key in ('text_encoder', 'decode.video', 'decode.audio'):
        workflow.sub_blocks.pop(key)
    with base.buffered_model_loading():
        pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    try:
        for ci in case_positions:
            case = read(EXP / 'protocol.json')['cases'][ci]
            assert case['index'] == CASES[ci]
            state, conditioning_path, conditioning_sha = base.cached_state(case)
            rotation = ci % len(arms)
            order = tuple(arms[rotation:]) + tuple(arms[:rotation])
            for arm in order:
                stem = f'{HEIGHT}_{arm}_{case["index"]:02}'
                target = EXP / 'records' / f'{stem}.json'
                if target.exists():
                    continue
                extra = arm_extra_dense_evals(arm)
                overrides = dict(sol_log_density=False, sol_extra_dense_evaluations=extra)
                if arm.startswith('sparse_eval_'):
                    overrides['warmup_percent'] = 0.0  # eval 0-3 gate moved to extra-dense
                cfg = h3.H3SparseAttentionConfig.spark(20, **overrides)
                allocator = UniformAllocator()
                prepared = p.clone_state(state)
                prepared.values['prompt_embeds'] = prepared.values['prompt_embeds'].cuda()
                write(EXP / f'status_gpu{slot}.json', dict(stage='inference', case=case['index'], arm=arm))
                with h3.install_h3_sparse_attention(pipe.transformer, cfg) as plugin:
                    plugin.controller.head_budget_allocator = allocator
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = pipe(state=prepared, num_frames=FRAMES, height=HEIGHT, width=WIDTH,
                            num_inference_steps=20, generator=torch.Generator(device='cpu').manual_seed(42),
                            output=['latents', 'audio_latents'])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    summary = plugin.summary()
                assert summary['completed_evaluations'] == 19
                evaluations = {r['evaluation'] for r in allocator.records}
                assert len(allocator.records) == expected_calls(arm), len(allocator.records)
                assert all(r['total'] == CALL_TOTAL for r in allocator.records)
                assert evaluations == arm_sparse_evals(arm)
                assert summary['processor_calls'].get('dense:extra_dense_evaluation', 0) == (
                    50 * len(extra))  # all 50 layers, incl. layer 0
                payload = {key: result[key].detach().cpu().contiguous() for key in ('latents', 'audio_latents')}
                assert all(torch.isfinite(v).all() for v in payload.values())
                path = OUT / 'latents' / f'{stem}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                p.atomic_torch_save(payload, path)
                write(target, dict(arm=arm, case=case['index'], height=HEIGHT,
                    width=WIDTH, frames=FRAMES, seed=42, steps=20, config=dataclasses.asdict(cfg), gpu=slot,
                    prompt_sha256=case['prompt_sha256'], conditioning_path=conditioning_path,
                    conditioning_sha256=conditioning_sha, latent_path=str(path), latent_sha256=sha(path),
                    denoise_seconds=elapsed, summary=summary, calls=allocator.records, placement=placement))
                print('INFERENCE COMPLETE', stem, elapsed, flush=True)
                del result, prepared, payload, allocator
                torch.cuda.empty_cache()
        write(EXP / f'status_gpu{slot}.json', dict(stage='inference_complete'))
    finally:
        acceleration.remove()


def worker(slot):
    generate(slot, range(slot, len(CASES), len(GPUS)), WORKER_ARMS)


def smoke(slot):
    """sparse_eval_00 for case 3, end to end through scoring."""
    assert slot == 0 and CASES[0] == 3
    generate(slot, (0,), ('sparse_eval_00',))
    full_arms = base.ARMS
    base.ARMS = ('dense', 'sparse_eval_00')
    try:
        base.decode(0)
        base.score(0)
    finally:
        base.ARMS = full_arms


def report():
    rows, summaries = [], []
    for case, arm in itertools.product(CASES, WORKER_ARMS):
        stem = f'480_{arm}_{case:02}'
        q = read(EXP / 'quality' / f'{stem}.json')
        r = read(EXP / 'records' / f'{stem}.json')
        d = read(EXP / 'decoded' / f'{stem}.json')
        assert len(r['calls']) == expected_calls(arm)
        assert all(c['total'] == CALL_TOTAL for c in r['calls'])
        assert {c['evaluation'] for c in r['calls']} == arm_sparse_evals(arm)
        assert d['evidence']['pyav_pixel_equal'] and d['evidence']['ffmpeg_pixel_equal']
        rows.append({**q, 'denoise_seconds': r['denoise_seconds']})
    for arm in WORKER_ARMS:
        group = [r for r in rows if r['arm'] == arm]
        summaries.append(dict(arm=arm, **{key: statistics.mean(r[key] for r in group)
            for key in ('psnr_db', 'ssim', 'lpips', 'denoise_seconds')}))
    uniform = next(s for s in summaries if s['arm'] == 'uniform')
    for s in summaries:
        s['delta_vs_uniform'] = {key: s[key] - uniform[key]
            for key in ('psnr_db', 'ssim', 'lpips', 'denoise_seconds')}
    by_arm = {s['arm']: s for s in summaries}
    # Combined whole-trajectory importance: evals 0-3 measured in reverse
    # (uniform minus sparse_eval_0X), evals 4-18 forward (dense_eval_XX minus
    # uniform). Positive = dense at that evaluation helps.
    importance = {}
    for e in range(19):
        if e < 4:
            s = by_arm[f'sparse_eval_{e:02}']
            importance[e] = {key: -s['delta_vs_uniform'][key] for key in ('psnr_db', 'ssim', 'lpips')}
        else:
            importance[e] = by_arm[f'dense_eval_{e:02}']['delta_vs_uniform']
    curve_psnr = {e: importance[e]['psnr_db'] for e in range(19)}
    warmup = statistics.mean(curve_psnr[e] for e in range(4))
    early = statistics.mean(curve_psnr[e] for e in range(4, 9))
    middle = statistics.mean(curve_psnr[e] for e in range(9, 14))
    late = statistics.mean(curve_psnr[e] for e in range(14, 19))
    peak = max(curve_psnr, key=curve_psnr.get)
    trough = min(curve_psnr, key=curve_psnr.get)
    ranking = sorted(range(19), key=lambda e: curve_psnr[e], reverse=True)
    write(EXP / 'results.json', dict(status='complete', summary=summaries, rows=rows,
        importance=importance, psnr_gain_ranking=ranking,
        curve=dict(gains=curve_psnr, warmup_0_3=warmup, early_4_8=early, middle_9_13=middle,
                   late_14_18=late, peak=peak, trough=trough)))
    lines = ['# Per-timestep dense-importance over the full trajectory: 5s 480p', '',
        'Four prompts, seed 42, 120 frames at 24 fps, 832x480, 20 requested steps (19 evaluations). '
        'GPUs 0-3. Layer 0 is always dense. Baseline: uniform 10% Top-K (22 blocks/head, exact 1,232 '
        'blocks per sparse call) with evaluations 0-3 dense. `dense_eval_XX` arms additionally run '
        'evaluation XX (X in 4..18) through the true production dense path. `sparse_eval_0X` reverse arms '
        'remove dense from warmup evaluation X (X in 0..3): built with warmup_percent=0 and the other '
        'three early evaluations passed as sol_extra_dense_evaluations, so X runs through the uniform '
        'sparse path. **Probe arms are not budget-matched to uniform: this is an importance probe, not a '
        'reallocation.** The uniform baseline and dense references are reused from the ramp-budget '
        'experiment after config and SHA256 verification.', '',
        '## Averages over four prompts', '',
        '| Arm | PSNR | SSIM | LPIPS | Denoise seconds | dPSNR vs uniform | dSSIM | dLPIPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        d = s['delta_vs_uniform']
        lines.append(f"| {s['arm']} | {s['psnr_db']:.4f} | {s['ssim']:.6f} | {s['lpips']:.6f} | "
                     f"{s['denoise_seconds']:.2f} | {d['psnr_db']:+.4f} | {d['ssim']:+.6f} | {d['lpips']:+.6f} |")
    lines += ['', '## Dense importance of every evaluation (positive = dense helps)', '',
        'Evaluations 0-3: dense gain = uniform minus `sparse_eval_0X`. Evaluations 4-18: dense gain = '
        '`dense_eval_XX` minus uniform. Sorted by evaluation index.', '',
        '| Evaluation | dPSNR | dSSIM | dLPIPS | Measured by |', '|---|---:|---:|---:|---|']
    for e in range(19):
        how = 'reverse (dense removed)' if e < 4 else 'forward (dense added)'
        lines.append(f"| {e} | {importance[e]['psnr_db']:+.4f} | {importance[e]['ssim']:+.6f} | "
                     f"{importance[e]['lpips']:+.6f} | {how} |")
    lines += ['', '## Evaluations ranked by PSNR importance', '',
        ', '.join(f'{e} ({curve_psnr[e]:+.3f})' for e in ranking), '',
        '## Importance curve shape (mean dPSNR)', '',
        f'Warmup evaluations 0-3: {warmup:+.4f}; early 4-8: {early:+.4f}; middle 9-13: {middle:+.4f}; '
        f'late 14-18: {late:+.4f}. Peak importance at evaluation {peak} ({curve_psnr[peak]:+.4f} dB); '
        f'least important is evaluation {trough} ({curve_psnr[trough]:+.4f} dB). '
        + ('Importance decreases along the whole trajectory.' if warmup > early > middle > late else
           'Importance increases along the whole trajectory.' if warmup < early < middle < late else
           'Importance is non-monotone across the trajectory.'), '',
        '## Per-prompt results', '',
        '| Case | Arm | PSNR | SSIM | LPIPS | Denoise seconds |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']:02} | {r['arm']} | {r['psnr_db']:.4f} | {r['ssim']:.6f} | "
                     f"{r['lpips']:.6f} | {r['denoise_seconds']:.2f} |")
    lines += ['', '## Validation and limits', '',
        '76 probe generations plus 4 reused uniform baselines: every sparse allocation call kept the '
        'exact 1,232-block total (686 calls per dense_eval generation, 735 for uniform, 784 for '
        'sparse_eval arms, which sparsify one warmup evaluation on top of evaluations 4-18). Each '
        'dense_eval generation took the production dense path at exactly its designated '
        'evaluation; each sparse_eval generation ran exactly its designated warmup evaluation through the '
        'sparse path with the other three early evaluations dense (warmup_percent=0 plus '
        'sol_extra_dense_evaluations, 150 extra-dense layer calls). 76 lossless FFV1 archives passed '
        'frame, dimension, frame-rate, exact RGB, and audio verification. Probe arms are not '
        'budget-matched to uniform; gains mix timestep importance with changed compute. The two '
        'measurement directions (adding vs removing dense) need not be symmetric. Four prompts and one '
        'seed do not establish broad generalization. Timings include allocation/audit overhead.', '',
        f'Checkpoint: `{MODEL}`. Videos: `{OUT / "videos"}`. Per-prompt metrics: `results.json`. '
        f'Protocol and manifests: `{EXP}`.']
    (EXP / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    target = REPO / 'reports' / EXP.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ('REPORT.md', 'results.json', 'protocol.json', 'source_manifest.json'):
        shutil.copy2(EXP / name, target / name)


def launch(stage):
    processes = []
    for slot in GPUS:
        with (EXP / f'{stage}_gpu{slot}.log').open('a') as log:
            processes.append(subprocess.Popen([sys.executable, __file__, stage, '--slot', str(slot)],
                cwd=REPO, env={**os.environ, **base.ENV, 'CUDA_VISIBLE_DEVICES': str(slot)},
                stdout=log, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        write(EXP / 'status.json', dict(status='running', stage=stage, gpus=list(GPUS),
            records=len(list((EXP/'records').glob('*.json'))),
            decoded=len(list((EXP/'decoded').glob('*.json'))),
            quality=len(list((EXP/'quality').glob('*.json'))),
            expected_records=84, expected_quality=80))
        time.sleep(10)
    codes = [p.wait() for p in processes]
    if any(codes):
        write(EXP / 'status.json', dict(status='failed', stage=stage, codes=codes))
        raise RuntimeError((stage, codes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'worker', 'finish', 'report', 'run'))
    parser.add_argument('--slot', type=int, choices=GPUS, default=0)
    args = parser.parse_args()
    os.environ.update(base.ENV)
    configure()
    EXP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'run':
        launch('worker')
        launch('finish')
        report()
        write(EXP / 'status.json', dict(status='complete', probe_generations=76, quality=80))
    elif args.stage in ('smoke', 'worker'):
        globals()[args.stage](args.slot)
    elif args.stage == 'finish':
        for ci in range(args.slot, len(CASES), len(GPUS)):
            base.decode(ci)
            base.score(ci)
    else:
        globals()[args.stage]()
