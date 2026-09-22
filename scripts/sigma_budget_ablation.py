"""Noise-level-based step-budget schedules vs linear decrease: 5s/480p, GPUs 0-3.

All arms spend the exact uniform grand total (735 calls x 1,232 = 905,520
routed blocks per generation) and differ only in how the per-head budget is
distributed across the 15 sparse evaluations 4..18 (per-head sum exactly 330):

- `uniform`: 22 blocks/head at every sparse evaluation (reused, verified).
- `step_13to7`: linear decrease 28.5->15.5 blocks/head (reused, verified).
- `step_sigma`: b_e proportional to sigma_e, the MiniMaxH3Scheduler noise level
  at evaluation e (shift=12 from the checkpoint scheduler config,
  set_timesteps(20), sigmas[e]).
- `step_sigma_pow2`: b_e proportional to sigma_e^2.

Budgets are largest-remainder integers (min 1 block/head). The uniform and
step_13to7 baselines plus the dense references are reused from the ramp-budget
experiment after config and SHA256 verification. Stages:
prepare/smoke/worker/finish/report/run.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
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
EXP = Path('/autodl-fs/data/h3_experiments/sigma_budget_5s_480p_20260920')
OUT = Path('/autodl-fs/data/h3_outputs') / EXP.name
BASELINE = Path('/autodl-fs/data/h3_experiments/ramp_budget_5s_480p_20260920')
REUSED_ARMS = ('uniform', 'step_13to7')
NEW_ARMS = ('step_sigma', 'step_sigma_pow2')
WORKER_ARMS = REUSED_ARMS + NEW_ARMS
CASES = base.CASES
FRAMES, HEIGHT, WIDTH = 120, 480, 832
HEADS, BLOCKS = 56, 225
SPARSE_EVALS = tuple(range(4, 19))
SPARSE_LAYERS = 49
CALLS = len(SPARSE_EVALS) * SPARSE_LAYERS  # 735
PER_HEAD_TOTAL = 330
GRAND_TOTAL = HEADS * PER_HEAD_TOTAL * SPARSE_LAYERS  # 905,520
RAMP_LO, RAMP_HI = 15.5, 28.5
GPUS = (0, 1, 2, 3)
read, write, sha = base.read, base.write, base.sha


def configure():
    base.REPO = REPO
    base.MODEL = MODEL
    base.EXP, base.OUT = EXP, OUT
    base.SHAPES = {480: (832, 225, 'head_sparsity_all50_20260919')}
    base.ARMS = ('dense',) + WORKER_ARMS


def ramp_budgets(count, lo=RAMP_LO, hi=RAMP_HI):
    """Largest-remainder integer ramp with an exact symmetric total."""
    values = np.linspace(lo, hi, count)
    floors = np.floor(values).astype(np.int64)
    remaining = int(round(values.sum())) - int(floors.sum())
    order = np.argsort(-(values - floors), kind='stable')
    floors[order[:remaining]] += 1
    assert int(floors.sum()) == count * 22
    return floors


STEP_13TO7 = ramp_budgets(len(SPARSE_EVALS))[::-1]


def proportional_budgets(weights, total=PER_HEAD_TOTAL, minimum=1):
    """Largest-remainder integer shares of `total`, floored at `minimum`."""
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or not weights.any():
        raise ValueError('weights must be finite, nonnegative, and not all zero')
    raw = weights / weights.sum() * total
    budgets = np.maximum(np.floor(raw).astype(np.int64), minimum)
    remaining = int(total - budgets.sum())
    if remaining < 0:
        raise ValueError('minimum budget makes the total infeasible')
    order = np.argsort(-(raw - np.floor(raw)), kind='stable')
    budgets[order[:remaining]] += 1
    assert int(budgets.sum()) == total
    return budgets


def sigma_schedule():
    """Noise levels at the 19 evaluations, exactly as the pipeline builds them."""
    sys.path.insert(0, '/autodl-fs/data/h3_repos/env_src/diffusers-src/src')
    from diffusers import MiniMaxH3Scheduler
    config_path = MODEL / 'scheduler' / 'scheduler_config.json'
    scheduler = MiniMaxH3Scheduler.from_config(json.loads(config_path.read_text()))
    scheduler.set_timesteps(20)
    sigmas = scheduler.sigmas.tolist()
    assert len(sigmas) == 20 and sigmas[-1] == 0.0
    return sigmas[:19], sha(config_path)


class StepAllocator:
    """Per-head budget schedule over sparse evaluations 4..18."""

    def __init__(self, budgets):
        self.budgets = {int(k): int(v) for k, v in budgets.items()}
        assert set(self.budgets) == set(SPARSE_EVALS)
        assert sum(self.budgets.values()) == PER_HEAD_TOTAL
        self.records = []

    def __call__(self, layer, evaluation, values, video_tokens):
        heads = values.shape[1]
        blocks = video_tokens // 64
        if heads != HEADS or blocks != BLOCKS:
            raise ValueError('unexpected head count or video-block resolution')
        if evaluation not in self.budgets or not 1 <= layer <= 49:
            raise ValueError('allocator called outside the sparse evaluations/layers')
        budget = np.full(heads, self.budgets[evaluation], dtype=np.int64)
        self.records.append(dict(layer=layer, evaluation=evaluation, blocks=blocks,
                                 total=int(budget.sum()), budgets=budget.tolist()))
        import torch
        return torch.as_tensor(budget, device=values.device, dtype=torch.int64)


def arm_budgets(arm):
    """Per-head budgets by sparse evaluation, per-arm; uniform is constant 22."""
    if arm == 'uniform':
        return {e: 22 for e in SPARSE_EVALS}
    if arm == 'step_13to7':
        return dict(zip(SPARSE_EVALS, STEP_13TO7.tolist()))
    data = read(EXP / 'calibration' / 'step_budgets.json')
    return {int(k): v for k, v in data[arm].items()}


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
    # Sigma schedule and proportional budget tables.
    sigmas, config_sha = sigma_schedule()
    sparse_sigmas = np.array([sigmas[e] for e in SPARSE_EVALS])
    budgets = {
        'step_sigma': proportional_budgets(sparse_sigmas),
        'step_sigma_pow2': proportional_budgets(sparse_sigmas ** 2),
    }
    (EXP / 'calibration').mkdir(parents=True, exist_ok=True)
    write(EXP / 'calibration' / 'step_budgets.json',
          {arm: dict(zip(map(str, SPARSE_EVALS), b.tolist())) for arm, b in budgets.items()})
    for arm, b in budgets.items():
        assert sum(b) == PER_HEAD_TOTAL and b.min() >= 1
        print(arm, b.tolist(), flush=True)
    # Uniform and linear-decrease baselines: reuse from the ramp-budget
    # experiment only if configuration, latents, videos, and schedules verify.
    sys.path.insert(0, str(REPO))
    from h3_sparse_attention import H3SparseAttentionConfig
    expected_config = json.loads(json.dumps(dataclasses.asdict(
        H3SparseAttentionConfig.spark(20, sol_log_density=False))))
    reused = []
    for arm in REUSED_ARMS:
        expected_totals = {e: HEADS * b for e, b in arm_budgets(arm).items()}
        for case in CASES:
            stem = f'480_{arm}_{case:02}'
            try:
                r = read(BASELINE / 'records' / f'{stem}.json')
                q = read(BASELINE / 'quality' / f'{stem}.json')
                d = read(BASELINE / 'decoded' / f'{stem}.json')
                got_config = dict(r['config'])
                got_config.setdefault('sol_extra_dense_evaluations', [])  # predates the field
                checks = [
                    got_config == expected_config,
                    (r['frames'], r['height'], r['width'], r['steps'], r['seed']) == (FRAMES, HEIGHT, WIDTH, 20, 42),
                    Path(r['latent_path']).is_file() and sha(r['latent_path']) == r['latent_sha256'],
                    Path(d['video_path']).is_file() and sha(d['video_path']) == d['sha256'],
                    len(r['calls']) == CALLS,
                    all(c['total'] == expected_totals[c['evaluation']] for c in r['calls']),
                    sum(c['total'] for c in r['calls']) == GRAND_TOTAL,
                    q['arm'] == arm and q['case'] == case,
                ]
                assert all(checks), checks
            except (AssertionError, KeyError, FileNotFoundError) as exc:
                print('BASELINE REUSE REJECTED', stem, repr(exc), flush=True)
                break
            for folder, data in (('records', r), ('decoded', d), ('quality', q)):
                write(EXP / folder / f'{stem}.json', data)
            print('BASELINE REUSED', stem, flush=True)
        else:
            reused.append(arm)
    write(EXP / 'protocol.json', dict(cases=cases, arms=['dense', *WORKER_ARMS],
        frames=FRAMES, fps=24, height=HEIGHT, width=WIDTH, steps=20, seed=42, gpus=list(GPUS),
        heads=HEADS, blocks=BLOCKS, per_head_total=PER_HEAD_TOTAL,
        sparse_evaluations='4..18', sparse_layers='1..49', calls_per_generation=CALLS,
        grand_total=GRAND_TOTAL, dense_evaluations=4, dense_layers=1,
        model=str(MODEL), environment=base.ENV,
        model_index_sha256={folder: sha(MODEL / folder / 'diffusion_pytorch_model.safetensors.index.json')
                            for folder in ('transformer', 'vae')},
        sigma_schedule=dict(
            source='MiniMaxH3Scheduler instantiated from the checkpoint scheduler config '
                   f'({MODEL}/scheduler/scheduler_config.json, shift=12.0, sha256={config_sha}); '
                   'set_timesteps(20) exactly as the modular pipeline denoise block calls it '
                   '(before_denoise.py: components.scheduler.set_timesteps(num_inference_steps)); '
                   'evaluation e uses sigmas[e]; the terminal sigma 0.0 drives no evaluation.',
            sigmas_at_evaluations={str(e): sigmas[e] for e in range(19)}),
        step_budgets_per_head={arm: {str(e): b for e, b in arm_budgets(arm).items()} for arm in WORKER_ARMS},
        methods=dict(uniform='22 blocks/head at every sparse evaluation',
            step_13to7='Linear decrease 28.5->15.5 blocks/head over evaluations 4->18 (largest-remainder)',
            step_sigma='Per-head blocks proportional to sigma_e over evaluations 4..18, sum 330',
            step_sigma_pow2='Per-head blocks proportional to sigma_e^2 over evaluations 4..18, sum 330'),
        baselines=dict(source=str(BASELINE), reused=reused,
            verification='Config equality, frames/height/width/steps/seed, latent SHA256, video SHA256, '
                         '735 calls with per-evaluation totals matching the arm schedule, grand total 905520'),
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
                cfg = h3.H3SparseAttentionConfig.spark(20, sol_log_density=False)
                expected_totals = {e: HEADS * b for e, b in arm_budgets(arm).items()}
                allocator = StepAllocator(arm_budgets(arm))
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
                assert len(allocator.records) == CALLS, len(allocator.records)
                assert all(r['total'] == expected_totals[r['evaluation']] for r in allocator.records)
                assert sum(r['total'] for r in allocator.records) == GRAND_TOTAL
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
    """step_sigma for case 3, end to end through scoring."""
    assert slot == 0 and CASES[0] == 3
    generate(slot, (0,), ('step_sigma',))
    full_arms = base.ARMS
    base.ARMS = ('dense', 'step_sigma')
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
        expected_totals = {e: HEADS * b for e, b in arm_budgets(arm).items()}
        assert len(r['calls']) == CALLS
        assert all(c['total'] == expected_totals[c['evaluation']] for c in r['calls'])
        assert sum(c['total'] for c in r['calls']) == GRAND_TOTAL
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
    best = max((s for s in summaries if s['arm'] != 'uniform'),
               key=lambda s: s['delta_vs_uniform']['psnr_db'])
    write(EXP / 'results.json', dict(status='complete', summary=summaries, rows=rows))
    protocol = read(EXP / 'protocol.json')
    sigmas = protocol['sigma_schedule']['sigmas_at_evaluations']
    lines = ['# Noise-level-based step-budget schedules vs linear decrease: 5s 480p', '',
        'Four prompts, seed 42, 120 frames at 24 fps, 832x480, 20 requested steps (19 evaluations). '
        'GPUs 0-3. First four evaluations and layer 0 remain dense. Every arm spends exactly the uniform '
        'grand total (735 calls, 905,520 routed blocks per generation; per-head sum exactly 330 over the '
        '15 sparse evaluations). All arms use the same explicit stable Top-K score sort with exact '
        'per-head budgets under the production Spark reweighting branch. The uniform and step_13to7 '
        'baselines and the dense references are reused from the ramp-budget experiment after config and '
        'SHA256 verification.', '',
        '## Per-evaluation budgets (blocks/head) and noise levels', '',
        '| Evaluation | sigma | uniform | step_13to7 | step_sigma | step_sigma_pow2 |',
        '|---|---:|---:|---:|---:|---:|']
    for e in SPARSE_EVALS:
        b = {arm: arm_budgets(arm)[e] for arm in WORKER_ARMS}
        lines.append(f"| {e} | {sigmas[str(e)]:.6f} | {b['uniform']} | {b['step_13to7']} | "
                     f"{b['step_sigma']} | {b['step_sigma_pow2']} |")
    lines += ['', '## Averages over four prompts', '',
        '| Arm | PSNR | SSIM | LPIPS | Denoise seconds | dPSNR vs uniform | dSSIM | dLPIPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        d = s['delta_vs_uniform']
        lines.append(f"| {s['arm']} | {s['psnr_db']:.4f} | {s['ssim']:.6f} | {s['lpips']:.6f} | "
                     f"{s['denoise_seconds']:.2f} | {d['psnr_db']:+.4f} | {d['ssim']:+.6f} | {d['lpips']:+.6f} |")
    lines += ['', '## Per-prompt results', '',
        '| Case | Arm | PSNR | SSIM | LPIPS | Denoise seconds |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']:02} | {r['arm']} | {r['psnr_db']:.4f} | {r['ssim']:.6f} | "
                     f"{r['lpips']:.6f} | {r['denoise_seconds']:.2f} |")
    lines += ['', '## Verdict', '',
        f"Best non-uniform schedule by mean PSNR: `{best['arm']}` "
        f"({best['delta_vs_uniform']['psnr_db']:+.4f} dB vs uniform). "
        'With shift=12 the sigma grid stays high until late in the trajectory (sigma_4=0.978, '
        'sigma_18=0.400), so sigma-proportional allocation is front-heavy but keeps a substantial tail; '
        'sigma^2 is much steeper. See the budget table and per-prompt rows for the full comparison of '
        'noise-level-based versus linear decrease schedules.', '',
        '## Validation and limits', '',
        f"{len(rows)} generations total ({len(REUSED_ARMS)} arms x 4 cases reused and verified, "
        f"{len(NEW_ARMS)} arms x 4 cases new): every sparse allocation call kept its arm-schedule "
        'per-evaluation total and every grand total was exactly 905,520 blocks. All FFV1 archives passed '
        'frame, dimension, frame-rate, exact RGB, and audio verification. All schedules redistribute the '
        'same fixed budget; they do not test budget scaling. Four prompts and one seed do not establish '
        'broad generalization. Timings include allocation/audit overhead.', '',
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
            expected_records=20, expected_quality=16))
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
        write(EXP / 'status.json', dict(status='complete', new_generations=8, quality=16))
    elif args.stage in ('smoke', 'worker'):
        globals()[args.stage](args.slot)
    elif args.stage == 'finish':
        for ci in range(args.slot, len(CASES), len(GPUS)):
            base.decode(ci)
            base.score(ci)
    else:
        globals()[args.stage]()
