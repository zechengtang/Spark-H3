"""Linear-slope scan of step-budget ramps: 5s/480p, GPUs 0-3.

All arms spend the exact uniform grand total (735 calls x 1,232 = 905,520
routed blocks per generation); per-head budgets over the 15 sparse evaluations
4..18 sum to exactly 330 (largest-remainder rounding, min 1 block/head). Arms
are symmetric linear ramps 22+-d blocks/head:

- `uniform`: d = 0, 22 blocks/head everywhere (reused, verified).
- `step_13to7`: d = 6.5, the linear winner of the ramp/sigma/exp experiments
  (reused, verified).
- `step_lin_30to14` (d=8), `step_lin_33to11` (d=11), `step_lin_36to8` (d=14),
  `step_lin_39to5` (d=17), `step_lin_42to2` (d=20): steeper slopes scanning for
  the PSNR peak before LPIPS collapses from tail starvation.

The baselines and dense references are reused from the ramp-budget experiment
after config and SHA256 verification. Stages: prepare/smoke/worker/finish/
report/run.
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
EXP = Path('/autodl-fs/data/h3_experiments/slope_budget_5s_480p_20260920')
OUT = Path('/autodl-fs/data/h3_outputs') / EXP.name
BASELINE = Path('/autodl-fs/data/h3_experiments/ramp_budget_5s_480p_20260920')
B2 = Path('/autodl-fs/data/h3_experiments/exp_budget_5s_480p_20260920')
REUSED_ARMS = ('uniform', 'step_13to7')
SLOPES = dict(step_lin_30to14=8, step_lin_33to11=11, step_lin_36to8=14,
              step_lin_39to5=17, step_lin_42to2=20)
NEW_ARMS = tuple(SLOPES)
WORKER_ARMS = REUSED_ARMS + NEW_ARMS
CASES = base.CASES
FRAMES, HEIGHT, WIDTH = 120, 480, 832
HEADS, BLOCKS = 56, 225
SPARSE_EVALS = tuple(range(4, 19))
SPARSE_LAYERS = 49
CALLS = len(SPARSE_EVALS) * SPARSE_LAYERS  # 735
PER_HEAD_TOTAL = 330
GRAND_TOTAL = HEADS * PER_HEAD_TOTAL * SPARSE_LAYERS  # 905,520
GPUS = (0, 1, 2, 3)
read, write, sha = base.read, base.write, base.sha


def configure():
    base.REPO = REPO
    base.MODEL = MODEL
    base.EXP, base.OUT = EXP, OUT
    base.SHAPES = {480: (832, 225, 'head_sparsity_all50_20260919')}
    base.ARMS = ('dense',) + WORKER_ARMS


def ramp_budgets(count, lo, hi):
    """Largest-remainder integer ramp with an exact symmetric total."""
    values = np.linspace(lo, hi, count)
    floors = np.floor(values).astype(np.int64)
    remaining = int(round(values.sum())) - int(floors.sum())
    order = np.argsort(-(values - floors), kind='stable')
    floors[order[:remaining]] += 1
    assert int(floors.sum()) == count * 22
    return floors


STEP_13TO7 = ramp_budgets(len(SPARSE_EVALS), 15.5, 28.5)[::-1]
SLOPE_RAMPS = {arm: ramp_budgets(len(SPARSE_EVALS), 22 + d, 22 - d)
               for arm, d in SLOPES.items()}


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
    return dict(zip(SPARSE_EVALS, SLOPE_RAMPS[arm].tolist()))


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
    for arm, b in SLOPE_RAMPS.items():
        assert int(b.sum()) == PER_HEAD_TOTAL and b.min() >= 1
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
        step_budgets_per_head={arm: {str(e): b for e, b in arm_budgets(arm).items()} for arm in WORKER_ARMS},
        slopes_d={'uniform': 0, 'step_13to7': 6.5, **SLOPES},
        methods=dict(uniform='22 blocks/head at every sparse evaluation (d=0)',
            step_13to7='Linear ramp 28.5->15.5 blocks/head (d=6.5), existing linear winner',
            step_lin_30to14='Linear ramp 30->14 (d=8)',
            step_lin_33to11='Linear ramp 33->11 (d=11)',
            step_lin_36to8='Linear ramp 36->8 (d=14)',
            step_lin_39to5='Linear ramp 39->5 (d=17)',
            step_lin_42to2='Linear ramp 42->2 (d=20)'),
        rounding='Largest remainder, ties by lowest evaluation index; documented integer tables may shift ramp endpoints by +/-1.',
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
    """step_lin_33to11 for case 3, end to end through scoring."""
    assert slot == 0 and CASES[0] == 3
    generate(slot, (0,), ('step_lin_33to11',))
    full_arms = base.ARMS
    base.ARMS = ('dense', 'step_lin_33to11')
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
    by_arm = {s['arm']: s for s in summaries}
    uniform = by_arm['uniform']
    for s in summaries:
        s['delta_vs_uniform'] = {key: s[key] - uniform[key]
            for key in ('psnr_db', 'ssim', 'lpips', 'denoise_seconds')}
    wins = {arm: dict(psnr=0, lpips=0) for arm in WORKER_ARMS}
    for case in CASES:
        group = [r for r in rows if r['case'] == case]
        wins[max(group, key=lambda r: r['psnr_db'])['arm']]['psnr'] += 1
        wins[min(group, key=lambda r: r['lpips'])['arm']]['lpips'] += 1
    slopes = {'uniform': 0.0, 'step_13to7': 6.5, **{a: float(d) for a, d in SLOPES.items()}}
    response = sorted((slopes[s['arm']], s['arm'], s['psnr_db'], s['lpips'], s['ssim'])
                      for s in summaries)
    psnr_peak_d, psnr_peak_arm = max(((d, a) for d, a, p, l, s in response if d > 0),
                                     key=lambda x: by_arm[x[1]]['psnr_db'])
    best_lpips_d, best_lpips_arm, _, _, _ = min(response, key=lambda x: x[3])
    write(EXP / 'results.json', dict(status='complete', summary=summaries, rows=rows, wins=wins,
        slope_response=[dict(d=d, arm=a, psnr_db=p, lpips=l, ssim=s) for d, a, p, l, s in response]))
    lines = ['# Linear-slope scan of step-budget ramps: 5s 480p', '',
        'Four prompts, seed 42, 120 frames at 24 fps, 832x480, 20 requested steps (19 evaluations). '
        'GPUs 0-3. First four evaluations and layer 0 remain dense. Every arm spends exactly the uniform '
        'grand total (735 calls, 905,520 routed blocks per generation; per-head sum exactly 330 over the '
        '15 sparse evaluations). Arms are symmetric linear ramps 22+-d blocks/head across evaluations '
        '4->18. All arms use the same explicit stable Top-K score sort with exact per-head budgets under '
        'the production Spark reweighting branch. The uniform and step_13to7 baselines and the dense '
        'references are reused from the ramp-budget experiment after config and SHA256 verification.', '',
        '## Per-evaluation budgets (blocks/head)', '',
        '| Evaluation | uniform | step_13to7 | 30to14 | 33to11 | 36to8 | 39to5 | 42to2 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in SPARSE_EVALS:
        b = arm_budgets
        lines.append(f"| {e} | {b('uniform')[e]} | {b('step_13to7')[e]} | {b('step_lin_30to14')[e]} | "
                     f"{b('step_lin_33to11')[e]} | {b('step_lin_36to8')[e]} | {b('step_lin_39to5')[e]} | "
                     f"{b('step_lin_42to2')[e]} |")
    lines += ['', '## Averages over four prompts', '',
        '| Arm | d | PSNR | SSIM | LPIPS | Denoise seconds | dPSNR vs uniform | dSSIM | dLPIPS | PSNR wins | LPIPS wins |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        du = s['delta_vs_uniform']
        w = wins[s['arm']]
        lines.append(f"| {s['arm']} | {slopes[s['arm']]:g} | {s['psnr_db']:.4f} | {s['ssim']:.6f} | "
                     f"{s['lpips']:.6f} | {s['denoise_seconds']:.2f} | {du['psnr_db']:+.4f} | "
                     f"{du['ssim']:+.6f} | {du['lpips']:+.6f} | {w['psnr']}/4 | {w['lpips']}/4 |")
    lines += ['', '## Slope response', '',
        '| d (blocks/head) | Arm | Mean PSNR | Mean SSIM | Mean LPIPS |', '|---|---|---:|---:|---:|']
    for d, a, p, l, s in response:
        lines.append(f'| {d:g} | {a} | {p:.4f} | {s:.6f} | {l:.6f} |')
    lines += ['',
        f'PSNR peaks at d={psnr_peak_d:g} (`{psnr_peak_arm}`, {by_arm[psnr_peak_arm]["psnr_db"]:.4f} dB). '
        f'Best mean LPIPS at d={best_lpips_d:g} (`{best_lpips_arm}`, {by_arm[best_lpips_arm]["lpips"]:.6f}). ',
        '']
    lines += ['## Per-prompt results', '',
        '| Case | Arm | PSNR | SSIM | LPIPS | Denoise seconds |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']:02} | {r['arm']} | {r['psnr_db']:.4f} | {r['ssim']:.6f} | "
                     f"{r['lpips']:.6f} | {r['denoise_seconds']:.2f} |")
    lines += ['', '## Cross-experiment context (B2 exponential arms)', '']
    b2_file = B2 / 'results.json'
    if b2_file.is_file():
        b2 = read(b2_file)
        b2s = {s['arm']: s for s in b2['summary']}
        lines.append('In B2 (same setup), geometric decays b_e ~ rate^(e-4) gave mean PSNR/LPIPS: '
            + '; '.join(f"{a} {b2s[a]['psnr_db']:.4f}/{b2s[a]['lpips']:.4f}"
                        for a in ('step_exp_r95', 'step_exp_r93', 'step_exp_r90', 'step_importance')
                        if a in b2s)
            + f", versus linear step_13to7 {b2s.get('step_13to7', {}).get('psnr_db', float('nan')):.4f}"
            f"/{b2s.get('step_13to7', {}).get('lpips', float('nan')):.4f} and uniform "
            f"{b2s.get('uniform', {}).get('psnr_db', float('nan')):.4f}"
            f"/{b2s.get('uniform', {}).get('lpips', float('nan')):.4f}. "
            'The slope scan above localizes the linear-family optimum; the B2 numbers show how the '
            'geometric family and the same-case oracle compare at the same total budget.')
    else:
        lines.append('B2 results.json was not available at report time.')
    lines += ['', '## Validation and limits', '',
        f"{len(rows)} generations total ({len(REUSED_ARMS)} arms x 4 cases reused and verified, "
        f"{len(NEW_ARMS)} arms x 4 cases new): every sparse allocation call kept its arm-schedule "
        'per-evaluation total and every grand total was exactly 905,520 blocks. All FFV1 archives passed '
        'frame, dimension, frame-rate, exact RGB, and audio verification. All ramps redistribute the '
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
            expected_records=32, expected_quality=28))
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
        write(EXP / 'status.json', dict(status='complete', new_generations=20, quality=28))
    elif args.stage in ('smoke', 'worker'):
        globals()[args.stage](args.slot)
    elif args.stage == 'finish':
        for ci in range(args.slot, len(CASES), len(GPUS)):
            base.decode(ci)
            base.score(ci)
    else:
        globals()[args.stage]()
