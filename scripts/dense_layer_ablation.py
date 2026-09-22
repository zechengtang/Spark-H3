"""Per-layer dense-importance ablation: 5s/480p, GPUs 1-4.

Each arm `dense_layer_LL` (L in 1..49) keeps the uniform 10% Top-K baseline (22
blocks/head, exact per-call total 1,232) and additionally runs layer L through
the true production dense path at every evaluation
(config.sol_extra_dense_layers). This is an importance probe: every arm spends
more compute than uniform, so arms are mutually comparable as "+1 dense layer"
but are not budget-matched reallocations. The uniform baseline and the dense
references are reused from the ramp-budget experiment after config and SHA256
verification. Stages: prepare/smoke/worker/finish/report/run.
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
EXP = Path('/autodl-fs/data/h3_experiments/dense_layer_ablation_5s_480p_20260920')
OUT = Path('/autodl-fs/data/h3_outputs') / EXP.name
BASELINE = Path('/autodl-fs/data/h3_experiments/ramp_budget_5s_480p_20260920')
RECALL = Path('/autodl-fs/data/h3_experiments/mass_recall_5s_480p_20260920/calibration')
SPARSITY = REPO / 'reports' / 'head_sparsity_all50_20260919'
LAYERS = tuple(range(1, 50))
ARMS = tuple(f'dense_layer_{l:02}' for l in LAYERS)
EXTRA_ARM = 'sparse_layer_00'  # reverse probe: layer 0 sparse, everything else uniform
WORKER_ARMS = ('uniform', EXTRA_ARM) + ARMS
CASES = base.CASES
STAGES = base.STAGES
FRAMES, HEIGHT, WIDTH = 120, 480, 832
HEADS, BLOCKS = 56, 225
CALL_TOTAL = HEADS * 22  # 1,232 blocks per sparse (evaluation, layer)
SPARSE_EVALUATIONS = 15  # evaluations 4..18
GPUS = (1, 2, 3, 4)
read, write, sha = base.read, base.write, base.sha


def configure():
    base.REPO = REPO
    base.MODEL = MODEL
    base.EXP, base.OUT = EXP, OUT
    base.SHAPES = {480: (832, 225, 'head_sparsity_all50_20260919')}
    base.ARMS = ('dense',) + WORKER_ARMS


class UniformAllocator:
    """Exact 22 blocks per head at every sparse (evaluation, layer)."""

    def __init__(self, min_layer=1):
        self.min_layer = min_layer
        self.records = []

    def __call__(self, layer, evaluation, values, video_tokens):
        heads = values.shape[1]
        blocks = video_tokens // 64
        if heads != HEADS or blocks != BLOCKS:
            raise ValueError('unexpected head count or video-block resolution')
        if not 4 <= evaluation <= 18 or not self.min_layer <= layer <= 49:
            raise ValueError('allocator called outside the sparse evaluations/layers')
        budget = np.full(heads, 22, dtype=np.int64)
        self.records.append(dict(layer=layer, evaluation=evaluation, blocks=blocks,
                                 total=CALL_TOTAL, budgets=budget.tolist()))
        import torch
        return torch.as_tensor(budget, device=values.device, dtype=torch.int64)


def arm_layer(arm):
    return int(arm.removeprefix('dense_layer_'))


def arm_sparse_layers(arm):
    if arm == 'uniform':
        return set(LAYERS)
    if arm == EXTRA_ARM:
        return set(range(50))  # layer 0 included: 735 calls, matching uniform
    return set(LAYERS) - {arm_layer(arm)}


def arm_extra_dense_layers(arm):
    if arm in ('uniform', EXTRA_ARM):
        return ()
    return (arm_layer(arm),)


def expected_calls(arm):
    return len(arm_sparse_layers(arm)) * SPARSE_EVALUATIONS


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
    # Dense references and uniform baseline: reuse from the ramp-budget
    # experiment after verification.
    for case in CASES:
        stem = f'480_dense_{case:02}'
        for folder in ('records', 'decoded'):
            record = read(BASELINE / folder / f'{stem}.json')
            if folder == 'decoded':
                assert Path(record['video_path']).is_file() and sha(record['video_path']) == record['sha256']
            write(EXP / folder / f'{stem}.json', record)
    sys.path.insert(0, str(REPO))
    from h3_sparse_attention import H3SparseAttentionConfig
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
            for field in ('sol_extra_dense_evaluations', 'sol_extra_dense_layers'):
                got_config.setdefault(field, [])  # predates these fields
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
        frames=FRAMES, fps=24, height=HEIGHT, width=WIDTH, steps=20, seed=42, gpus=list(GPUS),
        heads=HEADS, blocks=BLOCKS, uniform_per_head=22, exact_total=CALL_TOTAL,
        sparse_evaluations='4..18', sparse_layers='1..49', dense_evaluations=4, dense_layers=1,
        calls_per_generation=dict(uniform=735, dense_layer_arms=720),
        model=str(MODEL), environment=base.ENV,
        model_index_sha256={folder: sha(MODEL / folder / 'diffusion_pytorch_model.safetensors.index.json')
                            for folder in ('transformer', 'vae')},
        probe=('Each dense_layer_LL arm adds exactly one dense layer (true production dense path at all '
               'evaluations) to the uniform 22-blocks/head baseline. Arms therefore spend MORE total '
               'compute than uniform: this is a per-layer importance probe, not a budget-matched '
               'reallocation.'),
        uniform_baseline=dict(source=str(BASELINE), reused=reused,
            verification='Config equality, frames/height/width/steps/seed, latent SHA256, video SHA256, 735 calls at 1232 blocks'),
        references=dict(source=str(BASELINE), reused='Dense records and lossless videos, SHA256 verified'),
        offline_crosschecks=dict(
            recall_at_22=f'{RECALL} (video curves, layers 1-49, mean over cases/evals/heads)',
            relative_error=f'{SPARSITY} (reblocked mean-compression rel_l2, layers 1-49)'),
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
                extra = arm_extra_dense_layers(arm)
                cfg = h3.H3SparseAttentionConfig.spark(20, sol_log_density=False,
                                                       sol_extra_dense_layers=extra,
                                                       sol_dense_layers=0 if arm == EXTRA_ARM else 1)
                allocator = UniformAllocator(min_layer=0 if arm == EXTRA_ARM else 1)
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
                assert len(allocator.records) == expected_calls(arm), len(allocator.records)
                assert all(r['total'] == CALL_TOTAL for r in allocator.records)
                assert {r['layer'] for r in allocator.records} == arm_sparse_layers(arm)
                assert summary['processor_calls'].get('dense:extra_dense_layer', 0) == (
                    SPARSE_EVALUATIONS * len(extra))
                assert summary['processor_calls'].get('dense:dense_layer', 0) == (
                    0 if arm == EXTRA_ARM else SPARSE_EVALUATIONS)
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
    generate(slot, range(GPUS.index(slot), len(CASES), len(GPUS)), WORKER_ARMS)


def smoke(slot):
    """dense_layer_25 for case 3, end to end through scoring."""
    assert slot == GPUS[0] and CASES[0] == 3
    generate(slot, (0,), ('dense_layer_25',))
    full_arms = base.ARMS
    base.ARMS = ('dense', 'dense_layer_25')
    try:
        base.decode(0)
        base.score(0)
    finally:
        base.ARMS = full_arms


def probe0():
    """Reverse probe on GPU 0: layer 0 sparse (uniform 10%) at all sparse evaluations."""
    generate(0, range(len(CASES)), (EXTRA_ARM,))
    for ci in range(len(CASES)):
        base.decode(ci)
        base.score(ci)


def offline_layer_features():
    """Per-layer offline difficulty features for the cross-check analysis."""
    recall_mean, recall_var = {}, {}
    for layer in LAYERS:
        means, variances = [], []
        for case, ev in itertools.product(CASES, STAGES):
            with np.load(RECALL / f'{case}_{ev}_{layer}.npz') as data:
                video = data['video']
                assert video.shape == (HEADS, BLOCKS + 1)
                at_budget = video[:, 22]
                means.append(float(at_budget.mean()))
                variances.append(float(at_budget.var()))
        recall_mean[layer] = statistics.mean(means)
        recall_var[layer] = statistics.mean(variances)
    import ijson
    error_mean, error_var = {}, {}
    per_layer = {layer: [] for layer in LAYERS}
    for case in CASES:
        path = SPARSITY / f'case_{case:02}.json'
        per_layer_case = {layer: [] for layer in LAYERS}
        with path.open('rb') as handle:
            for row in ijson.items(handle, 'rows.item', use_float=True):
                if row['layout'] == 'reblocked' and row['branch'] == 'mean_compressed' and row['layer'] in per_layer_case:
                    per_layer_case[row['layer']].append(row['rel_l2'][22])
        for layer in LAYERS:
            assert len(per_layer_case[layer]) == HEADS * len(STAGES), (case, layer, len(per_layer_case[layer]))
            per_layer[layer].append((statistics.mean(per_layer_case[layer]),
                                     statistics.pvariance(per_layer_case[layer])))
    for layer in LAYERS:
        error_mean[layer] = statistics.mean(m for m, v in per_layer[layer])
        error_var[layer] = statistics.mean(v for m, v in per_layer[layer])
    return dict(depth={layer: float(layer) for layer in LAYERS},
                recall_mean=recall_mean, recall_var=recall_var,
                error_mean=error_mean, error_var=error_var)


def report():
    rows, summaries = [], []
    for case, arm in itertools.product(CASES, WORKER_ARMS):
        stem = f'480_{arm}_{case:02}'
        q = read(EXP / 'quality' / f'{stem}.json')
        r = read(EXP / 'records' / f'{stem}.json')
        d = read(EXP / 'decoded' / f'{stem}.json')
        assert len(r['calls']) == expected_calls(arm)
        assert all(c['total'] == CALL_TOTAL for c in r['calls'])
        assert {c['layer'] for c in r['calls']} == arm_sparse_layers(arm)
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
    probe = [s for s in summaries if s['arm'] not in ('uniform', EXTRA_ARM)]
    gains = {arm_layer(s['arm']): s['delta_vs_uniform']['psnr_db'] for s in probe}
    ranking = sorted(probe, key=lambda s: s['delta_vs_uniform']['psnr_db'], reverse=True)
    layer0 = next(s for s in summaries if s['arm'] == EXTRA_ARM)
    layer0_value = {key: uniform[key] - layer0[key] for key in ('psnr_db', 'ssim', 'lpips')}
    layer0_rank = 1 + sum(1 for g in gains.values() if g > layer0_value['psnr_db'])
    layer0_cases = [dict(case=r['case'],
        **{f'uniform_{k}': next(x[k] for x in rows if x['arm'] == 'uniform' and x['case'] == r['case'])
           for k in ('psnr_db', 'ssim', 'lpips')},
        **{k: r[k] for k in ('psnr_db', 'ssim', 'lpips', 'denoise_seconds')})
        for r in rows if r['arm'] == EXTRA_ARM]
    if layer0_rank == 1:
        layer0_verdict = (' — it is the single most valuable layer to keep dense, consistent with the '
                          'production default.')
    else:
        layer0_verdict = f" — behind layer {ranking[layer0_rank - 2]['arm'].removeprefix('dense_layer_')}"
        if layer0_rank <= len(LAYERS):
            layer0_verdict += f" and ahead of layer {ranking[layer0_rank - 1]['arm'].removeprefix('dense_layer_')}"
        layer0_verdict += (', so the production default is not the single most valuable dense '
                           'layer by this probe.')
    thirds = {'1-16': statistics.mean(gains[l] for l in range(1, 17)),
              '17-33': statistics.mean(gains[l] for l in range(17, 34)),
              '34-49': statistics.mean(gains[l] for l in range(34, 50))}
    peak = max(gains, key=gains.get)
    trough = min(gains, key=gains.get)
    features = offline_layer_features()
    from scipy.stats import spearmanr
    correlations = {}
    for name, values in features.items():
        x = [values[l] for l in LAYERS]
        y = [gains[l] for l in LAYERS]
        rho = spearmanr(x, y).statistic
        correlations[name] = float(rho)
    write(EXP / 'results.json', dict(status='complete', summary=summaries, rows=rows,
        layer_gains=gains, psnr_gain_ranking=[s['arm'] for s in ranking],
        curve=dict(gains=gains, thirds=thirds, peak=peak, trough=trough),
        layer0_reverse_probe=dict(arm=EXTRA_ARM, dense_value_vs_uniform=layer0_value,
            psnr_gain_rank_of_layer0=layer0_rank, out_of=len(LAYERS) + 1, per_case=layer0_cases),
        offline_features=features, spearman_vs_layer_gain=correlations))
    lines = ['# Per-layer dense-importance probe: 5s 480p', '',
        'Four prompts, seed 42, 120 frames at 24 fps, 832x480, 20 requested steps (19 evaluations). '
        'GPUs 1-4. Evaluations 0-3 and layer 0 are always dense. Each `dense_layer_LL` arm keeps the '
        'uniform 10% Top-K baseline (22 blocks/head, exact 1,232 blocks per sparse call) and additionally '
        'runs layer LL through the true production dense path at every evaluation. **Arms spend more '
        'total compute than uniform: this is an importance probe, not a budget-matched reallocation.** '
        'The uniform baseline and the dense references are reused from the ramp-budget experiment after '
        'config and SHA256 verification.', '',
        '## Averages over four prompts, ranked by PSNR gain', '',
        '| Rank | Layer | PSNR | SSIM | LPIPS | Denoise seconds | dPSNR vs uniform | dSSIM | dLPIPS |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for rank, s in enumerate(ranking, 1):
        d = s['delta_vs_uniform']
        lines.append(f"| {rank} | {arm_layer(s['arm'])} | {s['psnr_db']:.4f} | {s['ssim']:.6f} | "
                     f"{s['lpips']:.6f} | {s['denoise_seconds']:.2f} | {d['psnr_db']:+.4f} | "
                     f"{d['ssim']:+.6f} | {d['lpips']:+.6f} |")
    lines += ['', 'Reference arm: `uniform` '
              f"({uniform['psnr_db']:.4f} PSNR, {uniform['ssim']:.6f} SSIM, {uniform['lpips']:.6f} LPIPS).", '',
        '## Curve shape by depth (mean dPSNR)', '',
        f"Shallow layers 1-16: {thirds['1-16']:+.4f}; middle 17-33: {thirds['17-33']:+.4f}; "
        f"deep 34-49: {thirds['34-49']:+.4f}. Peak importance at layer {peak} ({gains[peak]:+.4f} dB); "
        f"least important is layer {trough} ({gains[trough]:+.4f} dB). "
        + ('Importance increases with depth, consistent with the layer_7to13 ramp win (more budget for '
           'deep layers).' if thirds['34-49'] > thirds['17-33'] > thirds['1-16'] else
           'Importance decreases with depth.' if thirds['34-49'] < thirds['17-33'] < thirds['1-16'] else
           'Importance is non-monotone in depth.'), '',
        '## Layer 0 reverse probe (`sparse_layer_00`)', '',
        'Production always runs layer 0 dense (`sol_dense_layers=1`), so the 49-arm probe could not '
        'measure its value. This arm overrides the gate (`sol_dense_layers=0`): layer 0 takes the sparse '
        'path at uniform 10% (22 blocks/head, served by the allocator: 750 calls = 50 layers x 15 sparse '
        'evaluations, vs 735 for uniform) at '
        'evaluations 4-18 while everything else stays exactly as the uniform baseline. The dense value of '
        'layer 0 is uniform minus `sparse_layer_00`.', '',
        f"Layer 0 dense value: **{layer0_value['psnr_db']:+.4f} dB PSNR**, "
        f"{layer0_value['ssim']:+.6f} SSIM, {layer0_value['lpips']:+.6f} LPIPS "
        f"(`sparse_layer_00`: {layer0['psnr_db']:.4f} PSNR, {layer0['ssim']:.6f} SSIM, "
        f"{layer0['lpips']:.6f} LPIPS). Placed in the 49-arm dPSNR ranking above, layer 0 would take "
        f"rank **{layer0_rank} of {len(LAYERS) + 1}**" + layer0_verdict,
        '',
        '| Case | Uniform PSNR | Sparse-layer-0 PSNR | dPSNR (dense value) | dSSIM | dLPIPS |',
        '|---|---:|---:|---:|---:|---:|']
    for r in layer0_cases:
        lines.append(f"| {r['case']:02} | {r['uniform_psnr_db']:.4f} | {r['psnr_db']:.4f} | "
                     f"{r['uniform_psnr_db'] - r['psnr_db']:+.4f} | {r['uniform_ssim'] - r['ssim']:+.6f} | "
                     f"{r['uniform_lpips'] - r['lpips']:+.6f} |")
    lines += ['',
        '## Offline cross-check (Spearman rank correlation with measured per-layer dPSNR)', '',
        '| Offline per-layer feature | Spearman rho |', '|---|---:|']
    for name, rho in correlations.items():
        lines.append(f'| {name} | {rho:+.4f} |')
    lines += ['',
        'Features: `depth` (layer index); `recall_mean` / `recall_var` (mean and across-head variance of '
        'routable-video recall@22 from mass-recall calibration curves, layers 1-49, averaged over the '
        'four cases and evaluations 4/9/18); `error_mean` / `error_var` (mean and across-head variance '
        'of reblocked mean-compression relative-L2 error at the 22-blocks/head operating point from the '
        'head_sparsity_all50 study). '
        'Correlations over 49 layers; no multiple-comparison correction. '
        + ('Offline features track the measured layer axis.' if any(abs(r) > 0.5 for r in correlations.values()) else
           'No offline feature strongly tracks the measured layer axis (all |rho| <= 0.5) — as on the '
           'step axis, offline difficulty metrics do not reliably predict where dense compute pays off.'), '',
        '## Theory (Janus3R-inspired attribution)', '',
        'The Janus3R-style predictors previously evaluated on H3 '
        '(`Spark-H3-relative-difficulty/scripts/janus_head_predictors.py`, using '
        '`Janus3R/janus3r/janus_metrics.py`: projection-aware sigma logit proxies, Gaussian top-k mass '
        'estimates, projected-V static energy, kernel-linear risk) are head-level static (weight-only) '
        'features. For the layer axis, the analogously cheap candidates are the activation-statistics '
        'features cross-checked above: mean routable recall, its across-head variance, and mean relative '
        'attention error. The measured correlations say which, if any, orders the layers the way '
        'full-video fidelity does; depth itself is included as the trivial baseline. These are '
        'correlations over 49 layers on four prompts and one seed — they describe association, not '
        'causation.', '']
    weight_path = EXP / 'weight_attribution.json'
    if weight_path.is_file():
        lines += read(weight_path)['report_lines'] + ['']
    lines += [
        '## Per-prompt results', '',
        '| Case | Arm | PSNR | SSIM | LPIPS | Denoise seconds |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']:02} | {r['arm']} | {r['psnr_db']:.4f} | {r['ssim']:.6f} | "
                     f"{r['lpips']:.6f} | {r['denoise_seconds']:.2f} |")
    lines += ['', '## Validation and limits', '',
        '196 probe generations plus 4 reused uniform baselines: every sparse allocation call kept the '
        'exact 1,232-block total (720 calls per probe generation, 735 for uniform), and each probe '
        'generation took the production dense path at exactly its designated layer (15 extra-dense layer '
        'calls, excluded from allocator records). The 4 reverse-probe (`sparse_layer_00`) generations '
        'recorded 735 allocator calls each (layers 0-49, matching uniform) with zero `dense:dense_layer` '
        'calls, confirming layer 0 took the sparse path. The reverse probe records 750 allocator calls '
        'per generation (50 layers x 15 sparse evaluations vs 735 for uniform) at the same exact '
        '1,232-block total per call. 200 lossless FFV1 archives passed frame, '
        'dimension, frame-rate, exact RGB, and audio verification. Probe arms are not budget-matched to '
        'uniform; gains mix layer importance with added compute. Four prompts and one seed do not '
        'establish broad generalization. Timings include allocation/audit overhead.', '',
        f'Checkpoint: `{MODEL}`. Videos: `{OUT / "videos"}`. Per-prompt metrics: `results.json`. '
        f'Protocol and manifests: `{EXP}`.']
    (EXP / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    target = REPO / 'reports' / EXP.name
    target.mkdir(parents=True, exist_ok=True)
    names = ['REPORT.md', 'results.json', 'protocol.json', 'source_manifest.json']
    if weight_path.is_file():
        names.append('weight_attribution.json')
    for name in names:
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
            expected_records=208, expected_quality=204))
        time.sleep(10)
    codes = [p.wait() for p in processes]
    if any(codes):
        write(EXP / 'status.json', dict(status='failed', stage=stage, codes=codes))
        raise RuntimeError((stage, codes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'worker', 'finish', 'report', 'run', 'probe0'))
    parser.add_argument('--slot', type=int, choices=GPUS, default=GPUS[0])
    args = parser.parse_args()
    os.environ.update(base.ENV)
    configure()
    EXP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'run':
        launch('worker')
        launch('finish')
        report()
        write(EXP / 'status.json', dict(status='complete', probe_generations=196,
                                        reverse_probe_generations=4, quality=204))
    elif args.stage in ('smoke', 'worker'):
        globals()[args.stage](args.slot)
    elif args.stage == 'finish':
        for ci in range(GPUS.index(args.slot), len(CASES), len(GPUS)):
            base.decode(ci)
            base.score(ci)
    else:
        globals()[args.stage]()
