"""Reverse-probe layer ablation on a dense background: 5s/480p, GPUs 0-3.

Mirror of dense_layer_ablation with the direction flipped: each arm
`reverse_sparse_layer_LL` runs layers 0-49 dense EXCEPT layer LL, which takes the
sparse path at uniform 10% (22 blocks/head, exact 1,232 blocks per sparse call,
evaluations 4-18; warmup evaluations 0-3 dense as usual). The dense background is
realized with config.sol_extra_dense_layers = all 50 layers except LL, while
sol_dense_layers keeps its default of 1 (layer 0 hits that gate first and is dense
in every arm; the allocator never sees it). The allocator serves only layer LL:
15 calls per generation. Scoring is against the full-dense reference per case
(PSNR/SSIM/LPIPS vs dense: larger PSNR = less damage).

Arm set (12): from the forward probe ranking (dense_layer_ablation results.json):
top-5 {45, 3, 24, 37, 1}, bottom-5 {22, 25, 12, 28, 29}, and the two layers whose
forward dPSNR is closest to the median of the 49 (7 and 40). This tests ranking
consistency and additivity between sparse-background gains and dense-background
damage. Stages: prepare/smoke/worker/finish/report/run.
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
EXP = Path('/autodl-fs/data/h3_experiments/reverse_layer_probe_5s_480p_20260921')
OUT = Path('/autodl-fs/data/h3_outputs') / EXP.name
BASELINE = Path('/autodl-fs/data/h3_experiments/ramp_budget_5s_480p_20260920')
FORWARD = Path('/autodl-fs/data/h3_experiments/dense_layer_ablation_5s_480p_20260920/results.json')
WEIGHTS = Path('/autodl-fs/data/h3_experiments/dense_layer_ablation_5s_480p_20260920/weight_attribution.json')
CASES = base.CASES
FRAMES, HEIGHT, WIDTH = 120, 480, 832
HEADS, BLOCKS = 56, 225
CALL_TOTAL = HEADS * 22  # 1,232 blocks per sparse (evaluation, layer)
SPARSE_EVALUATIONS = 15  # evaluations 4..18
GPUS = (0, 1, 2, 3)
read, write, sha = base.read, base.write, base.sha


def forward_data():
    results = read(FORWARD)
    gains = {int(k): v for k, v in results['layer_gains'].items()}
    rows = results['rows']
    per_case = {}
    for layer in gains:
        per_case[layer] = {}
        for r in rows:
            if r['arm'] == f'dense_layer_{layer:02}':
                u = next(x for x in rows if x['arm'] == 'uniform' and x['case'] == r['case'])
                per_case[layer][r['case']] = r['psnr_db'] - u['psnr_db']
    return gains, per_case


def arm_set():
    gains, _ = forward_data()
    ordered = sorted(gains, key=lambda l: -gains[l])
    top5 = ordered[:5]
    bottom5 = ordered[-5:]
    median = statistics.median(gains.values())
    median2 = sorted((l for l in gains if l not in top5 + bottom5),
                     key=lambda l: abs(gains[l] - median))[:2]
    return top5, bottom5, sorted(median2), median


TOP5, BOTTOM5, MEDIAN2, FORWARD_MEDIAN = arm_set()
PROBE_LAYERS = tuple(sorted(TOP5 + BOTTOM5 + MEDIAN2))
ARMS = tuple(f'reverse_sparse_layer_{l:02}' for l in PROBE_LAYERS)


def configure():
    base.REPO = REPO
    base.MODEL = MODEL
    base.EXP, base.OUT = EXP, OUT
    base.SHAPES = {480: (832, 225, 'head_sparsity_all50_20260919')}
    base.ARMS = ('dense',) + ARMS


class SingleLayerAllocator:
    """Exact 22 blocks per head for the one sparse layer, 15 calls per generation."""

    def __init__(self, layer):
        self.layer = layer
        self.records = []

    def __call__(self, layer, evaluation, values, video_tokens):
        heads = values.shape[1]
        blocks = video_tokens // 64
        if heads != HEADS or blocks != BLOCKS:
            raise ValueError('unexpected head count or video-block resolution')
        if layer != self.layer or not 4 <= evaluation <= 18:
            raise ValueError('allocator called outside the designated sparse layer/evaluations')
        budget = np.full(heads, 22, dtype=np.int64)
        self.records.append(dict(layer=layer, evaluation=evaluation, blocks=blocks,
                                 total=CALL_TOTAL, budgets=budget.tolist()))
        import torch
        return torch.as_tensor(budget, device=values.device, dtype=torch.int64)


def arm_layer(arm):
    return int(arm.removeprefix('reverse_sparse_layer_'))


def arm_dense_layers(arm):
    return tuple(l for l in range(50) if l != arm_layer(arm))


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
    # Dense references (records + lossless videos): reuse from the ramp-budget
    # experiment after SHA256 verification. Scoring is against these references.
    for case in CASES:
        stem = f'480_dense_{case:02}'
        for folder in ('records', 'decoded'):
            record = read(BASELINE / folder / f'{stem}.json')
            if folder == 'decoded':
                assert Path(record['video_path']).is_file() and sha(record['video_path']) == record['sha256']
            write(EXP / folder / f'{stem}.json', record)
    write(EXP / 'protocol.json', dict(cases=cases, arms=['dense', *ARMS],
        frames=FRAMES, fps=24, height=HEIGHT, width=WIDTH, steps=20, seed=42, gpus=list(GPUS),
        heads=HEADS, blocks=BLOCKS, uniform_per_head=22, exact_total=CALL_TOTAL,
        sparse_evaluations='4..18', sparse_layer_calls_per_generation=SPARSE_EVALUATIONS,
        dense_background=('config.sol_extra_dense_layers = all 50 layers except the probed one; '
                          'sol_dense_layers keeps its default of 1 so layer 0 hits the dense-layer '
                          'gate first and is dense in every arm (allocator never sees it). '
                          'Expected processor counts per generation: dense:dense_layer 15 (layer 0), '
                          'dense:extra_dense_layer 720 (48 layers x 15), allocator calls 15.'),
        arm_selection=dict(
            forward_results=str(FORWARD),
            top5=TOP5, bottom5=BOTTOM5, median2=MEDIAN2, forward_median_dpsnr=FORWARD_MEDIAN,
            why=('Extreme subset of the forward 49-arm ranking plus the two layers closest to the '
                 'forward median dPSNR, to test ranking consistency and additivity across backgrounds.')),
        model=str(MODEL), environment=base.ENV,
        model_index_sha256={folder: sha(MODEL / folder / 'diffusion_pytorch_model.safetensors.index.json')
                            for folder in ('transformer', 'vae')},
        references=dict(source=str(BASELINE),
                        reused='Dense records and lossless videos, SHA256 verified; scoring is against them'),
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
                layer = arm_layer(arm)
                cfg = h3.H3SparseAttentionConfig.spark(20, sol_log_density=False,
                                                       sol_extra_dense_layers=arm_dense_layers(arm))
                allocator = SingleLayerAllocator(layer)
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
                assert len(allocator.records) == SPARSE_EVALUATIONS, len(allocator.records)
                assert all(r['total'] == CALL_TOTAL and r['layer'] == layer for r in allocator.records)
                # Layer 0 hits the sol_dense_layers gate; the other 48 dense layers
                # come from sol_extra_dense_layers.
                assert summary['processor_calls'].get('dense:dense_layer', 0) == SPARSE_EVALUATIONS
                assert summary['processor_calls'].get('dense:extra_dense_layer', 0) == (
                    SPARSE_EVALUATIONS * 48)
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
    generate(slot, range(GPUS.index(slot), len(CASES), len(GPUS)), ARMS)


def smoke(slot):
    """reverse_sparse_layer_45 for case 3, end to end through scoring."""
    assert slot == GPUS[0] and CASES[0] == 3
    generate(slot, (0,), ('reverse_sparse_layer_45',))
    full_arms = base.ARMS
    base.ARMS = ('dense', 'reverse_sparse_layer_45')
    try:
        base.decode(0)
        base.score(0)
    finally:
        base.ARMS = full_arms


def report():
    forward_gains, forward_per_case = forward_data()
    rows, summaries = [], []
    for case, arm in itertools.product(CASES, ARMS):
        stem = f'480_{arm}_{case:02}'
        q = read(EXP / 'quality' / f'{stem}.json')
        r = read(EXP / 'records' / f'{stem}.json')
        d = read(EXP / 'decoded' / f'{stem}.json')
        layer = arm_layer(arm)
        assert len(r['calls']) == SPARSE_EVALUATIONS
        assert all(c['total'] == CALL_TOTAL and c['layer'] == layer for c in r['calls'])
        assert d['evidence']['pyav_pixel_equal'] and d['evidence']['ffmpeg_pixel_equal']
        rows.append({**q, 'denoise_seconds': r['denoise_seconds'],
                     'forward_dpsnr': forward_per_case[layer][case]})
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        summaries.append(dict(arm=arm, layer=arm_layer(arm), **{key: statistics.mean(r[key] for r in group)
            for key in ('psnr_db', 'ssim', 'lpips', 'denoise_seconds')}))
    for s in summaries:
        s['forward_dpsnr'] = forward_gains[s['layer']]
        s['damage_db'] = -s['psnr_db']  # larger = more damage when removed from dense
    ranking = sorted(summaries, key=lambda s: s['damage_db'])
    from scipy.stats import pearsonr, spearmanr
    x = [s['forward_dpsnr'] for s in summaries]
    y = [s['damage_db'] for s in summaries]
    overall = dict(spearman=float(spearmanr(x, y).statistic),
                   pearson=float(pearsonr(x, y).statistic))
    per_case = {}
    for case in CASES:
        group = [r for r in rows if r['case'] == case]
        per_case[case] = float(spearmanr([r['forward_dpsnr'] for r in group],
                                         [-r['psnr_db'] for r in group]).statistic)
    # Sign flips: forward says above/below the 12-arm forward median; reverse says
    # above/below the 12-arm damage median. Disagreement = flipped.
    fmed = statistics.median(s['forward_dpsnr'] for s in summaries)
    dmed = statistics.median(s['damage_db'] for s in summaries)
    flips = [s for s in summaries
             if (s['forward_dpsnr'] > fmed) != (s['damage_db'] > dmed)]
    weight_notes = []
    if WEIGHTS.is_file():
        w = read(WEIGHTS)
        features, population = w['features'], {}
        for name in ('static_sigma_max', 'static_projected_v_energy_mean',
                     'static_gaussian_kr95_mean', 'wo_frob_max', 'wq_frob_cv', 'wk_frob_cv'):
            values = np.array([features[name][str(l)] if str(l) in features[name]
                               else features[name][l] for l in range(1, 50)])
            mu, sd = values.mean(), values.std()
            population[name] = (mu, sd, features[name])
        flipped_layers = {s['layer'] for s in flips}
        held = [s['layer'] for s in summaries if s['layer'] not in flipped_layers]
        for name, (mu, sd, values) in population.items():
            def z(layer):
                v = values[str(layer)] if str(layer) in values else values[layer]
                return (v - mu) / sd
            entry = dict(feature=name,
                flipped={l: float(z(l)) for l in sorted(flipped_layers)},
                held_mean_z=float(statistics.mean(z(l) for l in held)) if held else None)
            weight_notes.append(entry)
    write(EXP / 'results.json', dict(status='complete', summary=summaries, rows=rows,
        damage_ranking=[s['arm'] for s in ranking],
        forward_gains={l: forward_gains[l] for l in PROBE_LAYERS},
        consistency=dict(overall=overall, per_case_spearman=per_case,
                         forward_median=fmed, damage_median=dmed,
                         flipped_layers=[s['layer'] for s in flips]),
        weight_cross=weight_notes))
    lines = ['# Reverse-probe layer ablation on a dense background: 5s 480p', '',
        'Four prompts, seed 42, 120 frames at 24 fps, 832x480, 20 requested steps (19 evaluations). '
        'GPUs 0-3. Each `reverse_sparse_layer_LL` arm runs layers 0-49 through the production dense '
        'path EXCEPT layer LL, which takes the sparse path at uniform 10% (22 blocks/head, exact 1,232 '
        'blocks, evaluations 4-18; warmup evaluations 0-3 dense). The dense background uses '
        '`sol_extra_dense_layers` = all 50 layers except LL with `sol_dense_layers=1` (layer 0 dense in '
        'every arm; the allocator only ever serves layer LL, 15 calls per generation). Metrics are '
        'against the full-dense reference per case (SHA256-verified reuse from the ramp-budget '
        'experiment): larger PSNR-vs-dense = less damage. Arm set: forward top-5 '
        f"{{{', '.join(map(str, TOP5))}}}, bottom-5 {{{', '.join(map(str, BOTTOM5))}}}, and the two "
        f"layers closest to the forward median dPSNR ({FORWARD_MEDIAN:+.5f}): "
        f"{{{', '.join(map(str, MEDIAN2))}}}.", '',
        '## Reverse damage, ranked (means over four prompts; damage = -PSNR vs dense)', '',
        '| Rank | Layer | PSNR vs dense | SSIM vs dense | LPIPS vs dense | Forward dPSNR | Denoise seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for rank, s in enumerate(ranking, 1):
        lines.append(f"| {rank} | {s['layer']} | {s['psnr_db']:.4f} | {s['ssim']:.6f} | {s['lpips']:.6f} | "
                     f"{s['forward_dpsnr']:+.4f} | {s['denoise_seconds']:.2f} |")
    lines += ['',
        '## Forward-vs-reverse consistency (12 layers)', '',
        'Forward: dPSNR from adding one dense layer to the uniform-sparse baseline '
        '(dense_layer_ablation). Reverse: damage from removing one layer from the dense trajectory '
        '(-PSNR vs dense). Consistent rankings mean a POSITIVE correlation between forward gain and '
        'reverse damage.', '',
        f"Overall: Spearman {overall['spearman']:+.4f}, Pearson {overall['pearson']:+.4f}. "
        'Per-case Spearman: ' + ', '.join(f"case {c:02} {r:+.4f}" for c, r in per_case.items()) + '. ',
        ('The ranking HOLDS on the dense background: layer importance is approximately additive, '
         'so per-layer importance measured in either regime transfers.'
         if overall['spearman'] >= 0.6 else
         'The ranking PARTIALLY holds: monotone association is moderate, so additivity is approximate '
         'at best.' if overall['spearman'] >= 0.3 else
         'The ranking FLIPS on the dense background: interaction-dominated — importance measured on '
         'the sparse baseline does not transfer to dense.'), '',
        f"Sign flips (forward above/below the 12-arm median {fmed:+.4f} disagrees with reverse "
        f"damage above/below its median {dmed:+.4f}): "
        + (', '.join(f"layer {s['layer']} (forward {s['forward_dpsnr']:+.4f}, "
                     f"PSNR-vs-dense {s['psnr_db']:.4f})" for s in flips)
           if flips else 'none.'), '']
    if weight_notes:
        lines += ['## Weight-signature cross-check (flipped vs held layers)', '',
            'z-scores of weight-only features (from the forward experiment weight attribution, '
            '49-layer population) for the sign-flipped layers vs the mean z of the held layers. '
            'Correlational only; 12 layers, multiple features.', '',
            '| Feature | Flipped-layer z-scores | Held mean z |', '|---|---|---:|']
        for note in weight_notes:
            zs = ', '.join(f"L{l} {z:+.2f}" for l, z in note['flipped'].items()) or '—'
            held = f"{note['held_mean_z']:+.2f}" if note['held_mean_z'] is not None else '—'
            lines.append(f"| {note['feature']} | {zs} | {held} |")
        lines.append('')
    lines += ['## Synthesis for budget allocation', '',
        ('Forward gains and reverse damage agree in rank: the per-layer importance ordering is '
         'approximately additive across backgrounds. A single importance probe (in either regime) '
         'suffices to order layers for budget reallocation; the layer_7to13 ramp win is consistent '
         'with this reading.' if overall['spearman'] >= 0.6 else
         'Forward gains and reverse damage do not fully agree in rank: layer importance is at least '
         'partly interaction-dominated (a layer’s value depends on whether the rest of the stack is '
         'dense or sparse). Budget allocation should therefore be optimized and validated in the '
         'target sparse regime; dense-background probes alone can misorder layers.'), '',
        '## Per-prompt results', '',
        '| Case | Layer | PSNR vs dense | SSIM vs dense | LPIPS vs dense | Forward dPSNR (case) |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']:02} | {arm_layer(r['arm'])} | {r['psnr_db']:.4f} | {r['ssim']:.6f} | "
                     f"{r['lpips']:.6f} | {r['forward_dpsnr']:+.4f} |")
    lines += ['', '## Validation and limits', '',
        '48 reverse-probe generations: the allocator was called exactly 15 times per generation '
        '(only the probed layer, exact 1,232 blocks per call), layer 0 took the dense-layer gate '
        '(15 calls) and the remaining 48 dense layers the extra-dense gate (720 calls) in every '
        'generation. 48 lossless FFV1 archives passed frame, dimension, frame-rate, exact RGB, and '
        'audio verification. The 12-arm subset covers the forward extremes and median; the middle '
        'of the ranking is extrapolated, not measured. Four prompts and one seed; damage metrics '
        'are against dense references decoded from the same checkpoint. Timings include '
        'allocation/audit overhead.', '',
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
            expected_records=52, expected_quality=48))
        time.sleep(10)
    codes = [p.wait() for p in processes]
    if any(codes):
        write(EXP / 'status.json', dict(status='failed', stage=stage, codes=codes))
        raise RuntimeError((stage, codes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'smoke', 'worker', 'finish', 'report', 'run'))
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
        write(EXP / 'status.json', dict(status='complete', probe_generations=48, quality=48))
    elif args.stage in ('smoke', 'worker'):
        globals()[args.stage](args.slot)
    elif args.stage == 'finish':
        for ci in range(GPUS.index(args.slot), len(CASES), len(GPUS)):
            base.decode(ci)
            base.score(ci)
    else:
        globals()[args.stage]()
