"""Weight-only (Janus3R-style) per-layer attribution for the dense_layer_ablation probe.

Computes per-head weight-only features for all 50 attention layers from the MiniMax-H3
checkpoint (no activations), aggregates them to per-layer scalars (mean/max over 56
heads), and correlates them (Spearman + Pearson) with the measured 49-layer dPSNR
curve from the dense-importance probe (results.json).

Feature recipe follows the weight-only part of
`Spark-H3-relative-difficulty/scripts/janus_head_predictors.py` (its `weights()`
function), using `Janus3R/janus3r/janus_metrics.py`:
- sigma_logit_proxy and log_kernelinear_mse_proxy via `_projection_aware_head_metrics`
  with normalized per-head Q/K weight-variance distributions and zero means;
- static_gaussian_kr95: retained-token ratio for 95% softmax mass from
  `estimate_topk_gaussian_order` at the proxy sigma (14,400 routable video tokens);
- static_projected_v_energy: trace(Wo^T Wo Wv Wv^T)/head_dim per head (the Gram
  construction of `RelativeDifficultyAllocator` in head_budget.py, combined with the
  V-projection Gram so no 5376x5376 map is materialized).
Added here: per-head Frobenius norms of W_q/W_k/W_v/W_o (mean and across-head
coefficient of variation), depth, and attention parameter count as trivial baselines.

Skipped as activation-dependent (not weight-only): runtime full-covariance query
sparse-ratio statistics, key-covariance effective rank, value second moments, and the
measured projected-output energy (all require captured Q/K/V activations).

CPU-only, bf16 -> float32 per tensor. Output: weight_attribution.json next to the
experiment results, including `report_lines` consumed by dense_layer_ablation.report().
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
MODEL = Path('/autodl-fs/data/models/MiniMax-H3/transformer')
EXP = Path('/autodl-fs/data/h3_experiments/dense_layer_ablation_5s_480p_20260920')
JANUS = REPO.parent / 'Janus3R/janus3r/janus_metrics.py'
OUT = EXP / 'weight_attribution.json'
HEADS, HEAD_DIM = 56, 128
VIDEO_TOKENS = 225 * 64  # 14,400 routable video tokens at 5s/480p
ALPHA = 0.95
LAYERS = tuple(range(1, 50))
PROBE_LAYERS = tuple(range(50))
TOP5 = (45, 3, 24, 37, 1)
BOTTOM3 = (22, 25, 12)


def janus_module():
    spec = importlib.util.spec_from_file_location('janus_metrics_reference', JANUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def layer_weights(layer, handles, janus):
    from safetensors import safe_open

    def load(suffix):
        name = f'transformer_blocks.{layer}.{suffix}'
        with safe_open(MODEL / handles[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name).float()

    gq, gk = load('attn.norm_q.weight'), load('attn.norm_k.weight')
    pre = load('norm1.weight')
    wq, wk, wv = (load(f'attn.to_{s}.weight').reshape(HEADS, HEAD_DIM, -1) * pre
                  for s in ('q', 'k', 'v'))
    wo = load('attn.to_out.0.weight').T.reshape(HEADS, HEAD_DIM, -1)
    gram = wo @ wo.transpose(-1, -2)
    vgram = wv @ wv.transpose(-1, -2)
    # trace(Wo^T Wo Wv Wv^T) per head avoids materializing a 5376 x 5376 map.
    static_energy = (gram * vgram).sum((-1, -2)) / HEAD_DIM
    qvar, kvar = wq.square().sum(-1), wk.square().sum(-1)
    frob = {s: w.square().sum((-1, -2)).sqrt() for s, w in
            (('wq', wq), ('wk', wk), ('wv', wv), ('wo', wo))}
    heads = []
    for h in range(HEADS):
        qp, kp = qvar[h] / qvar[h].sum(), kvar[h] / kvar[h].sum()
        metric = janus._projection_aware_head_metrics(
            qp, kp, HEAD_DIM * qp, HEAD_DIM * kp,
            torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM), gq, gk, head_dim=HEAD_DIM)
        sigma = metric['sigma_logit_proxy']
        heads.append(dict(
            static_sigma=float(sigma),
            static_gaussian_kr95=float(janus.estimate_topk_gaussian_order(
                VIDEO_TOKENS, sigma, ALPHA) / VIDEO_TOKENS),
            static_projected_v_energy=float(static_energy[h]),
            static_kernel_risk=float(metric['log_kernelinear_mse_proxy']),
            wq_frob=float(frob['wq'][h]), wk_frob=float(frob['wk'][h]),
            wv_frob=float(frob['wv'][h]), wo_frob=float(frob['wo'][h])))
    n_params = int(sum(w.numel() for w in (wq, wk, wv, wo)) + gq.numel() + gk.numel())
    return heads, n_params


def aggregate(heads_by_layer):
    features = {}
    for name in ('static_sigma', 'static_gaussian_kr95', 'static_projected_v_energy',
                 'static_kernel_risk', 'wq_frob', 'wk_frob', 'wv_frob', 'wo_frob'):
        values = {layer: [h[name] for h in heads_by_layer[layer]] for layer in heads_by_layer}
        features[f'{name}_mean'] = {layer: statistics.mean(v) for layer, v in values.items()}
        features[f'{name}_max'] = {layer: max(v) for layer, v in values.items()}
    for name in ('wq_frob', 'wk_frob'):
        features[f'{name}_cv'] = {
            layer: statistics.pstdev([h[name] for h in heads])
            / statistics.mean([h[name] for h in heads])
            for layer, heads in heads_by_layer.items()}
    features['depth'] = {layer: float(layer) for layer in heads_by_layer}
    return features


def main():
    torch.set_num_threads(16)
    janus = janus_module()
    handles = json.loads(
        (MODEL / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    gains = {int(k): v for k, v in json.loads((EXP / 'results.json').read_text())['layer_gains'].items()}
    assert set(gains) == set(LAYERS)
    heads_by_layer, n_params = {}, {}
    for layer in PROBE_LAYERS:
        heads_by_layer[layer], n_params[layer] = layer_weights(layer, handles, janus)
        print('WEIGHTS', layer, flush=True)
    features = aggregate(heads_by_layer)
    features['param_count'] = {layer: float(n_params[layer]) for layer in PROBE_LAYERS}

    from scipy.stats import pearsonr, spearmanr
    y = np.array([gains[layer] for layer in LAYERS])
    correlations = {}
    for name, values in features.items():
        x = np.array([values[layer] for layer in LAYERS])
        if np.ptp(x) == 0:
            correlations[name] = dict(spearman=None, pearson=None)
            continue
        correlations[name] = dict(spearman=float(spearmanr(x, y).statistic),
                                  pearson=float(pearsonr(x, y).statistic))

    diagnosis = {}
    for name, values in features.items():
        population = np.array([values[layer] for layer in LAYERS])
        mu, sd = population.mean(), population.std()
        if sd == 0:
            continue
        z = lambda layer: float((values[layer] - mu) / sd)
        diagnosis[name] = dict(
            top5={layer: z(layer) for layer in TOP5},
            bottom3={layer: z(layer) for layer in BOTTOM3},
            top5_mean_z=statistics.mean(z(layer) for layer in TOP5),
            bottom3_mean_z=statistics.mean(z(layer) for layer in BOTTOM3))
    separation = sorted(
        ((name, d['top5_mean_z'] - d['bottom3_mean_z']) for name, d in diagnosis.items()),
        key=lambda item: -abs(item[1]))

    ranked = sorted(correlations.items(),
                    key=lambda item: -(abs(item[1]['spearman']) if item[1]['spearman'] is not None else -1))
    lines = ['### Weight-based per-layer attribution (Janus3R-style, weight-only)', '',
        'Complement to the activation-statistics cross-check above: per-head weight-only features '
        f'computed from the checkpoint for all 50 layers (`scripts/weight_attribution.py`; recipe from '
        '`janus_head_predictors.py` / `janus_metrics.py`), aggregated to per-layer mean/max over 56 '
        'heads, correlated with the measured 49-layer dPSNR curve (Spearman and Pearson). '
        f'`static_gaussian_kr95` is the predicted retained ratio for {ALPHA:.0%} softmax mass over '
        f'{VIDEO_TOKENS} routable video tokens at the proxy sigma; `static_projected_v_energy` is '
        'trace(Wo^T Wo Wv Wv^T)/head_dim per head. Activation-dependent Janus metrics (runtime '
        'full-covariance query budgets, key-covariance ranks, measured projected-output energy) are '
        'not weight-only and were not used.', '',
        '| Weight-only feature | Spearman rho | Pearson r |', '|---|---:|---:|']
    for name, corr in ranked:
        if corr['spearman'] is None:
            lines.append(f'| {name} | n/a (constant) | n/a |')
        else:
            lines.append(f"| {name} | {corr['spearman']:+.4f} | {corr['pearson']:+.4f} |")
    best = next(c for c in ranked if c[1]['spearman'] is not None)
    strong = [c for c in ranked if c[1]['spearman'] is not None and abs(c[1]['spearman']) > 0.5]
    lines += ['',
        f"Strongest weight-only correlate: `{best[0]}` (Spearman {best[1]['spearman']:+.4f}). "
        + ('At least one weight feature moderately tracks the layer axis. '
           if strong else
           'No weight-only feature reaches |rho| > 0.5 — like the activation statistics, static '
           'weight geometry does not reliably order layers by end-to-end dense value. '), '',
        'Extreme-layer diagnosis (z-scores vs the 49-layer distribution; top-5 = layers '
        f"{', '.join(map(str, TOP5))}, bottom-3 = {', '.join(map(str, BOTTOM3))}). Largest "
        'top/bottom separations (mean z of top-5 minus mean z of bottom-3):']
    for name, sep in separation[:5]:
        d = diagnosis[name]
        lines.append(f"- `{name}`: separation {sep:+.2f} (top-5 mean z {d['top5_mean_z']:+.2f}, "
                     f"bottom-3 mean z {d['bottom3_mean_z']:+.2f})")
    for name in ('static_projected_v_energy_mean', 'wq_frob_cv', 'wk_frob_cv',
                 'static_kernel_risk_mean', 'static_sigma_mean'):
        d = diagnosis.get(name)
        if d:
            lines.append(f"- `{name}`: bottom-3 z-scores "
                         + ', '.join(f"L{l} {z:+.2f}" for l, z in d['bottom3'].items()))
    lines += ['',
        'z-score separations over 49 layers are descriptive, not causal; with 49 points and many '
        'candidate features some separations are expected by chance.',]
    payload = dict(settings=dict(heads=HEADS, head_dim=HEAD_DIM, video_tokens=VIDEO_TOKENS,
                                 alpha=ALPHA, layers=list(PROBE_LAYERS), gains_layers=list(LAYERS),
                                 model=str(MODEL), janus_source=str(JANUS),
                                 janus_sha256=hashlib.sha256(JANUS.read_bytes()).hexdigest(),
                                 top5=list(TOP5), bottom3=list(BOTTOM3)),
                   layer_gains=gains,
                   per_head_features={layer: heads_by_layer[layer] for layer in PROBE_LAYERS},
                   features=features, correlations=correlations, diagnosis=diagnosis,
                   separation_ranking=separation, report_lines=lines)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({name: corr for name, corr in ranked[:8]}, indent=2))


if __name__ == '__main__':
    sys.exit(main())
