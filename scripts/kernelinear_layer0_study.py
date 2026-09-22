"""Can Janus3R kernel-linear (FAVOR+ random-feature) attention replace MiniMax-H3
layer-0 softmax attention? Offline study on captured dense-trajectory Q/K/V.

Captures: /autodl-fs/data/h3_experiments/head_sparsity_all50_20260919/case_XX/
eval_EE_layer_LL.pt — q,k post qk-norm post-RoPE, v head-split, [56, 15045, 128]
bf16, first 14430 tokens video (grid 37x15x26), remaining 615 context/tail.
Verified from MiniMax-H3-Experiments/scripts/capture_head_sparsity_all_layers.py
(record(): attn.norm_q/norm_k then _apply_rotary_emb on q/k only).

Reference: exact softmax(q k^T/sqrt(d)) @ v over ALL keys at 256 evenly spaced
video queries per head (linspace(0, 14429, 256), same sampling as the recall
study), fp32, TF32 off.

Approximations (per head, Janus3R RandomFeatureMap + kernelinear_linear_attention,
softmax_scale=True, feature_stabilize=True, eps=1e-6, head-keyed generators via
janus's make_kernelinear_generator):
- gaussian_identity: weight_init='gaussian', variance_scale='identity'
- orthogonal_chi: weight_init='orthogonal', scale_mode='chi'
- gaussian_full_cov_mgf_runtime: variance_scale='full_cov_mgf' with the proposal
  second moment built from the ACTUAL captured (scaled) q/k moments per head:
  second = Cov(q~) + Cov(k~) + (mu_q + mu_k)(mu_q + mu_k)^T, q~ = q*d^-1/4 over
  video tokens, k~ over all tokens (mirrors kernelinear_attention.py's static
  _head_full_cov_mgf_second_moment = q_cov + k_cov + mean_sum mean_sum^T).
kernel_dim in {256, 1024, 4096, 8192}; 3 seeds for every config (base seeds
20260921 + 7919*s, layer/head-keyed by janus derive_kernelinear_seed).

Baselines: global mean(v) broadcast; production top-k ~10% (reblocked, drop and
mean_compressed branches) read from the existing head_sparsity_all50_20260919
per_head.csv (same captures, same 256 queries, same relative-L2 definition).

Metric: relative L2 ||approx - ref||_F / ||ref||_F per head; aggregated
mean/P10/P90 over 56 heads per (case, eval, layer). Output rows also keep
per-head values for the seed-mean.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

CAPTURES = Path('/autodl-fs/data/h3_experiments/head_sparsity_all50_20260919')
TOPK_CSV = Path('/autodl-fs/data/h3_repos/MiniMax-H3/reports/head_sparsity_all50_20260919/per_head.csv')
JANUS_KL = Path('/autodl-fs/data/h3_repos/Janus3R/janus3r/kernelinear.py')
OUT = Path('/autodl-fs/data/h3_repos/MiniMax-H3/reports/kernelinear_layer0_5s_480p_20260921')
CASES = (3, 7, 10, 15)
EVALS = (4, 9, 18)
LAYERS = (0, 1, 3, 22, 45, 49)
HEADS = 56
DIM = 128
N_QUERIES = 256
KERNEL_DIMS = (256, 1024, 4096, 8192)
SEEDS = tuple(20260921 + 7919 * s for s in range(3))
CONFIGS = ('gaussian_identity', 'orthogonal_chi', 'gaussian_full_cov_mgf_runtime')
VIDEO_GRID = (37, 15, 26)


def janus_module():
    spec = importlib.util.spec_from_file_location('janus_kernelinear_reference', JANUS_KL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def topk_baselines():
    """Aggregate the existing per-head 10%-density errors to (case, eval, layer)."""
    groups = {}
    with TOPK_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (int(row['case']), int(row['evaluation']), int(row['layer']),
                   row['layout'], row['branch'])
            groups.setdefault(key, []).append(float(row['error_at_10pct_density']))
    out = {}
    for key, vals in groups.items():
        vals = np.array(vals)
        out[key] = dict(mean=float(vals.mean()), p10=float(np.percentile(vals, 10)),
                        p90=float(np.percentile(vals, 90)))
    return out


def aggregate(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return dict(mean=float(vals.mean()), p10=float(np.percentile(vals, 10)),
                p90=float(np.percentile(vals, 90)))


@torch.inference_mode()
def process_file(case, evaluation, layer, kl, device, kernel_dims=KERNEL_DIMS):
    path = CAPTURES / f'case_{case:02}' / f'eval_{evaluation:02}_layer_{layer:02}.pt'
    cap = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    nv = int(cap['video_tokens'])
    assert tuple(cap['grid']) == VIDEO_GRID and nv == 37 * 15 * 26
    q_ids = torch.linspace(0, nv - 1, N_QUERIES).round().long()
    k = cap['k'].to(device=device, dtype=torch.float32)          # (56, N, 128) all keys
    v = cap['v'].to(device=device, dtype=torch.float32)
    q = cap['q'][:, q_ids].to(device=device, dtype=torch.float32)  # (56, 256, 128)
    n_tokens = k.shape[1]
    # Exact reference, fp32, logits also give the per-layer logit-std descriptor.
    logits = (q @ k.transpose(-1, -2)) * DIM ** -0.5              # (56, 256, N)
    logit_std = logits.std(dim=(1, 2))
    ref = torch.softmax(logits, dim=-1) @ v                        # (56, 256, 128)
    del logits
    ref_norm = ref.norm(dim=(1, 2)).clamp_min(1e-20)               # (56,)
    rows = []
    # Global mean(v) baseline (over all keys, as production keeps context exact
    # but kernel-linear mixes everything; both are recorded).
    mean_out = v.mean(dim=1, keepdim=True).expand_as(ref)
    rows.append(dict(config='mean_v_baseline', kernel_dim=0, seed=0,
                     rel_l2=((mean_out - ref).norm(dim=(1, 2)) / ref_norm).tolist()))
    del mean_out
    # Runtime full_cov_mgf proposal moments per head, on the scaled tensors the
    # RFM actually sees (kernelinear_linear_attention applies d^-1/4 internally).
    scale = DIM ** -0.25
    q_cal = (cap['q'][:, :nv].to(device=device, dtype=torch.float64)) * scale
    k_cal = (cap['k'].to(device=device, dtype=torch.float64)) * scale
    mu_q, mu_k = q_cal.mean(1), k_cal.mean(1)                      # (56, 128)
    qc = q_cal - mu_q[:, None, :]
    kc = k_cal - mu_k[:, None, :]
    cov_q = torch.einsum('hnd,hne->hde', qc, qc) / nv
    cov_k = torch.einsum('hnd,hne->hde', kc, kc) / n_tokens
    mu = mu_q + mu_k
    moments = (cov_q + cov_k + torch.einsum('hd,he->hde', mu, mu)).float()  # (56,128,128)
    del q_cal, k_cal, qc, kc, cov_q, cov_k
    torch.cuda.empty_cache()
    for config in CONFIGS:
        for m in kernel_dims:
            for seed in SEEDS:
                errs = []
                for h in range(HEADS):
                    generator = kl.make_kernelinear_generator(
                        seed, layer_idx=layer, dim=DIM, kernel_dim=m, device=device, head_idx=h)
                    kwargs = dict(eps=1e-6, generator=generator)
                    if config == 'orthogonal_chi':
                        kwargs.update(weight_init='orthogonal', scale_mode='chi')
                    elif config == 'gaussian_full_cov_mgf_runtime':
                        kwargs.update(weight_init='gaussian', variance_scale='full_cov_mgf',
                                      full_cov_mgf_second_moment=moments[h])
                    else:
                        kwargs.update(weight_init='gaussian')
                    rfm = kl.RandomFeatureMap(DIM, m, **kwargs).to(device)
                    out = kl.kernelinear_linear_attention(
                        q[h:h + 1].unsqueeze(0), k[h:h + 1].unsqueeze(0), v[h:h + 1].unsqueeze(0),
                        rfm, use_triton=False, softmax_scale=True, feature_stabilize=True)
                    out = out[0, 0].float()
                    assert torch.isfinite(out).all(), (config, m, seed, h)
                    errs.append(float((out - ref[h]).norm() / ref_norm[h]))
                    del rfm, out
                rows.append(dict(config=config, kernel_dim=m, seed=seed, rel_l2=errs))
    return dict(case=case, evaluation=evaluation, layer=layer, video_tokens=nv,
                logit_std=logit_std.tolist(), rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, nargs='*', default=list(LAYERS))
    parser.add_argument('--kernel-dims', type=int, nargs='*', default=None)
    parser.add_argument('--force', action='store_true',
                        help='recompute files even if already present (rows are replaced)')
    args = parser.parse_args()
    kernel_dims = tuple(args.kernel_dims) if args.kernel_dims else KERNEL_DIMS
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kl = janus_module()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'results.json'
    if target.is_file():
        results = json.loads(target.read_text())
    else:
        results = dict(provenance=dict(
            captures=str(CAPTURES), janus_kernelinear=str(JANUS_KL),
            janus_kernelinear_sha256=hashlib.sha256(JANUS_KL.read_bytes()).hexdigest(),
            topk_csv=str(TOPK_CSV),
            topk_csv_sha256=hashlib.sha256(TOPK_CSV.read_bytes()).hexdigest(),
            queries=f'{N_QUERIES} evenly spaced video queries (linspace(0, 14429, 256))',
            reference='exact softmax over all 15045 keys, fp32, TF32 off',
            seeds=list(SEEDS), kernel_dims=list(KERNEL_DIMS), configs=list(CONFIGS)),
            files=[])
    done = {(f['case'], f['evaluation'], f['layer']) for f in results['files']}
    if args.force:
        done = set()
        results['files'] = [f for f in results['files'] if f['layer'] not in args.layers]
    for case in CASES:
        for evaluation in EVALS:
            for layer in args.layers:
                if (case, evaluation, layer) in done:
                    continue
                started = time.time()
                results['files'].append(process_file(case, evaluation, layer, kl, device, kernel_dims))
                tmp = target.with_suffix('.tmp')
                tmp.write_text(json.dumps(results))
                tmp.replace(target)
                print('DONE', case, evaluation, layer, f'{time.time() - started:.1f}s', flush=True)
    # Aggregate.
    summary = []
    for f in results['files']:
        key = (f['case'], f['evaluation'], f['layer'])
        per_head_mean_v = next(r for r in f['rows'] if r['config'] == 'mean_v_baseline')['rel_l2']
        entry = dict(case=f['case'], evaluation=f['evaluation'], layer=f['layer'],
                     logit_std=aggregate(f['logit_std']), mean_v_baseline=aggregate(per_head_mean_v))
        present_dims = sorted({r['kernel_dim'] for r in f['rows'] if r['config'] != 'mean_v_baseline'})
        for config in CONFIGS:
            for m in present_dims:
                sel = [r['rel_l2'] for r in f['rows'] if r['config'] == config and r['kernel_dim'] == m]
                head_mean = np.mean(np.array(sel), axis=0)  # seed-mean per head
                entry[f'{config}@{m}'] = aggregate(head_mean)
        summary.append(entry)
    results['summary'] = summary
    results['topk_baselines'] = {
        f'{c},{e},{l},{layout},{branch}': v
        for (c, e, l, layout, branch), v in topk_baselines().items() if l in LAYERS}
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(results))
    tmp.replace(target)
    print('wrote', target, flush=True)


if __name__ == '__main__':
    main()
