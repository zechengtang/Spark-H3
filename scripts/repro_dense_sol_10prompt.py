"""Reproduce Sept-13 dense and sol (stock Sol-H3) generations on prompts 1-10 with warmup-excluded timing.

Dense runs without any attention plugin; sol installs the Sept-13 isolated_run
h3_sparse_attention with H3SparseAttentionConfig.sol(20), which is field-for-field
equal to the attention_config stored in the Sept-13 denoise_records. Timing
protocol matches the recent 50-prompt and tau1_power16 runs: synchronized
denoising only, one excluded full-generation warmup per worker, no route audit.
GPUs 0-3 run dense, GPUs 4-7 run sol; case shards are [1,5,9],[2,6,10],[3,7],[4,8].
"""
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SEPT13 = Path('/autodl-fs/data/h3_experiments/vbench20pct_768p10s_seed42_20260913')
ISO = SEPT13 / 'isolated_run'
HIST_OUT = Path('/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913')
ROOT = Path('/autodl-fs/data/h3_experiments/dense_sol_repro_10prompt_20260920')
OUT = Path('/autodl-fs/data/h3_outputs') / ROOT.name
SAMPLES = ISO / 'vbench_core5_percent_subsets/20pct/samples.json'
CASES = list(range(1, 11))
SHARDS = [[1, 5, 9], [2, 6, 10], [3, 7], [4, 8]]
ENV = dict(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='4',
           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', FLASHINFER_CUDA_ARCH_LIST='12.0')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp-{os.getpid()}.json')
    tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def prepare():
    assert not ROOT.exists(), ROOT
    ROOT.mkdir(parents=True)
    OUT.mkdir()
    for name in ('conditioning_cache', 'conditioning_manifest.json'):
        (OUT / name).symlink_to(HIST_OUT / name, target_is_directory=name == 'conditioning_cache')
    hist = {}
    for method in ('dense', 'sol'):
        recs = {read(f)['index']: read(f) for f in (HIST_OUT / method / 'denoise_records').glob('*.json')}
        for c in CASES:
            rec = recs[c]
            assert sha(rec['latent_path']) == rec['latent_sha256']
            hist[f'{method}_{c:02}'] = dict(latent_path=rec['latent_path'],
                latent_sha256=rec['latent_sha256'], historical_denoise_seconds=rec['denoise_seconds'])
    write(ROOT / 'historical_latents.json', hist)
    shutil.copy2(__file__, ROOT / 'runner_source.py')
    write(ROOT / 'protocol.json', dict(cases=CASES, seed=42, steps=20, frames=240, fps=24,
        height=768, width=1344, gpus=dict(dense=[0, 1, 2, 3], sol=[4, 5, 6, 7]), shards=SHARDS,
        sol_code=str(ISO / 'h3_sparse_attention'),
        sol_config='Sept-13 H3SparseAttentionConfig.sol(20), field-for-field equal to the Sept-13 denoise_records attention_config; stock sol: tau=1 diag threshold routing, no reblock, no top-k, no temporal grouping knob in the old schema.',
        timing='Synchronized denoising only; one excluded full-generation warmup per worker (its first shard prompt); loading, conditioning, saving and decoding excluded. torch.compile on (default), compile_deterministic on (default), matching the Sept-13 run.',
        consistency='New latents compared against Sept-13 latent_sha256 (exact sha, then torch.equal, then max-abs diff). Sept-13 cases 2,4,6,8,10 were remapped from a Sept-1 run without an exact code snapshot, so bitwise mismatch there is explainable.',
        environment=ENV))
    print('PREPARE OK', flush=True)


def worker(gpu):
    import torch
    torch.set_num_threads(4)
    method = 'dense' if gpu < 4 else 'sol'
    slot = gpu % 4
    my_cases = SHARDS[slot]
    spec = importlib.util.spec_from_file_location('pipeline', ISO / 'scripts/minimax_h3_vbench_4gpu_pipeline.py')
    p = importlib.util.module_from_spec(spec)
    sys.modules['pipeline'] = p
    spec.loader.exec_module(p)  # self-inserts ISO and ISO/exps into sys.path
    cfg = None
    if method == 'sol':
        import h3_sparse_attention as h3old
        assert str(ISO) in h3old.__file__, h3old.__file__
        cfg = h3old.H3SparseAttentionConfig.sol(20)
    args = p.build_parser().parse_args(['denoise', '--samples', str(SAMPLES),
        '--output', str(OUT), '--method', method, '--case-indices', *map(str, my_cases),
        '--frames', '240', '--height', '768', '--width', '1344', '--steps', '20', '--workers', '1'])
    cases = p.load_cases(SAMPLES, my_cases)
    write(ROOT / f'status_gpu{gpu}.json', dict(stage='loading', method=method))
    workflow, states = p.configure_denoise_workflow(args, cases)
    pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    try:
        for i in range(-1, len(my_cases)):
            case, state = cases[max(i, 0)], states[max(i, 0)]
            target = ROOT / 'records' / f'{method}_{case["index"]:02}.json'
            if i >= 0 and target.exists():
                continue
            prepared = p.clone_state(state)
            prepared.values['prompt_embeds'] = prepared.values['prompt_embeds'].cuda()
            write(ROOT / f'status_gpu{gpu}.json', dict(stage='warmup' if i < 0 else 'measuring',
                method=method, case=case['index']))
            if method == 'sol':
                import h3_sparse_attention as h3old
                ctx = h3old.install_h3_sparse_attention(pipe.transformer, cfg)
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    result = pipe(state=prepared, num_frames=240, height=768, width=1344,
                        num_inference_steps=20, generator=torch.Generator(device='cpu').manual_seed(42),
                        output=['latents', 'audio_latents'])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
            if i < 0:
                write(ROOT / f'warmup_{method}_gpu{gpu}.json', dict(seconds=seconds, case=case['index']))
            else:
                payload = {k: result[k].detach().cpu().contiguous() for k in ('latents', 'audio_latents')}
                assert all(torch.isfinite(v).all() for v in payload.values())
                path = OUT / 'latents' / f'{method}_{case["index"]:02}.pt'
                path.parent.mkdir(exist_ok=True)
                p.atomic_torch_save(payload, path)
                write(target, dict(method=method, case=case['index'], sample_id=case['sample_id'],
                    gpu=gpu, denoise_seconds=seconds, latent_path=str(path), latent_sha256=sha(path),
                    seed=42, steps=20, requested_frames=240, height=768, width=1344))
                del payload
            print(method, case['index'], 'warmup' if i < 0 else 'measured', seconds, flush=True)
            del result, prepared
        write(ROOT / f'status_gpu{gpu}.json', dict(stage='complete', method=method))
    finally:
        acceleration.remove()


def compare():
    import torch
    hist = read(ROOT / 'historical_latents.json')
    rows = []
    for method in ('dense', 'sol'):
        for c in CASES:
            new = read(ROOT / 'records' / f'{method}_{c:02}.json')
            old = hist[f'{method}_{c:02}']
            row = dict(method=method, case=c, denoise_seconds=new['denoise_seconds'],
                historical_denoise_seconds=old['historical_denoise_seconds'],
                historical_latent_sha256=old['latent_sha256'], new_latent_sha256=new['latent_sha256'],
                sha_equal=new['latent_sha256'] == old['latent_sha256'])
            if not row['sha_equal']:
                a = torch.load(new['latent_path'], map_location='cpu', weights_only=False)
                b = torch.load(old['latent_path'], map_location='cpu', weights_only=False)
                for k in ('latents', 'audio_latents'):
                    row[f'{k}_equal'] = bool(torch.equal(a[k], b[k]))
                    row[f'{k}_max_abs_diff'] = float((a[k] - b[k]).abs().max())
                row['latents_cosine'] = float(torch.nn.functional.cosine_similarity(
                    a['latents'].flatten().double(), b['latents'].flatten().double(), dim=0))
            rows.append(row)
    write(ROOT / 'consistency.json', dict(rows=rows))
    return rows


def aggregate():
    rows = read(ROOT / 'consistency.json')['rows']
    summary = {}
    for method in ('dense', 'sol'):
        mr = [r for r in rows if r['method'] == method]
        summary[method] = dict(
            mean_seconds=statistics.mean(r['denoise_seconds'] for r in mr),
            historical_mean_seconds=statistics.mean(r['historical_denoise_seconds'] for r in mr),
            sha_equal=sum(r['sha_equal'] for r in mr),
            tensor_equal=sum(r.get('latents_equal', r['sha_equal']) for r in mr),
            max_abs_diff=max((r.get('latents_max_abs_diff', 0.0) for r in mr), default=0.0),
            per_prompt_seconds={str(r['case']): r['denoise_seconds'] for r in mr})
    write(ROOT / 'results.json', dict(status='complete', summary=summary, rows=rows))
    lines = ['# Dense and stock Sol-H3 reproduction on prompts 1-10 with warmup-excluded timing', '',
        'Cases 1-10; seed 42; 1344x768; 240 frames at 24 fps; 20 requested steps (19 evaluations). Dense runs without any attention plugin; sol uses the Sept-13 isolated_run h3_sparse_attention with H3SparseAttentionConfig.sol(20) (stock Sol-H3, field-for-field equal to the Sept-13 records). One excluded full-generation warmup per worker. Synchronized denoising only. torch.compile and compile_deterministic on, matching Sept-13.', '',
        '| Arm | Mean seconds (this run) | Sept-13 record mean (cases 1-10) | sha256 equal | tensor equal | max abs diff |',
        '|---|---:|---:|---:|---:|---:|']
    for method, s in summary.items():
        lines.append(f"| {method} | {s['mean_seconds']:.3f} | {s['historical_mean_seconds']:.3f} | {s['sha_equal']}/10 | {s['tensor_equal']}/10 | {s['max_abs_diff']:.3e} |")
    lines += ['', 'GPUs 0-3 ran dense and GPUs 4-7 ran sol with case shards [1,5,9],[2,6,10],[3,7],[4,8]. Sept-13 cases 2,4,6,8,10 were remapped from a Sept-1 run whose exact worktree was never snapshotted, so bitwise mismatches on those cases are explainable. Sept-13 record times include first-use compilation on cold caches and were collected on a 3-GPU stride schedule.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    report = REPO / 'reports' / ROOT.name
    report.mkdir(parents=True, exist_ok=True)
    for name in ('results.json', 'REPORT.md', 'protocol.json', 'consistency.json'):
        shutil.copy2(ROOT / name, report / name)
    print('AGGREGATE DONE', json.dumps(summary, indent=1), flush=True)


def run():
    jobs = []
    for gpu in range(8):
        log = (ROOT / f'gpu{gpu}.log').open('a')
        proc = subprocess.Popen([sys.executable, __file__, 'worker', str(gpu)],
            env={**os.environ, **ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    write(ROOT / 'pids.json', [p.pid for p, _ in jobs])
    codes = [p.wait() for p, _ in jobs]
    for _, log in jobs:
        log.close()
    write(ROOT / 'exit_codes.json', codes)
    assert not any(codes), codes
    compare()
    aggregate()


if __name__ == '__main__':
    os.environ.update(ENV)
    if sys.argv[1] == 'worker':
        worker(int(sys.argv[2]))
    else:
        globals()[sys.argv[1]]()
