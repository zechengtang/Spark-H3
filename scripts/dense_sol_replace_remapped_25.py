"""Regenerate the 25 remapped (unreproducible) dense+sol cases and replace them in place.

Guard: only runs after dense_sol_verify_extend_20260920 confirms the provenance
conclusion (fresh cases reproducible, remapped cases divergent). Then:
  1. moves the 25 remapped cases' latents/records/videos/evidence (both methods)
     into a quarantine directory;
  2. reruns the Sept-13 isolated_run pipeline natively against the real output
     root (dense on GPUs 0-3, sol on GPUs 4-7, --workers 4, pipeline-default
     denoise protocol matching the surviving 25 fresh records);
  3. decodes videos with the pipeline's own decode stage;
  4. runs the pipeline verify stage over all 50 cases per method, regenerating
     the full generation_manifest.json;
  5. deletes the quarantined untrusted data and writes a regeneration manifest.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SEPT13 = Path('/autodl-fs/data/h3_experiments/vbench20pct_768p10s_seed42_20260913')
ISO = SEPT13 / 'isolated_run'
PIPELINE = ISO / 'scripts/minimax_h3_vbench_4gpu_pipeline.py'
HIST_OUT = Path('/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913')
SAMPLES = ISO / 'vbench_core5_percent_subsets/20pct/samples.json'
VERIFY_ROOT = Path('/autodl-fs/data/h3_experiments/dense_sol_verify_extend_20260920')
ROOT = Path('/autodl-fs/data/h3_experiments/dense_sol_replace_remapped_20260920')
QUARANTINE = ROOT / 'quarantine'
REMAPPED = [2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
ARTIFACTS = ('latents', 'denoise_records', 'videos', 'decode_records', 'render_evidence')
ENV = dict(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='4',
           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', FLASHINFER_CUDA_ARCH_LIST='12.0')


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp-{os.getpid()}.json')
    tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def stems():
    samples = json.loads(SAMPLES.read_text())
    entries = samples['samples'] if isinstance(samples, dict) and 'samples' in samples else samples
    by_index = {e['index']: e['sample_id'] for e in entries}
    return {c: f'{c:02d}_{by_index[c]}' for c in REMAPPED}


def guard():
    results = json.loads((VERIFY_ROOT / 'results.json').read_text())
    assert results['status'] == 'complete' and results['verdict'] == 'confirmed', results['verdict']
    print('GUARD OK: verification verdict confirmed', flush=True)


def quarantine():
    moved = []
    for method in ('dense', 'sol'):
        for case, stem in stems().items():
            for art in ARTIFACTS:
                src = HIST_OUT / method / art / f'{stem}.{"pt" if art == "latents" else "mkv" if art == "videos" else "json"}'
                if src.exists():
                    dst = QUARANTINE / method / art / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    moved.append(str(dst))
    write(ROOT / 'quarantine_manifest.json', dict(moved=moved, count=len(moved)))
    print(f'QUARANTINE OK: {len(moved)} files moved', flush=True)


def stage(command):
    jobs = []
    for gpu in range(8):
        method = 'dense' if gpu < 4 else 'sol'
        log = (ROOT / f'{command}_gpu{gpu}.log').open('a')
        cmd = [sys.executable, str(PIPELINE), command, '--samples', str(SAMPLES),
               '--output', str(HIST_OUT), '--method', method, '--case-indices',
               *map(str, REMAPPED), '--workers', '4', '--worker-rank', str(gpu % 4),
               '--frames', '240', '--height', '768', '--width', '1344', '--steps', '20']
        proc = subprocess.Popen(cmd, env={**os.environ, **ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    codes = [p.wait() for p, _ in jobs]
    for _, log in jobs:
        log.close()
    write(ROOT / f'{command}_exit_codes.json', codes)
    assert not any(codes), codes
    print(f'{command.upper()} OK', flush=True)


def verify():
    for method in ('dense', 'sol'):
        log = (ROOT / f'verify_{method}.log').open('a')
        cmd = [sys.executable, str(PIPELINE), 'verify', '--samples', str(SAMPLES),
               '--output', str(HIST_OUT), '--method', method, '--case-indices',
               *map(str, range(1, 51)), '--frames', '240', '--height', '768', '--width', '1344', '--steps', '20']
        code = subprocess.call(cmd, env={**os.environ, **ENV, 'CUDA_VISIBLE_DEVICES': ''},
                               stdout=log, stderr=subprocess.STDOUT)
        log.close()
        assert code == 0, (method, code)
    print('VERIFY OK: full 50-case generation manifests regenerated', flush=True)


def finalize():
    shutil.rmtree(QUARANTINE)
    write(HIST_OUT / 'regeneration_20260920.json', dict(
        status='complete', cases=REMAPPED, methods=['dense', 'sol'],
        reason='These 25 cases were remapped from the unsnapshotted Sept-1 10pct run and are not reproducible by current code (confirmed by dense_sol_repro_10prompt_20260920 and dense_sol_verify_extend_20260920). Regenerated natively with the Sept-13 isolated_run pipeline on 2026-09-20; old files deleted.',
        pipeline=str(PIPELINE), protocol='Sept-13 pipeline defaults (no per-case warmup), identical to the surviving 25 fresh cases; verify stage regenerated the full 50-case generation_manifest.json per method.'))
    shutil.copy2(__file__, ROOT / 'runner_source.py')
    print('FINALIZE OK: quarantine deleted, regeneration manifest written', flush=True)


def run():
    assert not QUARANTINE.exists(), 'quarantine already exists - resolve manually'
    ROOT.mkdir(parents=True, exist_ok=True)
    guard()
    quarantine()
    stage('denoise')
    stage('decode')
    verify()
    finalize()


if __name__ == '__main__':
    os.environ.update(ENV)
    globals()[sys.argv[1]]()
