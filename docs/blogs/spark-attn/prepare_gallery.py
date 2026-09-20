"""Rebuild the Visual Comparisons gallery from preserved FFV1 archives; no inference is run.

Dense side: the Sept-13 dense manifest (25 formerly-remapped cases regenerated
on 2026-09-20). Ours side: the 2026-09-20 topk10_reblock_global_reweight
50-prompt run (Spark-H3-10pct), whose FFV1 archives and warmup-excluded
per-case timings supersede the 2026-09-17 gallery run. Previews are H.264
CRF 18 + AAC 192k for the browser; FFV1 archives remain the source of truth.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / '.preview' / 'gallery'
DENSE = Path('/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json')
OURS_EXP = Path('/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920')
OURS_VIDEOS = Path('/autodl-fs/data/h3_outputs/topk_reblock_reweight_50prompt_20260920/videos/topk10_reblock_global_reweight')
CASES = [28, 23, 20, 40, 48, 44, 46, 32]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(item):
    started = time.monotonic()
    source = Path(item['archive_path'])
    assert digest(source) == item['archive_sha256'], source
    target = OUTPUT / item['file']
    poster = OUTPUT / item['poster']
    if target.exists() and poster.exists() and item.get('preview_sha256') == digest(target):
        print('Reused verified preview', item['file'], flush=True)
        return
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '4', '-i', str(source),
               '-map', '0:v:0', '-map', '0:a:0', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
               '-vf', 'setpts=N/(24*TB)', '-r', '24', '-vsync', '0',
               '-pix_fmt', 'yuv420p', '-threads', '4', '-c:a', 'aac', '-b:a', '192k',
               '-movflags', '+faststart', str(target)]
    subprocess.run(command, check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-of', 'json', str(target)]))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    assert (int(video['nb_read_frames']), video['width'], video['height'], video['r_frame_rate']) == (240, 1344, 768, '24/1')
    assert any(s['codec_type'] == 'audio' for s in probe['streams'])
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '2', '-i', str(target),
                    '-frames:v', '1', '-q:v', '3', '-threads', '1', str(OUTPUT / item['poster'])], check=True)
    item.update(preview_sha256=digest(target), bytes=target.stat().st_size, preparation_seconds=time.monotonic()-started,
                verified_frames=240, verified_dimensions=[1344, 768], verified_fps=24, verified_audio=True)
    print('Verified', item['file'], flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dense = {r['index']: r for r in json.loads(DENSE.read_text())['records']}
    ours_records = {c: json.loads((OURS_EXP / 'records' / f'topk10_reblock_global_reweight_{c:02}.json').read_text())
                    for c in CASES}
    old_manifest = json.loads((HERE / 'gallery.json').read_text())
    old_cases = {c['sample_id']: c for c in old_manifest['cases']}
    backup = OUTPUT / ('gallery_manifest_before_spark_h3_10pct_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
    backup.write_text(json.dumps(old_manifest, indent=2, ensure_ascii=False) + '\n')
    cases = []
    for case_id in CASES:
        d = dense[case_id]
        o = ours_records[case_id]
        sid = d['sample_id']
        ours_archive = OURS_VIDEOS / f'{case_id:02}.mkv'
        ours_quality = json.loads((OURS_EXP / 'quality_work/quality' / f'topk10_reblock_global_reweight_{case_id:02}.json').read_text())
        assert ours_quality['video_sha256'] == digest(ours_archive)
        assert ours_quality['reference_sha256'] == d['sha256']
        assert o.get('prompt_sha256', d['prompt_sha256']) == d['prompt_sha256']
        ours_seconds = o['denoise_seconds']
        case = {'sample_id': sid, 'case': case_id, 'prompt': d['prompt'],
                'prompt_sha256': d['prompt_sha256'], 'speedup': d['denoise_seconds'] / ours_seconds}
        for method, path, sha, seconds, tag in [
                ('dense', d['output_path'], d['sha256'], d['denoise_seconds'], 'dense'),
                ('ours', str(ours_archive), digest(ours_archive), ours_seconds, 'spark_h3_10pct')]:
            case[method] = {'archive_path': path, 'archive_sha256': sha, 'denoise_seconds': seconds,
                            'file': f'{sid}_{tag}.mp4', 'poster': f'{sid}_{tag}.jpg'}
        old_dense = old_cases.get(sid, {}).get('dense', {})
        if old_dense.get('archive_sha256') == case['dense']['archive_sha256']:
            case['dense'].update(old_dense)
        cases.append(case)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(encode, [c[m] for c in cases for m in ['dense', 'ours']]))
    manifest = {'output_directory': '.preview/gallery', 'ours_model': 'Spark-H3-10pct (topk10+reblock+global_reweight, 2026-09-20 50-prompt run)',
                'preview_encoding': {'video_workers': 4, 'encoder_threads': 4, 'decoder_threads': 4,
                    'codec': 'H.264', 'crf': 18, 'audio': 'AAC 192k', 'purpose': 'browser preview; FFV1 archives preserved'},
                'sources': [{'path': str(DENSE), 'sha256': digest(DENSE)},
                            {'path': str(OURS_EXP / 'results.json'), 'sha256': digest(OURS_EXP / 'results.json')},
                            {'path': str(OURS_EXP / 'timing_35case.json'), 'sha256': digest(OURS_EXP / 'timing_35case.json')}],
                'cases': cases}
    (HERE / 'gallery.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('GALLERY REBUILT', len(cases), 'cases', flush=True)


if __name__ == '__main__':
    main()
