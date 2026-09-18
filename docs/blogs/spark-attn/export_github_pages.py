"""Export the live article as a self-contained GitHub Pages artifact outside Git."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import preview

ATTR = re.compile(r'((?:href|src|poster|data-src)=")(/[^"\s]*)(")')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {'href', 'src', 'poster', 'data-src'} and value:
                if not value.startswith(('https:', 'http:', 'data:', '#', 'mailto:', '//')):
                    self.links.append(value.split('#')[0].split('?')[0])


def build(output):
    output = output.resolve()
    if output.is_relative_to(preview.REPO):
        raise ValueError('Generated media must stay outside the Git repository.')
    if output.exists():
        raise FileExistsError('Use a fresh output directory; previous exports are preserved.')
    page, math_count = preview.render()
    page = page.decode()
    output.mkdir(parents=True)
    shutil.copytree(preview.ASSETS, output / 'assets')
    routes = sorted(set(match[1] for match in ATTR.findall(page)))
    for route in routes:
        if route == '/':
            continue
        source = preview.resolve_file(route)
        if source is None or not source.is_file():
            raise ValueError(f'Unresolved route: {route}')
        target = output / route.lstrip('/')
        if not target.resolve().is_relative_to(output):
            raise ValueError(f'Invalid output route: {route}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    page = ATTR.sub(lambda m: m[1] + (m[2].lstrip('/') or 'index.html') + m[3], page)
    (output / 'index.html').write_text(page, encoding='utf-8')
    (output / '.nojekyll').write_text('')
    for path in output.rglob('*.html'):
        links = LocalLinks()
        links.feed(path.read_text())
        for link in links.links:
            if link.startswith('/') or not (path.parent / link).is_file():
                raise ValueError(f'Broken static link in {path}: {link}')
    records = [{'file': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                'sha256': digest(path)} for path in sorted(output.rglob('*')) if path.is_file()]
    total = sum(record['bytes'] for record in records)
    if total >= 1_000_000_000:
        raise ValueError('Site exceeds the GitHub Pages 1 GB size limit.')
    sources = [preview.HERE / name for name in
               ('README.md', 'blog.css', 'preview.py', 'gallery.json', 'export_github_pages.py')]
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_head': subprocess.check_output(['git', '-C', str(preview.REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                'source_state': 'Current working tree; source hashes identify the exported content.',
                'source_sha256': {str(p.relative_to(preview.REPO)): digest(p) for p in sources},
                'math_fragments': math_count, 'total_bytes': total, 'files': records}
    (output / 'site-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = output.parent / 'spark-h3-site.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    result = {'directory': str(output), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest(archive), 'site_bytes': total, 'files': len(records),
              'video_count': len(list(output.rglob('*.mp4'))), 'created_utc': manifest['created_utc']}
    (output.parent / 'export.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
