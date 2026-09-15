"""A narrow, read-only blog preview with locally hosted media and math assets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import threading
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

import markdown

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNTIME = HERE / '.preview'
ASSETS = RUNTIME / 'assets'
GIF = 'https://yang-song.net/assets/img/score/denoise_vp.gif'
KATEX_VERSION = '0.16.22'
CDN = f'https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/'
MATH = re.compile(r'\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)')
LOCK = threading.Lock()
CACHE = {}
ANIMATIONS = frozenset({'spark-reblock.html', 'spark-reweight.html'})
RESIZE_ANIMATIONS = '''<script>
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.data?.type !== 'spark-animation-height') return;
  const height = Number(event.data.height);
  if (!Number.isFinite(height) || height < 250 || height > 2000) return;
  for (const frame of document.querySelectorAll('iframe.spark-animation')) {
    if (frame.contentWindow === event.source) frame.style.height = Math.ceil(height) + 'px';
  }
});
</script>'''

STYLE = '''
:root{color-scheme:light;--ink:#1c2837;--muted:#667085;--line:#e1e7ed;--blue:#235cb5}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f5f7fa;
font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:18px max(24px,calc((100vw - 1080px)/2));
font-size:14px;color:var(--muted)}header a{margin-right:20px}a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}main{max-width:1080px;margin:32px auto 70px;background:white;
padding:48px 64px;border:1px solid var(--line);border-radius:14px;line-height:1.9;font-size:16px}
h1{font-size:clamp(28px,4vw,40px);line-height:1.4;letter-spacing:-.5px;margin-top:0}
h2{font-size:26px;margin-top:54px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-size:20px;margin-top:32px}p{margin:18px 0}li{margin:8px 0}img{display:block;width:100%;
height:auto;margin:24px auto 12px;border-radius:8px;background:#fff}em{color:var(--muted);font-size:14px}
pre{padding:20px;background:#f4f6f9;overflow:auto;border-radius:8px;line-height:1.6}
code{font-size:.88em;background:#f2f5f8;padding:2px 4px;border-radius:3px}pre code{padding:0;background:none}
table{display:block;overflow-x:auto;border-collapse:collapse;font-size:14px;margin:24px 0}
th,td{padding:10px 14px;border:1px solid var(--line);text-align:left}th{background:#f4f7fb}
blockquote{border-left:4px solid #417ac2;margin:24px 0;padding:6px 22px;background:#f2f7ff}
.katex-display{overflow-x:auto;overflow-y:hidden;padding:10px 0}.toc{border:1px solid var(--line);
padding:18px 24px;border-radius:8px;background:#fafbfd;font-size:14px}.toc ul{margin:4px 0}
.spark-animation{display:block;width:100%;height:880px;border:1px solid var(--line);border-radius:14px;margin:24px 0 12px}
@media(max-width:760px){main{margin:0;padding:28px 20px;border:none;border-radius:0}header{padding:14px 20px}
h2{font-size:23px}h3{font-size:19px}.katex{font-size:1em}}
'''


def download(url, relative):
    target = ASSETS / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urlopen(url, timeout=45) as response:
            payload = response.read()
        if relative.endswith('.gif'):
            assert payload[:6] in (b'GIF87a', b'GIF89a')
        temporary = target.with_suffix(target.suffix + '.download')
        temporary.write_bytes(payload)
        temporary.replace(target)
    return {'url': url, 'file': relative, 'bytes': target.stat().st_size,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


def prepare():
    records = [download(GIF, 'denoise_vp.gif'),
               download(CDN + 'dist/katex.min.js', 'katex/katex.min.js'),
               download(CDN + 'dist/katex.min.css', 'katex/katex.min.css'),
               download(CDN + 'LICENSE', 'katex/LICENSE')]
    css = (ASSETS / 'katex/katex.min.css').read_text()
    fonts = sorted(set(re.findall(r'url\([\"\']?(fonts/[^)\"\']+)', css)))
    assert fonts
    with ThreadPoolExecutor(max_workers=8) as pool:
        records += list(pool.map(lambda name: download(CDN + 'dist/' + name, 'katex/' + name), fonts))
    (RUNTIME / 'asset_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'assets': len(records), 'directory': str(ASSETS)}, ensure_ascii=False), flush=True)


def reference_paths():
    result = {}
    for target in re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text()):
        if target.startswith(('https://', 'http://', '#')):
            continue
        path = (HERE / target.split('#', 1)[0]).resolve()
        if path.is_file() and path.is_relative_to(REPO):
            result[target] = path
    return result


def render():
    with LOCK:
        source = (HERE / 'README.md').read_text()
        fingerprint = hashlib.sha256(source.encode()).hexdigest()
        if CACHE.get('fingerprint') == fingerprint:
            return CACHE['result']
        fragments = []
        def hold(match):
            index = len(fragments)
            fragments.append({'text': next(x for x in match.groups() if x is not None),
                              'display': match.group(3) is None})
            return f'H3MATHTOKEN{index}END'
        protected = MATH.sub(hold, source)
        protected = protected.replace('](' + GIF + ')', '](/assets/denoise_vp.gif)')
        protected = protected.replace('src="animations/', 'src="/animations/')
        for target, path in reference_paths().items():
            route = ('/animations/' + path.name if path.parent == HERE / 'animations' and path.name in ANIMATIONS
                     else '/references/' + str(path.relative_to(REPO)))
            protected = protected.replace('](' + target + ')', '](' + route + ')')
        converter = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'], extension_configs={'toc': {'toc_depth': '2-2'}})
        content = converter.convert(protected)
        process = subprocess.run(['node', str(HERE / 'render_math.js'), str(ASSETS / 'katex/katex.min.js')],
                                 input=json.dumps(fragments), text=True, capture_output=True, check=True)
        maths = json.loads(process.stdout)
        assert len(maths) == len(fragments)
        for index, rendered in enumerate(maths):
            marker = f'H3MATHTOKEN{index}END'
            content = content.replace('<p>' + marker + '</p>', rendered).replace(marker, rendered)
        assert 'H3MATHTOKEN' not in content
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Spark-Attn · Spark-MiniMax-H3</title>'
                '<link rel="stylesheet" href="/assets/katex/katex.min.css">'
                '<style>' + STYLE + '</style></head><body><header>'
                '<a href="/">Spark-MiniMax-H3 · Blog</a><a href="/source.md">Markdown source</a>'
                '<a href="/assets/denoise_vp.gif">Original animation</a></header><main>'
                '<details class="toc"><summary>Contents</summary>' + converter.toc + '</details>'
                + content + '</main>' + RESIZE_ANIMATIONS + '</body></html>')
        result = (page.encode(), len(fragments))
        CACHE.update(fingerprint=fingerprint, result=result)
        return result


def resolve_file(route):
    if route in {'/animations/' + name for name in ANIMATIONS}:
        return HERE / 'animations' / route.rsplit('/', 1)[-1]
    if route == '/source.md':
        return HERE / 'README.md'
    if route.startswith('/assets/'):
        name = route.removeprefix('/assets/')
        manifest = json.loads((RUNTIME / 'asset_manifest.json').read_text())
        if name in {record['file'] for record in manifest}:
            return ASSETS / name
    references = {'/references/' + str(path.relative_to(REPO)): path for path in reference_paths().values()}
    return references.get(route)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(False)

    def do_HEAD(self):
        self.respond(True)

    def respond(self, head):
        route = unquote(urlsplit(self.path).path)
        if route == '/':
            data, _ = render()
            content_type = 'text/html; charset=utf-8'
        else:
            path = resolve_file(route)
            if path is None or not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            content_type = (mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            if route.startswith('/references/') or route == '/source.md':
                content_type = 'text/plain; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if not head:
            self.wfile.write(data)


def check():
    page, count = render()
    text = page.decode()
    assert count > 0 and 'class="katex"' in text
    assert 'src="/assets/denoise_vp.gif"' in text and 'Yang Song' in text
    assert GIF not in text  # Both image and direct-image link are local; article attribution remains external.
    for target in ('/.git/config', '/etc/passwd', '/assets/../../etc/passwd', '/references/.git/config'):
        assert resolve_file(target) is None
    for record in json.loads((RUNTIME / 'asset_manifest.json').read_text()):
        payload = (ASSETS / record['file']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
    print(json.dumps({'status': 'passed', 'math_fragments': count, 'html_bytes': len(page),
                      'image': '/assets/denoise_vp.gif', 'unrestricted_file_access': False}))


def main():
    global RUNTIME, ASSETS
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'check', 'serve'])
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()
    RUNTIME = args.runtime.expanduser().resolve()
    ASSETS = RUNTIME / 'assets'
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'check':
        check()
    else:
        render()  # Fail before binding if math/assets are invalid.
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        (RUNTIME / 'server.pid').write_text(str(os.getpid()) + '\n')
        print(f'Listening on http://{args.host}:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
