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
RUNTIME = Path(os.environ.get('SPARK_BLOG_RUNTIME', str(HERE / '.preview'))).expanduser().resolve()
ASSETS = RUNTIME / 'assets'
GIF = 'https://yang-song.net/assets/img/score/denoise_vp.gif'
KATEX_VERSION = '0.16.22'
CDN = f'https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/'
FONTSOURCE_VERSION = '5.3.0'
FONTSOURCE = 'https://cdn.jsdelivr.net/npm/'
FONT_ASSETS = (
    ('@fontsource-variable/assistant', 'assistant-latin-wght-normal.woff2'),
    ('@fontsource-variable/newsreader', 'newsreader-latin-wght-normal.woff2'),
    ('@fontsource-variable/newsreader', 'newsreader-latin-wght-italic.woff2'),
)
MATH = re.compile(r'\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$')
LOCK = threading.Lock()
CACHE = {}
ANIMATIONS = frozenset({'spark-reblock.html', 'spark-reweight.html', 'branch-comparison.html'})
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

MARK_SCRIPT = '''<script>
(() => {
  const button = document.querySelector('.brand-switch');
  const setMark = mark => {
    button.dataset.mark = mark;
    button.setAttribute('aria-pressed', String(mark === 'green'));
    button.setAttribute('aria-label', `Show ${mark === 'green' ? 'orange' : 'green'} Spark wordmark`);
  };
  let saved = 'orange';
  try { saved = localStorage.getItem('spark-h3-wordmark-color') || 'orange'; } catch (_) {}
  setMark(saved === 'green' ? 'green' : 'orange');
  button.addEventListener('click', () => {
    const next = button.dataset.mark === 'orange' ? 'green' : 'orange';
    setMark(next);
    try { localStorage.setItem('spark-h3-wordmark-color', next); } catch (_) {}
  });
})();
</script>'''

STYLE = (HERE / 'blog.css').read_text()


def download(url, relative):
    target = ASSETS / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urlopen(url, timeout=45) as response:
            payload = response.read()
        if relative.endswith('.gif'):
            assert payload[:6] in (b'GIF87a', b'GIF89a')
        if relative.endswith('.woff2'):
            assert payload[:4] == b'wOF2'
        temporary = target.with_suffix(target.suffix + '.download')
        temporary.write_bytes(payload)
        temporary.replace(target)
    payload = target.read_bytes()
    if relative.endswith('.woff2'):
        assert payload[:4] == b'wOF2'
    if relative.endswith('-OFL.txt'):
        assert b'SIL OPEN FONT LICENSE' in payload
    return {'url': url, 'file': relative, 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}


def prepare():
    records = [download(GIF, 'denoise_vp.gif'),
               download(CDN + 'dist/katex.min.js', 'katex/katex.min.js'),
               download(CDN + 'dist/katex.min.css', 'katex/katex.min.css'),
               download(CDN + 'LICENSE', 'katex/LICENSE')]
    for package, filename in FONT_ASSETS:
        base = f'{FONTSOURCE}{package}@{FONTSOURCE_VERSION}/'
        records.append(download(base + 'files/' + filename, 'fonts/' + filename))
    for family in ('assistant', 'newsreader'):
        package = f'@fontsource-variable/{family}'
        records.append(download(
            f'{FONTSOURCE}{package}@{FONTSOURCE_VERSION}/LICENSE',
            f'fonts/{family}-OFL.txt'))
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


def attention_art():
    """Decorative attention-grid motif; not measured experiment data."""
    tiles = []
    for row in range(16):
        for col in range(16):
            selected = row // 4 == col // 4
            color = '#b6ee76' if selected else '#456351'
            opacity = (0.45 + ((row * 7 + col * 3) % 7) / 12 if selected
                       else 0.12 + ((row * 3 + col * 5) % 5) / 30)
            tiles.append(f'<rect x="{38 + col * 22}" y="{38 + row * 22}" '
                         f'width="17" height="17" rx="2" fill="{color}" opacity="{opacity:.2f}"/>')
    return ('<div class="hero-art" aria-hidden="true"><svg viewBox="0 0 430 430" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<path d="M20 105V20h85 M325 20h85v85 M410 325v85h-85 M105 410H20v-85" '
            'fill="none" stroke="#78947b" stroke-width="1"/>'
            + ''.join(tiles) + '</svg></div>')


def article_layout(content):
    """Style the Markdown's existing content without maintaining a second copy."""
    title = re.search(r'<h1[^>]*>(.*?)</h1>\s*<p>(.*?)</p>', content, re.S)
    if title is None:
        raise ValueError('Expected a blog title followed by its byline.')
    heading, meta = title.groups()
    name, separator, subtitle = heading.partition(': ')
    if not separator:
        name, subtitle = heading, ''
    content = content[title.end():]
    sections = list(re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', content, re.S))
    intro = content[:sections[0].start()] if sections else content
    features = re.search(r'<ul>(.*?)</ul>', intro, re.S)
    links = []
    if features:
        items = re.findall(r'<li>(.*?)</li>', features.group(1), re.S)
        for index, (item, section) in enumerate(zip(items, sections), 1):
            links.append(f'<a class="component-link" href="#{section.group(1)}">'
                         f'<span class="component-number">0{index}</span><span>{item}</span>'
                         '<span class="arrow" aria-hidden="true">↗</span></a>')
        intro = intro[:features.start()] + intro[features.end():]
    hero = ('<section class="hero" aria-labelledby="blog-title"><div class="hero-inner">'
            '<div class="hero-stage"><div class="hero-copy">'
            '<p class="eyebrow">MiniMax-H3 · BSA</p>'
            f'<h1 id="blog-title">{name}</h1><p class="hero-subtitle">{subtitle}</p>'
            f'<p class="hero-meta">{meta}</p><div class="hero-actions">'
            '<a class="button primary" href="#overview">Explore the method <span aria-hidden="true">↓</span></a>'
            '<a class="button secondary" href="https://github.com/zechengtang/Spark-H3">View the code <span aria-hidden="true">↗</span></a>'
            '</div></div>' + attention_art() + '</div>'
            '<div class="component-links">' + ''.join(links) + '</div></div></section>')
    overview = ('<section class="overview" id="overview" aria-labelledby="overview-title">'
                '<div class="section-heading"><p class="eyebrow">Overview</p>'
                '<h2 id="overview-title">Adaptive BSA</h2></div>'
                '<div class="overview-copy">' + intro + '</div></section>')
    chapters = []
    chapter_kinds = {
        'spark-reblock': 'METHOD',
        'spark-reweight': 'METHOD',
        'benchmark-results': 'EVALUATION',
        'spark-integration': 'COMMUNITY',
        'spark-ref2va-preview': 'PREVIEW',
        'visual-comparisons': 'SHOWCASE',
        'related-works': 'REFERENCES',
        'update-history': 'CHANGELOG',
    }
    figure_index = 0
    table_index = 0
    def figure(match):
        nonlocal figure_index
        figure_index += 1
        return ('<figure class="article-figure"><div class="figure-image">' + match.group(1)
                + '</div><figcaption><span class="figure-number">'
                + f'{figure_index:02d}</span>' + match.group(2) + '</figcaption></figure>')
    for index, section in enumerate(sections):
        end = sections[index + 1].start() if index + 1 < len(sections) else len(content)
        body = content[section.end():end]
        reading = re.match(r'\s*<p><em>(~\d+ min read)</em></p>', body)
        reading_html = ''
        if reading:
            body = body[reading.end():]
            reading_html = f'<span class="reading-time">{reading.group(1)}</span>'
        body = re.sub(r'<p>(<img\b[^>]*>)</p>\s*<p><em>(.*?)</em></p>', figure, body, flags=re.S)
        label = html.escape(section.group(2) + ' results', quote=True)
        def data_table(match):
            nonlocal table_index
            table_index += 1
            table = match.group(1)
            columns = len(re.findall(r'<th\b', table))
            has_intermediate_results = table_index in (3, 4)
            def highlight_method(row_match):
                row = row_match.group(0)
                first_cell = re.search(r'<td>(.*?)</td>', row, re.S)
                classes = []
                if first_cell and re.search(r'(?:Spark|Reblock|Reweight)', first_cell.group(1)):
                    classes.append('is-highlight')
                cells = re.findall(r'<td(?:\s[^>]*)?>(.*?)</td>', row, re.S)
                cell_text = [html.unescape(re.sub(r'<[^>]+>', '', cell)).strip() for cell in cells]
                if (has_intermediate_results
                        and any(re.search(r'(?:^|,\s)(?:15|20)%', value)
                                for value in cell_text)):
                    classes.append('is-intermediate-result')
                if classes:
                    attributes = f' class="{" ".join(classes)}"'
                    return row.replace('<tr>', f'<tr{attributes}>', 1)
                return row
            table = re.sub(r'<tr>.*?</tr>', highlight_method, table, flags=re.S)
            scroll_hint = ('' if table_index in (3, 4) else
                           '<span class="table-scroll-hint">Scroll to compare <i>→</i></span>')
            toolbar = (
                '<div class="table-toolbar" aria-hidden="true">'
                f'<span><b>Table</b> / {table_index:02d}</span>'
                + scroll_hint + '</div>')
            rendered_table = (
                f'<div class="table-wrap" role="region" tabindex="0" '
                f'aria-label="{label}" data-columns="{columns}" '
                f'data-table-index="{table_index}">'
                + toolbar + table + '</div>')
            if not has_intermediate_results:
                return rendered_table
            return (
                '<div class="table-results-group" data-intermediate-results>'
                + rendered_table
                + '<details class="table-results-details"><summary>'
                '<span class="prompt-heading"><span class="prompt-label">Additional densities</span>'
                '<span class="prompt-toggle"><span class="prompt-expand">Show 15% &amp; 20%</span>'
                '<span class="prompt-collapse">Collapse 15% &amp; 20%</span>'
                '<span class="prompt-chevron" aria-hidden="true">↗</span></span></span>'
                '</summary></details></div>')
        body = re.sub(r'(<table>.*?</table>)', data_table, body, flags=re.S)
        chapters.append(f'<section class="chapter" id="{section.group(1)}" '
                        f'aria-labelledby="{section.group(1)}-title">'
                        '<div class="section-heading">'
                        f'<span class="chapter-number">{index + 1:02d} / '
                        f'{chapter_kinds.get(section.group(1), "ARTICLE")}</span>'
                        f'<h2 id="{section.group(1)}-title"><a href="#{section.group(1)}">'
                        f'{section.group(2)}</a></h2>{reading_html}</div><div class="chapter-body">{body}</div></section>')
    return hero + overview + ''.join(chapters)


def gallery_data():
    data = json.loads((HERE / 'gallery.json').read_text())
    data['output_directory'] = str(RUNTIME / 'gallery')
    return data


def gallery_html():
    cards = []
    for index, case in enumerate(gallery_data()['cases'], 1):
        sid = html.escape(case['sample_id'])
        excerpt = re.sub(r'^integrated_multimodal_description:\s*(?:\[Shot 1\]\s*)?', '', case['prompt'])
        excerpt = ' '.join(excerpt.split())
        videos = []
        for method, label in [('dense', 'Dense'), ('ours', 'Spark-H3')]:
            item = case[method]
            speedup = case['dense']['denoise_seconds'] / item['denoise_seconds']
            primary_metric = (
                '<span class="timing-speedup is-baseline">Baseline</span>'
                if method == 'dense' else
                f'<span class="timing-speedup">{speedup:.2f}×</span>')
            videos.append(
                f'<figure class="comparison-video {method}"><figcaption>'
                f'<strong class="comparison-method"><i aria-hidden="true"></i>{label}</strong>'
                '<span class="comparison-timing">' + primary_metric
                + '<span class="timing-latency"><span class="timing-label">DiT latency</span>'
                f'<span class="timing-value">{item["denoise_seconds"]:.2f}<small>s</small></span>'
                '</span></span></figcaption>'
                f'<video muted playsinline preload="none" disablepictureinpicture disableremoteplayback '
                f'aria-label="{label}: {sid}" poster="/gallery/{item["poster"]}" '
                f'data-src="/gallery/{item["file"]}"></video></figure>')
        cards.append(
            f'<article class="comparison-card" aria-labelledby="case-{sid}">'
            '<div class="comparison-heading"><div class="comparison-identity">'
            f'<h3 class="comparison-index" id="case-{sid}" aria-label="Comparison {index}">{index:02d}</h3>'
            '<button class="video-reset" type="button" title="Restart all videos from the beginning">Restart</button>'
            '</div>'
            '</div>'
            '<div class="comparison-pair">' + ''.join(videos) + '</div>'
            '<details class="comparison-prompt"><summary>'
            '<span class="prompt-heading"><span class="prompt-label">Prompt</span>'
            '<span class="prompt-toggle"><span class="prompt-expand">Show full prompt</span>'
            '<span class="prompt-collapse">Collapse prompt</span><span class="prompt-chevron" aria-hidden="true">↗</span></span></span>'
            '<span class="prompt-excerpt">' + html.escape(excerpt) + '</span></summary>'
            '<div class="prompt-full">' + html.escape(case['prompt']) + '</div></details></article>')
    return '<div class="comparison-gallery">' + ''.join(cards) + '</div>'




def integration_data():
    return {name: json.loads((HERE / 'integration' / f'{name}.json').read_text())
            for name in ('turbo-lora', 'few-step', 'fasth3-0753', 'selflift', 'vdn10', 'ref2va')}


def resolve_integration_file(route):
    for name in ('turbo-lora', 'few-step', 'fasth3-0753', 'selflift', 'vdn10', 'ref2va'):
        if route == f'/{name}/manifest.json':
            return HERE / 'integration' / f'{name}.json'
    record = json.loads((HERE / 'integration/media.json').read_text()).get(route)
    if record is None:
        return None
    path = (RUNTIME / record['file']).resolve()
    return path if path.is_relative_to(RUNTIME.resolve()) else None


def integration_html(data, family, only_model=None):
    manifest = data[family]
    records = {r['variant']: r for r in manifest['variants']}
    groups = []
    for prompt_index, prompt in enumerate(('0753', '0685'), 1):
        if family == 'selflift':
            dense_variant = f'{prompt}_dense'
            models = [
                (dense_variant, 'Dense', '4 low-res + 3 high-res steps · dense', dense_variant),
                (f'{prompt}_spark', 'Spark-H3, 10%',
                 '4 low-res dense + 3 high-res sparse steps', dense_variant),
            ]
            prompt_text = data['fasth3-0753']['prompt' if prompt == '0753' else 'prompt0685']
        elif family in {'turbo-lora', 'few-step'}:
            model_meta = {
                'lightx2v': ('LightX2V', 8),
                'larry': ('Larryvrh', 8),
                'pdd': ('Alibaba-PAI Acc', 8),
                'dmad': ('ByteDance DMAD', 4),
            }
            if family == 'turbo-lora':
                mode_names = ('dense', 'spark10', 'spark10_nowarm')
                default_models = ('lightx2v', 'larry')
                prompt_text = manifest['cases'][prompt]['generation_prompt']
            else:
                mode_names = ('dense', 'spark10_warm', 'spark10_nowarm')
                default_models = ('pdd', 'dmad')
                prompt_text = data['turbo-lora']['cases'][prompt]['generation_prompt']
            model_names = (only_model,) if only_model else default_models
            models = []
            for model in model_names:
                label, steps = model_meta[model]
                details = {
                    'dense': f'{steps} steps · dense',
                    'spark10': f'{steps} steps · 90% sparsity · w/ warmup',
                    'spark10_warm': f'{steps} steps · 90% sparsity · w/ warmup',
                    'spark10_nowarm': f'{steps} steps · 90% sparsity · w/o warmup',
                }
                for mode in mode_names:
                    suffix = '' if mode == 'dense' else ' + Spark-H3, 10%'
                    models.append((f'{prompt}_{model}_{mode}', label + suffix, details[mode],
                                   f'{prompt}_{model}_dense'))
        else:
            prefix = '' if prompt == '0753' else '0685_'
            dense_variant = prefix + 'v1_dense'
            models = [(prefix + name, title, detail, dense_variant) for name, title, detail in (
                ('v1_dense', 'FastH3 v1 Dense', '4 steps · dense'),
                ('v1_dense_spark10_warm', 'FastH3 v1 Dense + Spark-H3, 10%',
                 '4 steps · 90% sparsity · w/ warmup'),
                ('v1_dense_spark10_nowarm', 'FastH3 v1 Dense + Spark-H3, 10%',
                 '4 steps · 90% sparsity · w/o warmup'),
                ('v1_vsa', 'FastH3 v1 VSA', '4 steps · 90% sparsity'),
                ('v2_vsa', 'FastH3 v2 VSA', '8 steps · 80% sparsity'))]
            prompt_text = manifest['prompt' if prompt == '0753' else 'prompt0685']
        cards = []
        for variant, title, detail, dense_variant in models:
            record = records[variant]
            speedup = records[dense_variant]['denoise_seconds'] / record['denoise_seconds']
            speedup_badge = (
                '<span class="speedup-badge is-baseline" title="Baseline">Baseline</span>'
                if abs(speedup - 1.0) < 1e-9 else
                f'<span class="speedup-badge" title="Relative to Dense">{speedup:.2f}×</span>')
            video = ('<video muted playsinline preload="none" disablepictureinpicture disableremoteplayback '
                     f'aria-label="{html.escape(title)}: {prompt_index:02d}" '
                     f'data-src="{html.escape(record["preview"], quote=True)}"></video>')
            cards.append(
                '<figure class="integration-video"><figcaption>'
                f'<strong>{html.escape(title)}</strong><span>{html.escape(detail)}</span>'
                f'{speedup_badge}'
                '</figcaption>' + video + '<div class="integration-video-footer">'
                f'<span>DiT latency <strong>{record["denoise_seconds"]:.1f} s</strong></span>'
                '</div></figure>')
        excerpt = re.sub(r'^integrated_multimodal_description:\s*(?:\[Shot 1\]\s*)?', '', prompt_text)
        excerpt = ' '.join(excerpt.split())
        groups.append(
            '<div class="integration-group">'
            f'<h4>{prompt_index:02d}<button class="video-reset" type="button" '
            'title="Restart all videos from the beginning">Restart</button></h4>'
            '<div class="integration-video-grid">' + ''.join(cards) + '</div>'
            '<details class="comparison-prompt"><summary>'
            '<span class="prompt-heading"><span class="prompt-label">Prompt</span>'
            '<span class="prompt-toggle"><span class="prompt-expand">Show full prompt</span>'
            '<span class="prompt-collapse">Collapse prompt</span><span class="prompt-chevron" aria-hidden="true">↗</span></span></span>'
            '<span class="prompt-excerpt">' + html.escape(excerpt) + '</span></summary>'
            f'<div class="prompt-full">{html.escape(prompt_text)}</div></details></div>')
    return '<div class="integration-gallery">' + ''.join(groups) + '</div>'


def integration_results_html(content, open_by_default=False):
    open_attribute = ' open' if open_by_default else ''
    return (
        f'<details class="integration-results-details"{open_attribute}><summary>'
        '<span class="prompt-heading"><span class="prompt-label">Video comparisons</span>'
        '<span class="prompt-toggle"><span class="prompt-expand">Show results</span>'
        '<span class="prompt-collapse">Hide results</span>'
        '<span class="prompt-chevron" aria-hidden="true">↗</span></span></span>'
        '</summary>'
        + content
        + '</details>')


def collapsible_integration_html(data, family, only_model, open_by_default=False):
    return integration_results_html(
        integration_html(data, family, only_model), open_by_default)


def ref2va_integration_html(manifest):
    case = manifest['case']
    records = {record['variant']: record for record in manifest['variants']}
    dense_seconds = records['dense']['denoise_seconds']
    cards = []
    for record in manifest['variants']:
        title = html.escape(record['title'])
        speedup = dense_seconds / record['denoise_seconds']
        speedup_badge = (
            '<span class="speedup-badge is-baseline" title="Baseline">Baseline</span>'
            if abs(speedup - 1.0) < 1e-9 else
            f'<span class="speedup-badge" title="Relative to Dense">{speedup:.2f}×</span>')
        cards.append(
            '<figure class="integration-video"><figcaption>'
            f'<strong>{title}</strong><span>{html.escape(record["detail"])}</span>'
            f'{speedup_badge}'
            '</figcaption><video muted playsinline preload="none" disablepictureinpicture '
            f'disableremoteplayback aria-label="{title}" '
            f'data-src="{html.escape(record["preview"], quote=True)}"></video>'
            '<div class="integration-video-footer">'
            f'<span>DiT latency <strong>{record["denoise_seconds"]:.1f} s</strong></span>'
            '</div></figure>')
    prompt = case['prompt']
    return (
        '<div class="integration-gallery"><div class="integration-group">'
        f'<h4>01 · {html.escape(case["name"])}<button class="video-reset" type="button" '
        'title="Restart all videos from the beginning">Restart</button></h4>'
        '<div class="integration-video-grid">' + ''.join(cards) + '</div>'
        '<div class="comparison-prompt static-prompt">'
        '<div class="prompt-heading"><span class="prompt-label">Case prompt</span></div>'
        f'<div class="prompt-full">{html.escape(prompt)}</div></div></div></div>')


def vdn10_showcase_html(manifest):
    """Hero carousel of the ten VDN-H3 page prompts, layout after openvdn.github.io."""
    cards = []
    for case in manifest['cases']:
        sid = html.escape(case['sample_id'])
        slug = html.escape(case['slug'].replace('_', ' '))
        excerpt = re.sub(r'^integrated_multimodal_description:\s*(?:\[Shot 1\]\s*)?', '', case['prompt'])
        excerpt = ' '.join(excerpt.split())
        cards.append(
            f'<article class="vdn-result-card" role="listitem" '
            f'aria-label="Showcase video {case["position"]}: {slug}">'
            f'<video class="vdn-result-video" controls loop muted playsinline preload="none" '
            f'disablepictureinpicture disableremoteplayback aria-label="{sid}: {slug}" '
            f'data-src="{html.escape(case["preview"], quote=True)}"></video>'
            '<details class="comparison-prompt"><summary>'
            '<span class="prompt-heading"><span class="prompt-label">Prompt</span>'
            '<span class="prompt-toggle"><span class="prompt-expand">Show full prompt</span>'
            '<span class="prompt-collapse">Collapse prompt</span>'
            '<span class="prompt-chevron" aria-hidden="true">↗</span></span></span>'
            f'<span class="prompt-excerpt">{html.escape(excerpt)}</span></summary>'
            f'<div class="prompt-full">{html.escape(case["prompt"])}</div></details></article>')
    total = len(cards)
    return ('<section class="vdn-showcase" aria-labelledby="vdn-showcase-title">'
            '<div class="vdn-showcase-inner">'
            '<div class="vdn-showcase-head"><div>'
            '<p class="eyebrow">Selected Showcase</p>'
            '<h2 class="vdn-showcase-title" id="vdn-showcase-title">'
            '<span class="model-tag"><span class="model-name">LightX2V</span>'
            '<span class="model-note">8-step</span></span> + '
            '<span class="model-tag"><span class="model-name">Spark-H3</span>'
            '<span class="model-note">90% sparse</span></span></h2>'
            '<p class="vdn-showcase-sub">Prompts come from the '
            '<a href="https://openvdn.github.io/">VDN-H3 project page</a>.</p></div>'
            '<div class="vdn-result-strip-controls" aria-label="Gallery navigation">'
            '<button type="button" data-vdn-strip-prev aria-label="Scroll video results left">←</button>'
            f'<output data-vdn-strip-status aria-live="polite">01 / {total:02d}</output>'
            '<button type="button" data-vdn-strip-next aria-label="Scroll video results right">→</button>'
            '</div></div>'
            '<div class="vdn-result-showcase" data-vdn-result-showcase>'
            '<div class="vdn-result-strip" data-vdn-result-strip role="list" tabindex="0" '
            'aria-label="LightX2V Spark-H3, 10% video results">'
            + ''.join(cards) + '</div></div></div></section>')


VDN10_SCRIPT = """<script>
(() => {
  const showcase = document.querySelector('[data-vdn-result-showcase]');
  if (!showcase) return;
  const section = showcase.closest('.vdn-showcase') || showcase;
  const strip = showcase.querySelector('[data-vdn-result-strip]');
  const prevBtn = section.querySelector('[data-vdn-strip-prev]');
  const nextBtn = section.querySelector('[data-vdn-strip-next]');
  const status = section.querySelector('[data-vdn-strip-status]');
  const cards = Array.from(strip.querySelectorAll('.vdn-result-card'));
  const videos = cards.map(card => card.querySelector('video'));
  if (!cards.length) return;
  let activeIndex = 0;
  let heightFrame = 0;
  let playbackWanted = true;
  const automaticPauses = new WeakSet();

  const prepare = video => {
    if (video.dataset.src) {
      video.src = video.dataset.src;
      delete video.dataset.src;
      video.preload = 'metadata';
      video.load();
    }
  };
  const updateHeight = () => {
    if (heightFrame) cancelAnimationFrame(heightFrame);
    heightFrame = requestAnimationFrame(() => {
      heightFrame = 0;
      const activeCard = cards[activeIndex];
      if (activeCard) strip.style.setProperty('--vdn-carousel-height', `${activeCard.offsetHeight}px`);
    });
  };
  const pauseWithoutChangingIntent = video => {
    if (video.paused) return;
    automaticPauses.add(video);
    video.pause();
  };
  const updatePlayback = () => {
    const rect = showcase.getBoundingClientRect();
    const visible = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
    if (document.hidden || visible < Math.min(120, rect.height * 0.25)) {
      videos.forEach(pauseWithoutChangingIntent);
      return;
    }
    videos.forEach((video, index) => {
      if (index === activeIndex) {
        prepare(video);
        if (playbackWanted) video.play().catch(() => {});
        else pauseWithoutChangingIntent(video);
      } else pauseWithoutChangingIntent(video);
    });
  };
  let playbackFrame = 0;
  const schedulePlayback = () => {
    if (playbackFrame) return;
    playbackFrame = requestAnimationFrame(() => { playbackFrame = 0; updatePlayback(); });
  };
  const updateCarousel = (nextIndex, shouldPlay = true) => {
    activeIndex = (nextIndex + cards.length) % cards.length;
    cards.forEach((card, index) => {
      const forward = (index - activeIndex + cards.length) % cards.length;
      const isCurrent = index === activeIndex;
      card.dataset.carouselPosition = isCurrent ? 'current'
        : forward === cards.length - 1 ? 'previous' : forward === 1 ? 'next' : 'hidden';
      card.style.setProperty('--vdn-carousel-scale', isCurrent ? '1' : '.92');
      card.toggleAttribute('aria-current', isCurrent);
      card.setAttribute('aria-hidden', String(!isCurrent));
      card.inert = !isCurrent;
      if (!isCurrent) {
        pauseWithoutChangingIntent(videos[index]);
        const prompt = card.querySelector('.comparison-prompt');
        if (prompt && prompt.open) prompt.open = false;
      }
    });
    strip.dataset.activeIndex = String(activeIndex);
    if (status) status.textContent = `${String(activeIndex + 1).padStart(2, '0')} / ${String(cards.length).padStart(2, '0')}`;
    updateHeight();
    if (shouldPlay) schedulePlayback();
  };

  prevBtn && prevBtn.addEventListener('click', () => updateCarousel(activeIndex - 1));
  nextBtn && nextBtn.addEventListener('click', () => updateCarousel(activeIndex + 1));
  strip.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    updateCarousel(activeIndex + (event.key === 'ArrowRight' ? 1 : -1));
  });
  let pointerStartX = null;
  strip.addEventListener('pointerdown', event => {
    if (event.pointerType === 'mouse' || event.target.closest('video, button, a, summary, input')) return;
    pointerStartX = event.clientX;
  });
  strip.addEventListener('pointerup', event => {
    if (pointerStartX === null) return;
    const distance = event.clientX - pointerStartX;
    pointerStartX = null;
    if (Math.abs(distance) > 44) updateCarousel(activeIndex + (distance < 0 ? 1 : -1));
  });
  strip.addEventListener('pointercancel', () => { pointerStartX = null; });

  videos.forEach((video, index) => {
    video.addEventListener('play', () => {
      if (index !== activeIndex) { pauseWithoutChangingIntent(video); return; }
      playbackWanted = true;
      videos.forEach((other, otherIndex) => {
        if (otherIndex !== activeIndex) pauseWithoutChangingIntent(other);
      });
    });
    video.addEventListener('pause', () => {
      if (automaticPauses.delete(video)) return;
      if (index === activeIndex && !video.ended) playbackWanted = false;
    });
    video.addEventListener('loadedmetadata', updateHeight);
  });
  strip.querySelectorAll('.comparison-prompt').forEach(details => {
    details.addEventListener('toggle', updateHeight);
  });
  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(updateHeight);
    cards.forEach(card => observer.observe(card));
  }
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(schedulePlayback, { threshold: [0, 0.25, 0.5] }).observe(showcase);
  }
  document.addEventListener('visibilitychange', schedulePlayback);
  updateCarousel(0, false);
  schedulePlayback();
})();
</script>"""


INTEGRATION_SCRIPT = """<script>
(() => {
  const groups = [...document.querySelectorAll('.comparison-card, .integration-group')].map(element => ({
    element, videos: [...element.querySelectorAll('video')], visible: false, token: 0, syncing: false,
  }));
  const states = new WeakMap(groups.map(group => [group.element, group]));

  const waitFor = (video, ready, event) => new Promise((resolve, reject) => {
    if (ready()) { resolve(); return; }
    const cleanup = () => {
      video.removeEventListener(event, onReady);
      video.removeEventListener('error', onError);
    };
    const onReady = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(video.error || new Error('Video failed to load')); };
    video.addEventListener(event, onReady);
    video.addEventListener('error', onError);
  });
  const canPlay = video => waitFor(video, () => video.readyState >= 3, 'canplay');
  const seekTo = (video, time) => {
    if (Math.abs(video.currentTime - time) < 0.02) return Promise.resolve();
    const seeked = waitFor(video, () => Math.abs(video.currentTime - time) < 0.02 && !video.seeking, 'seeked');
    video.currentTime = time;
    return seeked;
  };
  const prepare = group => {
    for (const video of group.videos) {
      if (!video.dataset.src) continue;
      video.preload = 'auto';
      video.src = video.dataset.src;
      delete video.dataset.src;
      video.load();
    }
  };
  const synchronize = async (group, restart = false) => {
    const token = ++group.token;
    group.syncing = true;
    group.videos.forEach(video => video.pause());
    prepare(group);
    try {
      await Promise.all(group.videos.map(canPlay));
      if (token !== group.token || !group.visible) return;
      const time = restart ? 0 : group.videos[0].currentTime;
      await Promise.all(group.videos.map(video => seekTo(video, time)));
      if (token !== group.token || !group.visible) return;
      await Promise.all(group.videos.map(video => video.play()));
    } catch (error) {
      group.videos.forEach(video => video.pause());
      console.warn('Could not synchronize comparison videos', error);
    } finally {
      if (token === group.token) group.syncing = false;
    }
  };

  for (const group of groups) {
    for (const video of group.videos) {
      video.addEventListener('waiting', () => {
        if (group.visible && !group.syncing) synchronize(group);
      });
      video.addEventListener('ended', () => {
        if (group.visible) synchronize(group, true);
      });
    }
  }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const group = states.get(entry.target);
      if (!group) continue;
      if (group.visible === entry.isIntersecting) continue;
      group.visible = entry.isIntersecting;
      if (group.visible) synchronize(group, !group.videos[0].hasAttribute('src'));
      else {
        group.token++;
        group.videos.forEach(video => video.pause());
      }
    }
  }, {rootMargin: '200px'});
  groups.forEach(group => observer.observe(group.element));
  document.addEventListener('visibilitychange', () => {
    for (const group of groups) {
      if (!group.visible) continue;
      if (document.hidden) {
        group.token++;
        group.videos.forEach(video => video.pause());
      } else synchronize(group);
    }
  });
  setInterval(() => {
    if (document.hidden) return;
    for (const group of groups) {
      if (!group.visible || group.syncing || group.videos.some(video => video.paused)) continue;
      const time = group.videos[0].currentTime;
      for (const video of group.videos.slice(1)) {
        if (video.readyState >= 2 && Math.abs(video.currentTime - time) > 0.12)
          video.currentTime = time;
      }
    }
  }, 500);
  document.addEventListener('click', event => {
    const button = event.target.closest('.video-reset');
    if (!button) return;
    const group = states.get(button.closest('.comparison-card, .integration-group'));
    if (group) { group.visible = true; synchronize(group, true); }
  });
})();
</script>"""

def render():
    with LOCK:
        source = (HERE / 'README.md').read_text()
        style = (HERE / 'blog.css').read_text()
        integrations = integration_data()
        fingerprint = hashlib.sha256((source + style + (HERE / 'gallery.json').read_text() + json.dumps(integrations, sort_keys=True)).encode()).hexdigest()
        if CACHE.get('fingerprint') == fingerprint:
            return CACHE['result']
        fragments = []
        def hold(match):
            index = len(fragments)
            fragments.append({'text': next(x for x in match.groups() if x is not None),
                              'display': match.group(1) is not None or match.group(2) is not None})
            return f'H3MATHTOKEN{index}END'
        protected = MATH.sub(hold, source)
        protected = protected.replace('](' + GIF + ')', '](/assets/denoise_vp.gif)')
        for target, path in reference_paths().items():
            route = ('/animations/' + path.name if path.parent == HERE / 'animations' and path.name in ANIMATIONS
                     else '/references/' + str(path.relative_to(REPO)))
            protected = protected.replace('](' + target + ')', '](' + route + ')')
        converter = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'], extension_configs={'toc': {'toc_depth': '2-2'}})
        content = converter.convert(protected).replace('<!-- VIDEO_GALLERY -->', gallery_html())
        content = content.replace('<!-- TURBO_INTEGRATION -->', integration_html(integrations, 'turbo-lora'))
        content = content.replace('<!-- LIGHTX2V_INTEGRATION -->',
                                  collapsible_integration_html(
                                      integrations, 'turbo-lora', 'lightx2v', True))
        content = content.replace('<!-- LARRY_INTEGRATION -->',
                                  collapsible_integration_html(
                                      integrations, 'turbo-lora', 'larry', True))
        content = content.replace('<!-- PDD_INTEGRATION -->',
                                  collapsible_integration_html(integrations, 'few-step', 'pdd'))
        content = content.replace('<!-- DMAD_INTEGRATION -->',
                                  collapsible_integration_html(integrations, 'few-step', 'dmad'))
        content = content.replace('<!-- FASTH3_INTEGRATION -->',
                                  collapsible_integration_html(
                                      integrations, 'fasth3-0753', None, True))
        content = content.replace('<!-- SELFLIFT_INTEGRATION -->',
                                  collapsible_integration_html(
                                      integrations, 'selflift', None, True))
        content = content.replace('<!-- REF2VA_INTEGRATION -->',
                                  integration_results_html(
                                      ref2va_integration_html(integrations['ref2va']), True))
        content = article_layout(content)
        if '<!-- VDN10_SHOWCASE -->' in source:
            content = content.replace('<section class="overview"',
                                      vdn10_showcase_html(integrations['vdn10']) + '<section class="overview"', 1)
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
                '<title>Spark-H3 · MiniMax-H3</title>'
                '<link rel="preload" href="/assets/fonts/assistant-latin-wght-normal.woff2" as="font" type="font/woff2" crossorigin>'
                '<link rel="preload" href="/assets/fonts/newsreader-latin-wght-normal.woff2" as="font" type="font/woff2" crossorigin>'
                '<link rel="stylesheet" href="/assets/katex/katex.min.css">'
                '<meta name="description" content="Spark-H3: adaptive block partitioning and reweighted pooling for MiniMax-H3 sparse attention.">'
                '<style>' + style + '</style></head><body id="top">'
                '<a class="skip-link" href="#overview">Skip to article</a>'
                '<header class="site-nav"><nav class="nav-inner" aria-label="Main navigation">'
                '<button class="brand brand-switch" type="button" data-mark="orange" '
                'aria-pressed="false" aria-label="Show green Spark wordmark" title="Switch wordmark color">'
                '<img class="brand-logo brand-logo-orange" src="/brand/spark-h3-wordmark.png" alt="">'
                '<img class="brand-logo brand-logo-light-green" '
                'src="/brand/spark-h3-wordmark-light-green.png" alt=""></button>'
                '<div class="nav-links"><a href="#spark-reblock">Reblock</a><a href="#spark-reweight">Reweight</a>'
                '<a class="nav-source" href="https://github.com/zechengtang/Spark-H3">Code ↗</a></div></nav></header><main>'
                + content + '</main><footer class="site-footer"><div class="footer-inner">'
                '<div><strong>Spark-H3</strong><br>SparkH3 Team · MiniMax-H3</div>'
                '<div class="footer-links"><a href="/source.md">Markdown source ↗</a>'
                '<a href="#top">Back to top ↑</a></div></div></footer>'
                + RESIZE_ANIMATIONS + INTEGRATION_SCRIPT + VDN10_SCRIPT + MARK_SCRIPT + '</body></html>')
        result = (page.encode(), len(fragments))
        CACHE.update(fingerprint=fingerprint, result=result)
        return result


def resolve_file(route):
    distilled = resolve_integration_file(route)
    if distilled is not None:
        return distilled
    if route in {'/brand/spark-h3-wordmark.png', '/brand/spark-h3-wordmark-light-green.png'}:
        return REPO / 'assets' / route.removeprefix('/brand/')
    if route.startswith('/gallery/'):
        data = gallery_data()
        allowed = {c[m][kind] for c in data['cases'] for m in ('dense', 'ours') for kind in ('file', 'poster')}
        name = route.removeprefix('/gallery/')
        return Path(data['output_directory']) / name if name in allowed else None
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

    def send_media(self, path, head):
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if match and any(match.groups()):
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), end) if right else end
                else:
                    start = max(0, size - int(right))
            else:
                start = size
            if start >= size or start > end:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if not head:
            try:
                with path.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

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
            if route.startswith('/gallery/') or path.suffix.lower() in {'.mp4', '.mkv'}:
                self.send_media(path, head)
                return
            data = path.read_bytes()
            content_type = (mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            if (route.startswith('/references/') and not content_type.startswith('image/')) or route == '/source.md':
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
    assert 'Assistant Variable' in text and 'Newsreader Variable' in text
    assert text.count('rel="preload"') == 2
    assert 'url("/assets/fonts/' not in text
    assert 'src="/assets/denoise_vp.gif"' in text and 'Yang Song' in text
    assert len(re.findall(r'class="article-figure"', text)) == 2
    assert len(re.findall(r'class="table-wrap"', text)) == 5
    assert len(re.findall(r'class="table-toolbar"', text)) == 5
    assert len(re.findall(
        r'<div class="table-wrap"[^>]* data-table-index="\d+"', text)) == 5
    benchmark_tables = re.findall(
        r'<div class="table-results-group".*?</details></div>', text, re.S)
    assert len(benchmark_tables) == 2
    assert all('Scroll to compare' not in table for table in benchmark_tables)
    assert len(re.findall(r'<tr class="is-highlight(?: |")', text)) == 21
    assert text.count('data-intermediate-results') == 2
    assert len(re.findall(r'class="is-highlight is-intermediate-result"', text)) == 7
    assert len(re.findall(r'class="table-results-details"', text)) == 2
    assert text.count('Show 15% &amp; 20%') == 2
    assert text.count('Collapse 15% &amp; 20%') == 2
    assert 'aria-label="Benchmark Results results" data-columns="8"' in text
    assert text.count('class="density-profile">Speed</span>') == 4
    assert text.count('class="density-profile">Fidelity</span>') == 4
    assert text.count('class="density-profile">Balanced</span>') == 7
    assert '<th>Method</th>' in text and 'DiT<br>latency (s) ↓' in text
    benchmark_table = re.search(
        r'aria-label="Benchmark Results results".*?</table>', text, re.S).group(0)
    headers = re.findall(r'<th(?: [^>]*)?>(.*?)</th>', benchmark_table, re.S)
    assert headers == [
        'Method', 'density ↓', 'DiT<br>latency (s) ↓', 'DiT<br>speedup ↑',
        'PSNR ↑', 'SSIM ↑', 'LPIPS ↓', 'ATTN<br>speedup ↑']
    assert text.count('Spark-H3-Lite') >= 8
    assert '<span class="chapter-number">03 / EVALUATION</span>' in text
    assert '<span class="chapter-number">05 / PREVIEW</span>' in text
    assert '<span class="chapter-number">07 / REFERENCES</span>' in text
    assert '<span class="chapter-number">08 / CHANGELOG</span>' in text
    assert '<a class="component-link" href="#spark-ref2va-preview">' in text
    assert '<a class="component-link" href="#visual-comparisons">' in text
    assert text.count('class="component-link"') == 6
    assert '<a href="#spark-ref2va-preview">Ref2VA</a>' not in text
    assert 'Alibaba-PAI Acc 8-step LoRA' in text
    assert 'ByteDance DMAD 4-step LoRA' in text
    assert 'Fast-H3' not in text and 'FastH3: an alternative for fixed block partitioning' in text
    assert 'SelfLift two-stage sampling based on LBH upsampler' in text
    integration_order = [
        'FastH3: an alternative for fixed block partitioning',
        'LightX2V 8-step LoRA',
        'Larryvrh 8-step LoRA',
        'Alibaba-PAI Acc 8-step LoRA',
        'ByteDance DMAD 4-step LoRA',
        'SelfLift two-stage sampling based on LBH upsampler',
        'Video DeltaNet',
    ]
    assert [text.index(title) for title in integration_order] == sorted(
        text.index(title) for title in integration_order)
    assert text.count('class="integration-results-details"') == 7
    assert text.count('<details class="integration-results-details" open>') == 5
    assert text.count('<details class="integration-results-details"><summary>') == 2
    assert len(re.findall(r'class="speedup-badge(?: is-baseline)?"', text)) == 41
    assert text.count('class="speedup-badge is-baseline"') == 13
    assert '<span class="speedup-badge" title="Relative to Dense">1.56×</span>' in text
    assert '<span class="speedup-badge" title="Relative to Dense">3.50×</span>' in text
    for record in integration_data()['few-step']['variants']:
        assert f'data-src="{record["preview"]}"' in text
    for record in integration_data()['selflift']['variants']:
        assert f'data-src="{record["preview"]}"' in text
        route_record = json.loads((HERE / 'integration/media.json').read_text())[record['preview']]
        payload = resolve_integration_file(record['preview']).read_bytes()
        assert len(payload) == route_record['bytes']
        assert hashlib.sha256(payload).hexdigest() == route_record['sha256']
    ref2va = integration_data()['ref2va']
    assert 'Spark-Ref2VA Preview' in text
    assert ('<a href="https://github.com/MiniMax-AI/MiniMax-H3#case-ref2va">'
            'official 5-second Ref2VA case</a>') in text
    assert text.count('01 · Official Ref2VA case') == 1
    assert 'class="comparison-prompt static-prompt"' in text
    for record in ref2va['variants']:
        assert f'data-src="{record["preview"]}"' in text
        route_record = json.loads((HERE / 'integration/media.json').read_text())[record['preview']]
        payload = resolve_integration_file(record['preview']).read_bytes()
        assert len(payload) == route_record['bytes']
        assert hashlib.sha256(payload).hexdigest() == route_record['sha256']
    assert '<tr class="is-highlight">\n<td>BSA + Reblock</td>' in text
    assert '<tr class="is-highlight">\n<td>BSA</td>' not in text
    assert '<tr class="is-highlight">\n<td>Dense</td>' not in text
    assert GIF not in text  # Both image and direct-image link are local; article attribution remains external.
    for target in ('/.git/config', '/etc/passwd', '/assets/../../etc/passwd', '/references/.git/config'):
        assert resolve_file(target) is None
    for record in json.loads((RUNTIME / 'asset_manifest.json').read_text()):
        payload = (ASSETS / record['file']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
    print(json.dumps({'status': 'passed', 'math_fragments': count, 'html_bytes': len(page),
                      'image': '/assets/denoise_vp.gif', 'unrestricted_file_access': False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'check', 'serve'])
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()
    globals()['RUNTIME'] = args.runtime.expanduser().resolve()
    globals()['ASSETS'] = RUNTIME / 'assets'
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
