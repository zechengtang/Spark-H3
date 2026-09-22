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

STYLE = (HERE / 'blog.css').read_text()


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
            '<a class="button" href="https://github.com/zechengtang/Spark-H3">Code <span aria-hidden="true">↗</span></a>'
            '</div></div>' + attention_art() + '</div>'
            '<div class="component-links">' + ''.join(links) + '</div></div></section>')
    overview = ('<section class="overview" id="overview" aria-labelledby="overview-title">'
                '<div class="section-heading"><p class="eyebrow">Overview</p>'
                '<h2 id="overview-title">Better BSA</h2></div>'
                '<div class="overview-copy">' + intro + '</div></section>')
    chapters = []
    figure_index = 0
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
        body = re.sub(r'(<table>.*?</table>)',
                      lambda m: f'<div class="table-wrap" role="region" tabindex="0" aria-label="{label}">{m.group(1)}</div>',
                      body, flags=re.S)
        chapters.append(f'<section class="chapter" id="{section.group(1)}" '
                        f'aria-labelledby="{section.group(1)}-title">'
                        '<div class="section-heading">'
                        f'<span class="chapter-number">0{index + 1} / METHOD</span>'
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
            videos.append(
                f'<figure class="comparison-video {method}"><figcaption>'
                f'<strong class="comparison-method"><i aria-hidden="true"></i>{label}</strong>'
                '<span class="comparison-timing"><span class="timing-label">Denoising time</span>'
                f'<span class="timing-value">{item["denoise_seconds"]:.2f}<small> s</small></span>'
                f'<span class="timing-speedup">{speedup:.2f}×</span>'
                '</span></figcaption>'
                f'<video autoplay muted loop playsinline preload="metadata" disablepictureinpicture disableremoteplayback '
                f'aria-label="{label}: {sid}" poster="/gallery/{item["poster"]}" '
                f'src="/gallery/{item["file"]}"></video></figure>')
        cards.append(
            f'<article class="comparison-card" aria-labelledby="case-{sid}">'
            '<div class="comparison-heading"><div class="comparison-identity">'
            f'<h3 class="comparison-index" id="case-{sid}" aria-label="Comparison {index}">{index:02d}</h3>'
            '<button class="video-reset" type="button" title="同时从头播放本 prompt 的所有视频">⟲ 复位</button>'
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
            for name in ('turbo-lora', 'fasth3-0753', 'vdn10')}


def resolve_integration_file(route):
    for name in ('turbo-lora', 'fasth3-0753', 'vdn10'):
        if route == f'/{name}/manifest.json':
            return HERE / 'integration' / f'{name}.json'
    record = json.loads((HERE / 'integration/media.json').read_text()).get(route)
    if record is None:
        return None
    path = (RUNTIME / record['file']).resolve()
    return path if path.is_relative_to(RUNTIME.resolve()) else None


def integration_html(data, family):
    manifest = data[family]
    records = {r['variant']: r for r in manifest['variants']}
    groups = []
    for prompt_index, prompt in enumerate(('0753', '0685'), 1):
        if family == 'turbo-lora':
            mode_specs = [('dense', 'Dense', '8 denoising steps'),
                          ('spark10', 'Spark-H3-10pct', '8 denoising steps · first 2 steps + first layer dense'),
                          ('spark10_nowarm', 'Spark-H3-10pct', '8 denoising steps · no warmup, fully sparse')]
            models = [(f'{prompt}_{model}_{mode}',
                       ('LightX2V' if model == 'lightx2v' else 'Larry v4 EMA') + f' · {title}',
                       detail)
                      for model in ('lightx2v', 'larry') for mode, title, detail in mode_specs]
            prompt_text = manifest['cases'][prompt]['generation_prompt']
        else:
            prefix = '' if prompt == '0753' else '0685_'
            models = [(prefix + name, title, detail) for name, title, detail in (
                ('v1_dense', 'FastH3 v1 Dense', '4 denoising steps · full attention'),
                ('v1_dense_spark10_nowarm', 'FastH3 v1 Dense · Spark-H3-10pct', '4 denoising steps · no warmup'),
                ('v1_dense_spark10_warm', 'FastH3 v1 Dense · Spark-H3-10pct + warmup',
                 '4 denoising steps · first step + first layer dense'),
                ('v1_vsa', 'FastH3 v1 VSA', '4 denoising steps · 90% sparsity'),
                ('v2_vsa', 'FastH3 v2 VSA', '8 denoising steps · 80% sparsity'))]
            prompt_text = manifest['prompt' if prompt == '0753' else 'prompt0685']
        cards = []
        for variant, title, detail in models:
            record = records[variant]
            video = ('<video autoplay muted loop playsinline preload="none" disablepictureinpicture disableremoteplayback '
                     f'aria-label="{html.escape(title)}: {prompt_index:02d}" '
                     f'data-src="{html.escape(record["preview"], quote=True)}"></video>')
            cards.append(
                '<figure class="integration-video"><figcaption>'
                f'<strong>{html.escape(title)}</strong><span>{html.escape(detail)}</span>'
                '</figcaption>' + video + '<div class="integration-video-footer">'
                f'<span>Generation <strong>{record["generation_seconds"]:.2f} s</strong></span>'
                '</div></figure>')
        excerpt = re.sub(r'^integrated_multimodal_description:\s*(?:\[Shot 1\]\s*)?', '', prompt_text)
        excerpt = ' '.join(excerpt.split())
        groups.append(
            '<div class="integration-group">'
            f'<h4>{prompt_index:02d}<button class="video-reset" type="button" '
            'title="同时从头播放本 prompt 的所有视频">⟲ 复位</button></h4>'
            '<div class="integration-video-grid">' + ''.join(cards) + '</div>'
            '<details class="comparison-prompt"><summary>'
            '<span class="prompt-heading"><span class="prompt-label">Prompt</span>'
            '<span class="prompt-toggle"><span class="prompt-expand">Show full prompt</span>'
            '<span class="prompt-collapse">Collapse prompt</span><span class="prompt-chevron" aria-hidden="true">↗</span></span></span>'
            '<span class="prompt-excerpt">' + html.escape(excerpt) + '</span></summary>'
            f'<div class="prompt-full">{html.escape(prompt_text)}</div></details></div>')
    return '<div class="integration-gallery">' + ''.join(groups) + '</div>'


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
            '<span class="model-tag"><span class="model-name">LightX2V Turbo</span>'
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
            'aria-label="LightX2V Spark-H3-10pct video results">'
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
  const updatePlayback = () => {
    const rect = showcase.getBoundingClientRect();
    const visible = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
    if (document.hidden || visible < Math.min(120, rect.height * 0.25)) {
      videos.forEach(video => video.pause());
      return;
    }
    videos.forEach((video, index) => {
      if (index === activeIndex) { prepare(video); video.play().catch(() => {}); }
      else video.pause();
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
        videos[index].pause();
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
    if (event.pointerType === 'mouse') return;
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
      if (index !== activeIndex) { video.pause(); return; }
      videos.forEach((other, otherIndex) => { if (otherIndex !== activeIndex) other.pause(); });
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
  const prepare = video => {
    if (video.dataset.src) {
      video.src = video.dataset.src;
      delete video.dataset.src;
      video.preload = 'metadata';
      video.load();
    }
  };
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      prepare(entry.target);
      entry.target.play().catch(() => {});
      observer.unobserve(entry.target);
    }
  }, {rootMargin: '200px'});
  document.querySelectorAll('.integration-video video').forEach(v => observer.observe(v));
  document.addEventListener('click', event => {
    const button = event.target.closest('.video-reset');
    if (!button) return;
    const group = button.closest('.comparison-card, .integration-group');
    if (!group) return;
    for (const video of group.querySelectorAll('video')) {
      prepare(video);
      video.currentTime = 0;
      video.play().catch(() => {});
    }
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
        content = content.replace('<!-- FASTH3_INTEGRATION -->', integration_html(integrations, 'fasth3-0753'))
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
                '<link rel="stylesheet" href="/assets/katex/katex.min.css">'
                '<meta name="description" content="Spark-H3: adaptive block partitioning and reweighted pooling for MiniMax-H3 sparse attention.">'
                '<style>' + style + '</style></head><body id="top">'
                '<a class="skip-link" href="#overview">Skip to article</a>'
                '<header class="site-nav"><nav class="nav-inner" aria-label="Main navigation">'
                '<a class="brand" href="#top"><span class="brand-mark" aria-hidden="true">✳</span>Spark-H3</a>'
                '<div class="nav-links"><a href="#spark-reblock">Reblock</a><a href="#spark-reweight">Reweight</a>'
                '<a class="nav-source" href="https://github.com/zechengtang/Spark-H3">Code ↗</a></div></nav></header><main>'
                + content + '</main><footer class="site-footer"><div class="footer-inner">'
                '<div><strong>Spark-H3</strong><br>SparkH3 Team · MiniMax-H3</div>'
                '<div class="footer-links"><a href="/source.md">Markdown source ↗</a>'
                '<a href="#top">Back to top ↑</a></div></div></footer>'
                + RESIZE_ANIMATIONS + INTEGRATION_SCRIPT + VDN10_SCRIPT + '</body></html>')
        result = (page.encode(), len(fragments))
        CACHE.update(fingerprint=fingerprint, result=result)
        return result


def resolve_file(route):
    distilled = resolve_integration_file(route)
    if distilled is not None:
        return distilled
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
    assert 'src="/assets/denoise_vp.gif"' in text and 'Yang Song' in text
    assert len(re.findall(r'class="article-figure"', text)) == 3
    assert len(re.findall(r'class="table-wrap"', text)) == 4
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
