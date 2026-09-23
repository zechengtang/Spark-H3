# Read the Spark-H3 blog

[Read the article](README.md), or render its interactive animations, visual
comparisons, and distilled-model integration samples locally.

## Local preview

From the repository root (Python Markdown and Node.js are required):

```bash
python -m pip install -r docs/blogs/spark-attn/requirements.txt
python docs/blogs/spark-attn/preview.py prepare
python docs/blogs/spark-attn/preview.py check
python docs/blogs/spark-attn/preview.py serve --port 6008
```

Open http://127.0.0.1:6008/. Choose another port if it is already occupied.
The preview has no dependency on the MiniMax-H3-Sparse checkout or its gallery
server. Model weights and a GPU are not needed.

`prepare` downloads the attributed Yang Song illustration, pinned KaTeX
assets, and pinned Assistant/Newsreader WOFF2 fonts into the ignored
`.preview/` directory. KaTeX and both SIL Open Font License texts are retained
beside their assets.
Use `--runtime /path/to/cache` on each command, or set `SPARK_BLOG_RUNTIME`, to
choose another location. Rendering then works offline once media is present.

## Video assets

This workspace has all 32 browser-preview videos and comparison posters in
`.preview/`. Large media is excluded from Git. On another machine, supply:

- `<runtime>/gallery/`: filenames listed in `gallery.json`.
- `<runtime>/integration/`: paths listed in `integration/media.json`.

The integration manifests beside `media.json` supply prompts, labels, and
measurements. The preview serves only allowlisted files and supports video
byte-range requests. Missing video assets do not prevent article rendering,
but their browser requests will return 404.

For a complete portable page with all media, use the static ZIP documented in
[GITHUB_PAGES.md](GITHUB_PAGES.md). It needs no Python server at runtime.

Rebuild the animations with:

```bash
python docs/blogs/spark-attn/animations/build.py
```
