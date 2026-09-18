# Publish the Spark-H3 blog on GitHub Pages

The exported site is static HTML, CSS, JavaScript, fonts, animations, images and 32 browser-preview videos. It needs no Python server or build tools at runtime. All site assets use relative URLs, including when hosted under `/Spark-H3/`.

The current export is recorded in `github_pages_export.json`. Its ZIP contains `index.html` at the root. Large media and generated bundles stay outside Git; deployment reads the bundle from a GitHub Release asset.

## Deploy the prepared export

1. Commit `.github/workflows/deploy-blog.yml` to `zechengtang/Spark-H3`.
2. Create a release and attach the prepared `spark-h3-site.zip` from the path in `github_pages_export.json`.
3. In the repository's **Settings → Pages**, select **GitHub Actions** as the publishing source.
4. Run **Actions → Deploy Spark-H3 blog → Run workflow**, entering that release's tag.

The workflow verifies every exported file's hash and deploys the package through GitHub Pages. With the repository's default Pages domain, the page will be at `https://zechengtang.github.io/Spark-H3/`.

No release, commit, push or deployment is performed by the exporter. The ZIP is also usable on other static hosts.

## Export an updated version

Run from this repository with its existing preview assets and experiment media available:

```bash
python docs/blogs/spark-attn/export_github_pages.py \
  --output /path/outside/git/new-export/site
```

The build requires the same Python Markdown, Node.js and cached KaTeX assets as `preview.py`. It preserves earlier exports by refusing to overwrite the destination. It produces `site/`, `spark-h3-site.zip` and `export.json`; upload the new ZIP to a new release and run the deployment workflow with that tag.

For a local project-path preview:

```bash
python -m http.server 8000 --directory /path/outside/git/new-export
```

Open `http://localhost:8000/site/`. Keep the entire directory structure intact.

Original FFV1 archives remain in experiment storage. The deployment contains browser previews, not the large lossless archives. Per-file sizes and hashes are in `site/site-manifest.json`.
