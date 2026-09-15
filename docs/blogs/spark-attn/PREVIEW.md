# Read the Spark-Attn blog

[Read the Markdown article](README.md) on GitHub. GitHub displays its equations
and tables; interactive animations run in the browser preview below. The two
animation HTML files can also be downloaded and opened directly offline.

## Local preview

From the repository root, install the small preview dependency. Node.js is also
required to render equations with KaTeX; model weights and a GPU are not needed.

```bash
python -m pip install -r docs/blogs/spark-attn/requirements.txt
python docs/blogs/spark-attn/preview.py prepare
python docs/blogs/spark-attn/preview.py check
python docs/blogs/spark-attn/preview.py serve
```

Open **http://127.0.0.1:6006/**. Stop the server with Ctrl+C.

`prepare` downloads the attributed Yang Song illustration and pinned KaTeX
assets, including its license, into the ignored `.preview/` directory beside
this file. Subsequent rendering works offline, including equations, fonts,
and both animations; external attribution links still need internet access.
Use `--runtime /path/to/cache` on each command to choose a different cache,
and `serve --port 8000` to choose another port.

Rebuild the self-contained animations with:

```bash
python docs/blogs/spark-attn/animations/build.py
```

These illustrate toy examples, not measured model performance. Their playback
controls support pause, replay, and timeline scrubbing.
