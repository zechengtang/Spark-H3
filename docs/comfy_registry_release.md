# Comfy Registry release

Spark-H3 publishes one cross-platform Registry archive under publisher
`zechengtang` and node ID `comfyui-spark-h3`. The Registry archive contains the
node, installer, and workflows, but not architecture-specific backend wheels.
During installation, the installer downloads the matching platform archive from
the pinned GitHub Release and extracts an isolated `spark-h3-kitchen` wheel.
Users therefore need GitHub access during installation unless they provide a
matching local wheel.

Registry version `0.2.0` routes backend installation to GitHub Release
`comfyui-v0.2.0`. It supports ComfyUI 0.38.x-0.40.x and 0.42.x-0.44.x and keeps
ComfyUI's official `comfy-kitchen` package separate from Spark-H3's isolated
backend. Older withdrawn Registry version numbers cannot be reused. The
Registry security review status is visible on the
[node page](https://registry.comfy.org/nodes/comfyui-spark-h3).

The Registry node description is:

> Spark-H3 accelerates ComfyUI's native MiniMax-H3 video generation with
> adaptive block-sparse attention by grouping similar tokens and correcting
> tail tokens through reweighting. v0.1.3 Linux benchmarks (LightX2V 768p
> 8-step, 1344x768, 345 frames, 24 FPS; topk_ratio 20%, warmup_steps 2, dense
> SDPA): RTX 3090 - Fidelity 2.42x / Speed 4.62x; RTX 4090 - 2.37x / 4.35x;
> RTX PRO 6000 Blackwell - 2.05x / 3.15x.

## Build locally

Build the Registry candidate and pin it to the matching GitHub Release tag:

```bash
python tools/build_comfyui_package.py \
  --registry \
  --version 0.2.0 \
  --backend-release-tag comfyui-v0.2.0 \
  --publisher-id zechengtang \
  --output-dir dist/registry
```

The preview archive is
`dist/registry/ComfyUI-Spark-H3-0.2.0-registry.zip`. Before publishing it,
confirm that the pinned GitHub Release contains all six Linux/Windows and
SM86/SM89/SM120 platform ZIPs.

## Publish

Create a Registry publishing API key for `@zechengtang`, then save it as the
GitHub repository secret `REGISTRY_ACCESS_TOKEN`. Run the **Publish Comfy
Registry package** workflow manually with version `0.2.0`, backend release tag
`comfyui-v0.2.0`, and the release changelog. It verifies that the six platform
archives are present, builds the Registry package, runs `comfy node validate`,
and publishes Registry version 0.2.0.

The workflow is intentionally manual because a Registry version must not be
published twice. For later releases, update the version and release tag in the
workflow only after the matching six platform archives are public.
