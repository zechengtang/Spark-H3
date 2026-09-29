<div align="center">

<h1><img src="assets/spark-h3-wordmark.png" alt="Spark-H3" width="270"></h1>
<h3>Block sparse attention for MiniMax-H3 video generation</h3>

<p><strong>Reblock similar tokens · Reweight tail tokens</strong></p>

<p>
  <a href="https://zechengtang.github.io/Spark-H3/"><img src="https://img.shields.io/badge/Blog-Spark--H3-f97316?style=flat-square" alt="Spark-H3 technical blog"></a>
  <a href="comfyui/README.md"><img src="https://img.shields.io/badge/ComfyUI-Spark_Node-f97316?style=flat-square" alt="Spark-H3 ComfyUI node"></a>
  <a href="h3_sparse_attention/README.md"><img src="https://img.shields.io/badge/Docs-Usage_%26_Configuration-2563eb?style=flat-square" alt="Usage and configuration guide"></a>
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-MiniMax--H3-ffc107?style=flat-square" alt="MiniMax-H3 model weights on Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-15803d?style=flat-square" alt="License: Apache 2.0"></a>
</p>

<p>
  <a href="#demo">🎬 Demo</a> &nbsp;·&nbsp;
  <a href="#quick-start">🚀 Quick Start</a> &nbsp;·&nbsp;
  <a href="https://zechengtang.github.io/Spark-H3/">📖 Technical Blog</a> &nbsp;·&nbsp;
  <a href="#todo">🗓️ TODO</a>
</p>

</div>

---

## ✨ Spark-Attn

Spark-Attn is a block sparse attention method for MiniMax-H3 video generation.
It combines two operations:

- **🧩 Reblocking** groups similar tokens into blocks so sparse attention can
  select more relevant interactions.
- **⚖️ Reweighting** assigns different weights to tokens within each tail block
  when building key/value summaries, with a correction for attention mass.

Spark-H3 provides the Spark attention kernels and MiniMax-H3 inference
integration. Conditioning video, text, and audio retain exact attention handling.

<a id="demo"></a>

## 🎬 Demo

The full demo compares Dense, Spark-H3, and Sol-H3 with matched seeds, then
shows Spark-H3 integrated with the 8-step LightX2V LoRA. The animated preview
collects the current slow-motion detail segments from Cases 1–3.

[![Spark-H3 visual comparison: Dense vs. Spark-H3 vs. Sol-H3](assets/spark-h3-demo-preview.webp)](assets/spark-h3-demo.mp4)

<p align="center"><a href="assets/spark-h3-demo.mp4">▶ Watch the full demo with sound</a></p>

<a id="quick-start"></a>

## 🚀 Quick Start

Set up your H3 pipeline using the [upstream MiniMax-H3 instructions](https://github.com/MiniMax-AI/MiniMax-H3),
then install Spark in the same environment with a compatible PyTorch/CUDA stack:

```bash
python -m pip install -e '.[cuda]'
```

Wrap an already loaded pipeline with the Spark installer:

```python
from h3_sparse_attention import install_h3_spark_attn

with install_h3_spark_attn(pipe.transformer, num_inference_steps=20):
    result = pipe(**inputs, num_inference_steps=20)
```

### Step-count convention

As explained in the [Hugging Face MiniMaxH3Scheduler documentation](https://huggingface.co/docs/diffusers/main/api/schedulers/minimax_h3),
`num_inference_steps=N` specifies **N sigma grid points**, including the final
`0`. The default schedule evaluates the transformer only at the preceding
points (`self.timesteps = 1 - sigmas[:-1]`), so `N=20` produces **19 model
evaluations**. Spark follows that pipeline convention.

<!-- ### Benchmark comparability

Match the **actual** `torch.compile` state across all Diffusers runs before
comparing quality or speed. In a controlled 25-prompt H3 check, switching
denoising compilation changed Spark PSNR by **3.15 dB**, despite otherwise
matched settings. This is a protocol difference, not evidence of a Spark
kernel regression; see the [compile discrepancy audit](docs/diffusers_compile_psnr_attribution_20260927.md).
The sibling MiniMax-H3-Benchmark's standard Diffusers denoising runner and
radial/attention harnesses require compilation and verify that all transformer
blocks were wrapped. Do not mix their results
with historical uncompiled experiments.

`inputs` contains your pipeline's generation arguments. The context manager
restores the original attention processors on exit.

→ See the [usage and configuration guide](h3_sparse_attention/README.md) for
requirements, defaults, and options. -->

<a id="todo"></a>

## 🗓️ TODO

- [x] Release the ComfyUI implementation.
- [ ] Release Ref2VA inference examples.
- [ ] Release the technical report.
- [ ] Optimize the ComfyUI implementation for better end-to-end efficiency.
- [ ] Implement SM80, SM90 support.

## 🤝 Acknowledgments
 
Thanks to [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) and
[Sol-Attn / Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine) for the
model and attention infrastructure. See the [third-party notices](sol_attn/THIRD_PARTY_NOTICES.md)
for kernel attributions and included licenses.

## 📄 License

Spark-Attn’s original code and documentation are licensed under the
[Apache License 2.0](LICENSE). Third-party components retain their original
licenses and notices; see [third-party notices](sol_attn/THIRD_PARTY_NOTICES.md).
MiniMax-H3 model weights remain subject to their upstream license.
