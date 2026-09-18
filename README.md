# Spark-Attn for MiniMax-H3

Spark-Attn is a block sparse attention method for MiniMax-H3 video generation. It combines two operations:

- **Reblocking** groups similar tokens into blocks so sparse
  attention can select more relevant interactions.
- **Reweighting** uses query-conditioned key/value summaries to approximate
  unselected blocks while preserving their attention mass and value contribution.

This repository provides the Spark attention kernels and MiniMax-H3 inference integration. Conditioning video, text, and audio retain
exact attention handling.

## Quick start

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

`inputs` contains your pipeline's generation arguments. The context manager
restores the original attention processors on exit.

See the [usage and configuration guide](h3_sparse_attention/README.md) for
requirements and options, or the [Spark-Attn article](docs/blogs/spark-attn/README.md)
for the method and illustrations. Kernel attributions are in
[third-party notices](sol_attn/THIRD_PARTY_NOTICES.md).

## TODO

- [ ] Release a ComfyUI version.
- [ ] Release the technical report.
- [ ] Conduct further evaluation.
