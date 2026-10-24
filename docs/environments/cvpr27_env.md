# cvpr27_env — isolated H20 / SM90 environment (2026-10-22)

This records the shared, isolated environment that passed the local Sol and
Spark SM90 operator suite. The earlier KC A/B benchmark used the same core
PyTorch/CUDA versions in the original local venv. MiniMax-H3 124-frame latent
generation has also been validated on three H20 servers.

## Host and GPU

| Component | Observed value |
| --- | --- |
| OS | Ubuntu 22.04.4 LTS, x86_64, glibc 2.35 |
| Kernel | 6.8.0-88-generic |
| GPU | NVIDIA H20, compute capability 9.0, 97,871 MiB reported by `nvidia-smi` |
| NVIDIA driver | 550.54.15 |
| `nvidia-smi` CUDA version | 12.4 (driver capability) |
| System CUDA toolkit | `/usr/local/cuda-12.4`; `nvcc` 12.4.131 |

The Python environment is
`/mnt/cpfs/zechengtang/envs/cvpr27_env`, using CPython 3.10.19. Its
`pyvenv.cfg` has **`include-system-site-packages = false`**. Required CUDA,
testing, and model-library dependencies were installed into this venv; `pip
check` reports no broken requirements. The adjacent
[package inventory](cvpr27_env_packages.txt) records all 86 installed
distributions using `pip list --format=freeze`. It is a version inventory,
not a hash-locked installation file. The old `/tmp/spark-h20-venv` was the
source for the move and had system-site packages enabled. Its old files were
removed; `/tmp/spark-h20-venv` is now a compatibility symlink to the shared
isolated environment.

| Python component | Observed version |
| --- | --- |
| PyTorch | `2.8.0+cu126` |
| Diffusers | `0.41.0` (PyPI wheel) |
| PyTorch CUDA runtime | 12.6; `nvidia-cuda-runtime-cu12==12.6.77` |
| torchvision / torchaudio | `0.23.0+cu126` / `2.8.0+cu126` |
| Triton | 3.4.0 |
| NVIDIA CUTLASS DSL | 4.8.1 |
| cuda-python | 13.4.4 |
| apache-tvm-ffi | 0.1.15.post0 |
| NumPy | 1.26.4 |
| pytest | 8.3.3 |
| transformers / accelerate | 5.20.2 / 1.15.0 |
| safetensors / einops | 0.8.0 / 0.8.0 |
| av / soundfile | 17.1.0 / 0.14.0 |

Diffusers is installed as the standard `diffusers==0.41.0` PyPI wheel. Its
SHA-256 is `ea8918b7dfd92ce793b6db689a551b8cd25d52c6c0fcd3ade0706a5fd2a25990`,
matching `uv.lock`; no separate Diffusers source checkout is needed. Spark-H3
is installed editable from this shared repository. Its HEAD was
`36e78b6f15a6d4efef2cbfa383d73a54eb3ba6ed` with uncommitted SM90 changes;
the shared working tree, not HEAD alone, is the tested code state. Another
server mounting the same `/mnt/cpfs` path sees the environment and Spark-H3
source.

The repository's general dependency files do not pin this H20 runtime:
`pyproject.toml` and `requirements.txt` mainly set minimum versions, while
`uv.lock` resolves the project's CPU PyTorch index and a different PyTorch
version. Do not use `uv sync` from that lock to recreate this `cu126` setup.

## Required PyTorch 2.8 compile adaptation

PyTorch 2.8 does not list or accept the newer Inductor `deterministic` compile
option requested by Spark-H3's reproducibility repair. The supported policy in
this environment is to retain `compile_reproducibility_fix=True`, including the
eager attention boundary, while omitting only the unsupported Inductor option.
The acceleration plugin performs this capability check automatically.

Do not translate the missing option into
`torch.use_deterministic_algorithms(True)`: Spark's CUDA `kthvalue` path has no
strict deterministic implementation in PyTorch 2.8. Do not use
`compile_reproducibility_fix=False` as the compatibility mechanism either; the
legacy whole-block path repeatedly recompiles custom sparse kernels for prompt
shapes and was substantially slower in the three-server end-to-end check.

Benchmark records distinguish the requested policy from the effective option:
`compile_deterministic_requested` remains true, while
`compile_deterministic` is false when the installed Inductor does not support
that option. No `sitecustomize.py`, `CUBLAS_WORKSPACE_CONFIG`, or global
deterministic-algorithms switch is required for this PyTorch 2.8 adaptation.

## Use on another server with the same `/mnt/cpfs` mount

The venv and the editable Spark-H3 source are on the shared filesystem. Use it
directly if the second server mounts `/mnt/cpfs` at the same path and has a
compatible OS, NVIDIA driver, H20, and CPython 3.10 executable at
`/usr/local/bin/python` (the venv's Python symlink target). Check these before
experiments, then rerun the tests below on that server. The system CUDA toolkit
does not need to be 12.6: this environment uses `cu126` user-space libraries.

```bash
/mnt/cpfs/zechengtang/envs/cvpr27_env/bin/python -V
grep '^include-system-site-packages = false$' \
  /mnt/cpfs/zechengtang/envs/cvpr27_env/pyvenv.cfg
/mnt/cpfs/zechengtang/envs/cvpr27_env/bin/python -m pip check
```

Run from the Spark-H3 root. These variables match the operator tests;
`HF_HUB_OFFLINE=1` assumes all model assets are already local:

```bash
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTE_DSL_ARCH=sm_90a
```

The current server has a 550.54.15 driver and a CUDA 12.4 system toolkit.
`cu126` supplies CUDA 12.6 user-space libraries in the Python environment;
those libraries ran the tested SM90 kernels against the 550 driver. This does
not imply every CUDA 12.6 feature works with that driver.

## Validation performed here

With GPU 0 visible and Diffusers 0.41.0 installed, the targeted Sol/Spark suite
plus the MiniMax-H3 attention processor test completed with **27 passed**:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/cpfs/zechengtang/envs/cvpr27_env/bin/python -m pytest -q \
  tests/test_spark_reweight_sm90_sm100.py \
  tests/test_spark_route_backends_sm90_sm100.py \
  'tests/test_sol_spark.py::test_sol_exact_sinks' \
  'tests/test_sol_spark.py::test_h3_processor_context_and_restoration'
```

Before the Diffusers upgrade, the isolated shared venv passed **26 tests** on
H20 with the `0.40.0.dev0` checkout. The local, gitignored evidence file is
`reports/h20_10s768p_20261022/cvpr27_env_isolated_pytest.log`.
It also completed one 124-frame-equivalent, 37,296-token, 56-head KC A/B
smoke round with bitwise-equal outputs. Its local, gitignored record is
`reports/h20_10s768p_20261022/cvpr27_env_sm90_smoke.json`. The corresponding
historical A/B record is `reports/h20_10s768p_20261022/sm90_kc_default_ab.json`.
Those timings use fixed Q/K/V, route and summaries, not a full model run.
