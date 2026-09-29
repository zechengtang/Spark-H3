#!/usr/bin/env bash
# Install the Spark ComfyUI node and its patched comfy-kitchen CUDA backend.
set -euo pipefail

if [[ $# -ne 1 || ! -x "$1" ]]; then
    echo "Usage: bash comfyui/install.sh /path/to/ComfyUI/python" >&2
    exit 2
fi

python_bin="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
node_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch="$node_dir/comfyui/patches/comfy-kitchen-spark-v0.2.36.patch"
revision="888b13e2c0e721f6576fe351a2ad79894b1c451f"
source_url="${SPARK_COMFY_KITCHEN_SOURCE:-https://github.com/Comfy-Org/comfy-kitchen.git}"
build_dir="$(mktemp -d -t spark-comfy-kitchen-XXXXXX)"
trap 'rm -rf "$build_dir"' EXIT

if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA nvcc is required to build the Spark backend (CUDA >= 12.8)." >&2
    exit 1
fi

"$python_bin" -m pip install -e "$node_dir[cuda]"
git clone --quiet --depth 1 --branch v0.2.36 --recursive "$source_url" "$build_dir/comfy-kitchen"
git -C "$build_dir/comfy-kitchen" checkout --quiet "$revision"
git -C "$build_dir/comfy-kitchen" submodule update --init --recursive --quiet
git -C "$build_dir/comfy-kitchen" apply --check "$patch"
git -C "$build_dir/comfy-kitchen" apply "$patch"

# Spark's current ComfyUI implementation targets compute capability 12.0.
COMFY_CUDA_ARCHS="${COMFY_CUDA_ARCHS:-120f}" \
    "$python_bin" -m pip install --force-reinstall --no-deps "$build_dir/comfy-kitchen"

"$python_bin" - <<'PY'
from comfy_kitchen.backends.cuda import spark_attn
assert callable(spark_attn)
print("Spark ComfyUI backend installed: comfy_kitchen.backends.cuda.spark_attn")
PY

echo "Restart ComfyUI to load the Spark node."
