#!/usr/bin/env bash
# Compatibility wrapper around the cross-platform Python installer.
set -euo pipefail

if [[ $# -ne 1 || ! -x "$1" ]]; then
    echo "Usage: bash comfyui/install.sh /path/to/ComfyUI/python" >&2
    exit 2
fi

python_bin="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
installer="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.py"
exec "$python_bin" "$installer"
