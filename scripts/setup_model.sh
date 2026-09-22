#!/usr/bin/env bash
# Export and set up a fresh tiny U-Net circuit using the active Python environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
artifacts_dir="${1:-model}"
if [[ -e "$artifacts_dir/network.onnx" || -e "$artifacts_dir/pk.key" ]]; then
    echo "Setup already exists; supply a fresh artifacts directory." >&2
    exit 1
fi
python model/tiny_unet.py --output "$artifacts_dir/network.onnx" --seed 42
python model/setup_ezkl.py "$artifacts_dir/network.onnx" "$artifacts_dir"
