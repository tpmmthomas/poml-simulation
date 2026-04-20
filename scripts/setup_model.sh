#!/usr/bin/env bash
# Setup script: exports the tiny U-Net to ONNX and runs EZKL setup.
# Run from the project root: bash scripts/setup_model.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Step 1: Export Tiny U-Net to ONNX ==="
python model/tiny_unet.py

echo ""
echo "=== Step 2: Run EZKL setup ==="
python model/setup_ezkl.py model/network.onnx model/

echo ""
echo "=== Setup complete ==="
echo "Artifacts in model/:"
ls -la model/*.onnx model/*.ezkl model/*.json model/*.key model/*.srs 2>/dev/null || true
