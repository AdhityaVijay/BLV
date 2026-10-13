#!/usr/bin/env bash
# One-shot environment setup for the BLV project.
# Usage: bash scripts/setup_venv.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "Warning: $PYTHON_BIN not found, falling back to python3"
  PYTHON_BIN="python3"
fi

echo "Creating venv with $PYTHON_BIN..."
"$PYTHON_BIN" -m venv venv

echo "Activating venv and installing requirements..."
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "Done. Activate the venv in new shells with:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the demo with:"
echo "  python -m blv.main"
