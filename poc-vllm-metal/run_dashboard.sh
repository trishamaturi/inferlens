#!/bin/bash
# Re-renders dashboard.html from recorded data, without recording. Run from this directory.
#   ./run_dashboard.sh [--run-id ID]
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python build_dashboard.py "$@"
