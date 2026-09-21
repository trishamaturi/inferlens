#!/bin/bash
# Activates the vllm-metal venv and runs basic_test.py (run from this directory).
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python basic_test.py
