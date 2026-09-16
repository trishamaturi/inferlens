#!/bin/bash
# Runs basic_test.py using the vllm-metal venv, without needing to remember
# the activate path or cd into this directory first.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$HOME/.venv-vllm-metal"

if [[ ! -d "$venv" ]]; then
  echo "vllm-metal venv not found at $venv -- run install.sh first." >&2
  exit 1
fi

source "$venv/bin/activate"
python "$script_dir/basic_test.py"
