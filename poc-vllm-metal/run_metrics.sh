#!/bin/bash
# Runs the vLLM /metrics + per-query recorder (default) or the dashboard
# builder, using the vllm-metal venv. Usage:
#   ./run_metrics.sh                              # record a run, build dashboard.html
#   ./run_metrics.sh build_dashboard.py [--run-id ID]   # re-render without recording
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$HOME/.venv-vllm-metal"
target="${1:-metrics_test.py}"
[[ $# -gt 0 ]] && shift

if [[ ! -d "$venv" ]]; then
  echo "vllm-metal venv not found at $venv -- run install.sh first." >&2
  exit 1
fi

source "$venv/bin/activate"
python "$script_dir/$target" "$@"
