#!/bin/bash
# Runs the vLLM /metrics + per-query recorder (default) or the dashboard
# builder, using the vllm-metal venv. Usage:
#   ./run_metrics.sh [--stress] [--max-num-seqs N]      # record a run, build dashboard.html
#   ./run_metrics.sh build_dashboard.py [--run-id ID]   # re-render without recording
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$HOME/.venv-vllm-metal"

# Only treat $1 as the target script if it names one (foo.py); otherwise
# every arg is a flag passed through to the default recorder script.
if [[ "${1:-}" == *.py ]]; then
  target="$1"
  shift
else
  target="metrics_test.py"
fi

if [[ ! -d "$venv" ]]; then
  echo "vllm-metal venv not found at $venv -- run install.sh first." >&2
  exit 1
fi

source "$venv/bin/activate"
python "$script_dir/$target" "$@"
