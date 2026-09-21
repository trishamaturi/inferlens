#!/bin/bash
# Records a run with metrics_test.py and builds dashboard.html. Run from this directory.
#   ./run_metrics.sh [--max-num-seqs N]
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python metrics_test.py "$@"
