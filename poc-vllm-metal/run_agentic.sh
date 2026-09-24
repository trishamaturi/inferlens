#!/bin/bash
# Records an agentic-loop run with agentic_test.py and builds dashboard.html. Run from this directory.
#   ./run_agentic.sh [--max-num-seqs N] [--max-tokens N]   (see agentic_test.py --help)
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python agentic_test.py "$@"
