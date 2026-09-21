#!/bin/bash
# Records a scheduler-overloading run with stress_test.py and builds dashboard.html. Run from this directory.
#   ./run_stress.sh [--max-num-seqs N] [--num-hogs N] ...   (see stress_test.py --help)
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python stress_test.py "$@"
