#!/bin/bash
# Records a prefix-cache run with cache_test.py and builds dashboard.html. Run from this directory.
#   ./run_cache.sh [--max-num-seqs N]   (see cache_test.py --help)
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python cache_test.py "$@"
