#!/bin/bash
# Observes an already-running vLLM instance and builds dashboard.html from
# its real traffic. Run from this directory.
#   ./run_observe.sh --host HOST --port PORT [--otlp-port 4318] [--duration SECONDS] [--model NAME]
#   (see observe.py --help)
set -e
source ~/.venv-vllm-metal/bin/activate || { echo "venv missing -- run install.sh first." >&2; exit 1; }
python observe.py "$@"
