"""
Agentic-loop run: fires several sessions concurrently, each its own
sequential chain of turns where turn N's prompt includes the real
assistant reply from turn N-1 (see agentic_workload.py). This is the
traffic shape none of stress_test.py/cache_test.py/metrics_test.py
produce -- every job they fire is independent of every other job's
*output*, even cache_test.py's sequential-but-independent prompts.
Recording and dashboard rendering are shared with metrics_test.py's
record_run() via its `sessions=` mode (see run_sessions).

Run inside the vllm-metal venv:
    python agentic_test.py [--max-num-seqs N] [--max-tokens N]

Or via the helper script from this directory:
    ./run_agentic.sh [...]
"""

import argparse

from agentic_workload import build_sessions
from metrics_test import record_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="Cap vLLM's scheduler batch size (passed through to `vllm serve`). "
        "Left at vLLM's own default if unset.",
    )
    parser.add_argument("--max-tokens", type=int, default=96, help="max_tokens per turn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = build_sessions()
    total_turns = sum(len(s["turns"]) for s in sessions)
    serve_args = ["--max-num-seqs", str(args.max_num_seqs)] if args.max_num_seqs is not None else []
    print(f"Agentic mode: firing {len(sessions)} concurrent sessions, {total_turns} total turns.")
    record_run(None, serve_args, sessions=sessions, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
