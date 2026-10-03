"""
Prefix-cache run: fires a shared-prefix, multi-question workload
sequentially against vLLM (see cache_workload.py) to get real
prefix_cache_queries_total/prefix_cache_hits_total signal. Recording and
dashboard rendering are shared with metrics_test.py's record_run().

Run inside the vllm-metal venv:
    python cache_test.py [--max-num-seqs N]

Or via the helper script from this directory:
    ./run_cache.sh [...]
"""

import argparse

from basic_test import MODEL
from cache_workload import build_cache_jobs
from metrics_test import record_run


def cache_job(job: dict) -> tuple[str, str, str, dict]:
    payload = {
        "model": MODEL,
        "prompt": job["prompt"],
        "temperature": 0.7,
        "max_tokens": job["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return job["kind"], job["label"], "/v1/completions", payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help="Cap vLLM's scheduler batch size (passed through to `vllm serve`). "
        "Left at vLLM's own default if unset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [cache_job(j) for j in build_cache_jobs()]
    serve_args = ["--max-num-seqs", str(args.max_num_seqs)] if args.max_num_seqs is not None else []
    print(f"Cache mode: firing {len(jobs)} requests sequentially against a shared prefix.")
    record_run(jobs, serve_args, sequential=True)


if __name__ == "__main__":
    main()
