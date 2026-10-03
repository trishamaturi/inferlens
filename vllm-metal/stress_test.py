"""
Scheduler-overloading run: many concurrent short requests plus long-context
"hog" requests against a low --max-num-seqs cap, which is what actually
produces queueing/KV-pressure signal for the dashboard (see
stress_workload.py). Recording and dashboard rendering are shared with
metrics_test.py's record_run().

Run inside the vllm-metal venv:
    python stress_test.py [--max-num-seqs N] [--num-gpu-blocks-override N --max-model-len N]

Or via the helper script from this directory:
    ./run_stress.sh [...]
"""

import argparse

from basic_test import MODEL
from metrics_test import record_run
from stress_workload import build_stress_jobs


def stress_job(job: dict) -> tuple[str, str, str, dict]:
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
        "--max-num-seqs", type=int, default=4,
        help="Cap vLLM's scheduler batch size (passed through to `vllm serve`). Low by default to force real queueing.",
    )
    parser.add_argument(
        "--num-gpu-blocks-override", type=int, default=None,
        help="Shrink vLLM's KV cache pool to this many blocks (passed through to `vllm serve`, which documents "
        "this flag as 'used for testing preemption'). Use a value well below the profiled default (~4700 blocks "
        "here) to force real vllm:num_preemptions_total once concurrent requests' KV demand exceeds it. vLLM "
        "refuses to start unless --max-model-len also fits within this budget -- pair the two.",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Passed through to `vllm serve`. Required alongside --num-gpu-blocks-override, since vLLM won't "
        "start if the shrunk KV cache can't hold even one request at the model's default max context length.",
    )
    parser.add_argument("--num-hogs", type=int, default=2, help="Long-context requests to fire.")
    parser.add_argument("--short-multiplier", type=int, default=4, help="Copies of each short prompt.")
    parser.add_argument("--hog-words", type=int, default=3000, help="Approx. word count per hog prompt.")
    parser.add_argument(
        "--short-max-tokens", type=int, default=64,
        help="max_tokens for the short prompts. Set high (e.g. 800) with --num-hogs 0 to target "
        "mid-decode preemption instead of admission-time queueing (see stress_workload.build_stress_jobs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [
        stress_job(j)
        for j in build_stress_jobs(
            short_multiplier=args.short_multiplier, num_hogs=args.num_hogs, hog_words=args.hog_words,
            short_max_tokens=args.short_max_tokens,
        )
    ]
    serve_args = ["--max-num-seqs", str(args.max_num_seqs)]
    if args.num_gpu_blocks_override is not None:
        serve_args += ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
    if args.max_model_len is not None:
        serve_args += ["--max-model-len", str(args.max_model_len)]
    print(
        f"Stress mode: firing {len(jobs)} concurrent requests "
        f"(max-num-seqs={args.max_num_seqs}, num-gpu-blocks-override={args.num_gpu_blocks_override})."
    )
    record_run(jobs, serve_args)


if __name__ == "__main__":
    main()
