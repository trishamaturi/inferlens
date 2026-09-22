"""
A workload built to exercise vLLM's automatic prefix caching: a shared,
multi-block prefix (~120 tokens, well over the 16-token block size) reused
across several distinct questions, plus a couple of verbatim repeats. Used
by cache_test.py, which fires these sequentially -- concurrent firing (as
stress_test.py uses) races every request's prefill in parallel before any
block is cached for a sibling to reuse, which is why every run recorded
before this script showed 0 prefix_cache_hits_total, despite
short_multiplier copies sharing identical prompts (also too short to span
even one block). cache_test.py fixes both: real hit rates confirmed at
~82% (shared-prefix requests) and ~99% (verbatim repeats) -- see
README.md section 3.
"""

SHARED_PREFIX = (
    "You are a helpful assistant answering questions about distributed "
    "inference systems. Keep in mind that a scheduler must balance many "
    "concurrent requests against a fixed pool of GPU memory used for the "
    "key-value cache, that preemption or queuing occurs when demand "
    "exceeds capacity, and that prefix caching lets requests sharing a "
    "common context reuse already-computed KV blocks instead of "
    "recomputing them from scratch, which is a meaningful throughput win "
    "for multi-turn or shared-context workloads. "
)

QUESTIONS = [
    "What causes queueing in this scheduler?",
    "How does preemption differ from queueing?",
    "What is prefix caching used for?",
    "Why does a fixed KV cache pool matter here?",
]


def build_cache_jobs() -> list[dict]:
    """Returns job specs in firing order: each of QUESTIONS appended to
    SHARED_PREFIX once (the first pays full prefill cost; later ones can
    hit the now-cached prefix blocks), then the first two repeated
    verbatim to also exercise a full-prompt cache hit (the "ask it again"
    multi-turn case)."""
    jobs = [
        {"kind": "completion", "label": q, "prompt": SHARED_PREFIX + q, "max_tokens": 64}
        for q in QUESTIONS
    ]
    jobs += [
        {
            "kind": "completion", "label": f"{QUESTIONS[0]} (repeat)",
            "prompt": SHARED_PREFIX + QUESTIONS[0], "max_tokens": 64,
        },
        {
            "kind": "completion", "label": f"{QUESTIONS[1]} (repeat)",
            "prompt": SHARED_PREFIX + QUESTIONS[1], "max_tokens": 64,
        },
    ]
    return jobs
