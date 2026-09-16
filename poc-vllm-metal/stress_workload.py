"""
A workload deliberately sized to overload vLLM's scheduler, unlike
basic_test.py's tiny smoke-test prompts (which never queue or contend for
anything). This is what actually gives the dashboard's waterfall and
correlation view something real to show -- PLAN.md's own motivating
question is "was this short request queued behind a half-million-token
context request hogging the batch?"; you can't observe that from 8 prompts
totaling a few hundred tokens.

Used by metrics_test.py's --stress flag.
"""

SHORT_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

HOG_FILLER = (
    "In distributed inference serving, a scheduler must balance many "
    "concurrent requests against a fixed pool of GPU memory used for the "
    "key-value cache, and preemption or queuing occurs when demand exceeds "
    "capacity. "
)


def long_context_prompt(target_words: int = 3000) -> str:
    """A long, coherent-ish filler prompt sized in words (roughly ~1.3
    tokens/word for English text), meant to dominate prefill time and
    consume a real chunk of KV cache -- not the content, just the size."""
    words_per_repeat = len(HOG_FILLER.split())
    repeats = target_words // words_per_repeat + 1
    return (HOG_FILLER * repeats).strip()


def build_stress_jobs(short_multiplier: int = 4, num_hogs: int = 2, hog_words: int = 3000) -> list[dict]:
    """Returns job specs: `short_multiplier` copies of each short prompt
    (fired concurrently, so vLLM sees real repeated load) plus `num_hogs`
    long-context requests to create genuine queueing/KV pressure."""
    jobs = []
    for i in range(short_multiplier):
        for prompt in SHORT_PROMPTS:
            jobs.append({"kind": "completion", "label": f"{prompt} (#{i + 1})", "prompt": prompt, "max_tokens": 64})
    for i in range(num_hogs):
        prompt = long_context_prompt(hog_words)
        jobs.append({
            "kind": "hog",
            "label": f"long-context hog #{i + 1} (~{hog_words} words)",
            "prompt": prompt,
            "max_tokens": 64,
        })
    return jobs
