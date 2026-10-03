"""
Session specs for agentic_test.py: several independent, multi-turn
"agent" conversations, each a sequential chain where every turn is a
genuine follow-up to the model's own prior answer (formed at record time
by metrics_test.run_sessions, not scripted in advance) -- unlike every
other workload here (stress_test.py, cache_test.py), which only ever
fires requests that are independent of each other's *output*.

Each session models one agent working a task across several turns, with
a per-session system prompt framing the task. Turn text is deliberately
generic/short-answer-shaped (not literal tool-call JSON) since the model
being exercised is a small base chat model, not a tool-calling-tuned one
-- what's under test is the traffic *shape* (concurrent, per-session
growing context, turn N depends on turn N-1), not real tool execution.
"""

SESSIONS = [
    {
        "label": "agent A (incident triage)",
        "system": "You are an SRE assistant helping triage a production incident. Be concise.",
        "turns": [
            "We're seeing high p99 latency on our inference service. What are the top 3 likely causes?",
            "GPU utilization is only at 40%, so it's probably not compute-bound. Does that change your answer?",
            "Given that, what's the single most useful metric to check next?",
        ],
    },
    {
        "label": "agent B (code review)",
        "system": "You are a code reviewer. Be concise.",
        "turns": [
            "A function acquires a lock, calls another function that also acquires the same lock, and never releases on the error path. What's wrong with this?",
            "Assume the lock is not reentrant. What's the simplest fix?",
            "Would you recommend a context manager for this instead? Why or why not?",
        ],
    },
    {
        "label": "agent C (data pipeline debugging)",
        "system": "You are a data engineering assistant. Be concise.",
        "turns": [
            "A daily batch job's row count dropped by 90% overnight with no code changes. What would you check first?",
            "The upstream source's schema didn't change and row counts there look normal. What's next?",
            "Turns out a filter step's date comparison is timezone-naive. How would you prevent this class of bug going forward?",
        ],
    },
    {
        "label": "agent D (capacity planning)",
        "system": "You are a capacity planning assistant. Be concise.",
        "turns": [
            "Our GPU fleet is at 85% average utilization but requests are still queueing during peak hours. Why might that happen even with headroom left?",
            "Peak-hour traffic is bursty, not steady. Does that change your recommendation?",
            "What's one metric that would confirm burstiness is the actual cause?",
        ],
    },
]


def build_sessions() -> list[dict]:
    return SESSIONS
