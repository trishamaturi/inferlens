"""
Correlation rules: cross-reference a request's own timing against the
engine's concurrent state (other requests, metric time series) during its
lifetime, to attribute *why* something happened instead of just reporting
that it did -- e.g. not just "this request queued for 1.8s" but "queued
behind a 3000-word request that was already running." This is PLAN.md's
Phase 3 ("correlation rules ... rather than generic threshold alerts"),
scoped to the two rules confirmed buildable from data already on disk
(see README.md §3): large-context starvation and preemption-vs-KV-
pressure. MoE load balance and pipeline-parallel bubbles need hardware
this setup doesn't have; poor continuous-batching packing needs a new
run design (KV blocks binding before the --max-num-seqs cap does) that
hasn't been recorded yet.

Called by build_dashboard.py, which folds the results into the same JSON
payload the dashboard already consumes -- so dashboard_template.html just
renders strings, it doesn't recompute any of this in JS.
"""

# "Big" is relative (a ratio) since prompt sizes vary hugely by workload,
# plus an absolute floor so two small requests a few tokens apart never
# trigger this on noise.
BIG_CONTEXT_TOKEN_RATIO = 5
BIG_CONTEXT_MIN_TOKENS = 500
QUEUE_MS_NOTABLE = 200  # only attribute queue delay worth explaining


def attribute_queue_delay(request: dict, all_requests: list[dict]) -> str | None:
    """Rule: was this request queued behind a much larger *concurrent*
    request? PLAN.md §2's own motivating example ("was this short request
    queued behind a half-million-token context request hogging the
    batch?"), turned into a per-request attribution instead of an
    aggregate observation. Looks at every other request in the run whose
    [submitted_t, completed_t] window overlaps this request's queue
    window, and flags ones with a meaningfully larger prompt."""
    otel = request.get("otel")
    queue_ms = otel["queue_ms"] if otel else None
    if not queue_ms or queue_ms < QUEUE_MS_NOTABLE or not request.get("prompt_tokens"):
        return None

    queue_start = request["submitted_t"]
    queue_end = queue_start + queue_ms / 1000

    culprits = [
        other for other in all_requests
        if other["request_id"] != request["request_id"]
        and other.get("prompt_tokens")
        and other["submitted_t"] < queue_end and other["completed_t"] > queue_start
        and other["prompt_tokens"] >= request["prompt_tokens"] * BIG_CONTEXT_TOKEN_RATIO
        and other["prompt_tokens"] - request["prompt_tokens"] >= BIG_CONTEXT_MIN_TOKENS
    ]
    if not culprits:
        return None

    culprits.sort(key=lambda r: r["prompt_tokens"], reverse=True)
    biggest = culprits[0]
    others = f" (+{len(culprits) - 1} more)" if len(culprits) > 1 else ""
    ratio = biggest["prompt_tokens"] / request["prompt_tokens"]
    return (
        f"Queued {queue_ms:.0f}ms behind a {biggest['prompt_tokens']:.0f}-token request "
        f"({ratio:.0f}x larger context){others}"
    )


def preemption_during_request(request: dict, samples: list[dict]) -> str | None:
    """Rule: did vllm:num_preemptions_total (real ground truth) actually
    increment while this request was in its decode phase? Upgrades the
    inter-token-gap stall heuristic (dashboard_template.html's
    detectStalls, an estimate -- vLLM exposes no per-request preemption
    count) into a confirmed claim when the run-wide counter corroborates
    it during this request's own window."""
    otel = request.get("otel")
    if not otel:
        return None
    queue_end = request["submitted_t"] + otel["queue_ms"] / 1000
    decode_start = queue_end + otel["prefill_ms"] / 1000
    decode_end = decode_start + otel["decode_ms"] / 1000

    prev = None
    increments = 0
    for s in samples:
        val = s.get("vllm:num_preemptions_total")
        if val is None:
            continue
        if prev is not None and val > prev and decode_start <= s["t"] <= decode_end:
            increments += int(val - prev)
        prev = val
    if increments == 0:
        return None
    plural = "s" if increments != 1 else ""
    return (
        f"{increments} real preemption{plural} occurred during this request's decode "
        f"(confirmed via vllm:num_preemptions_total, not just the inter-token-gap estimate)"
    )


def preemption_kv_threshold(samples: list[dict]) -> str | None:
    """Rule: in THIS run, at what KV cache utilization did preemptions
    actually start? Derived per-run rather than a guessed constant, since
    different --num-gpu-blocks-override pool sizes will onset at
    different utilization points."""
    prev_preemptions = None
    onset_kv_pct = None
    peak_kv_pct = 0.0
    for s in samples:
        kv_pct = (s.get("vllm:kv_cache_usage_perc") or 0) * 100
        peak_kv_pct = max(peak_kv_pct, kv_pct)
        preemptions = s.get("vllm:num_preemptions_total")
        if preemptions is None:
            continue
        if prev_preemptions is not None and preemptions > prev_preemptions and onset_kv_pct is None:
            onset_kv_pct = kv_pct
        prev_preemptions = preemptions

    if onset_kv_pct is None:
        return None
    return (
        f"Preemptions began once KV cache usage reached {onset_kv_pct:.1f}% "
        f"(peaked at {peak_kv_pct:.1f}% this run)"
    )


def poor_batching_packing(samples: list[dict], max_num_seqs: int | None) -> str | None:
    """Rule: did concurrency plateau below the *configured* scheduler cap
    while requests were still queueing -- i.e. was KV cache capacity the
    real bottleneck, not --max-num-seqs? Needs the configured cap itself
    (metrics_db.runs.max_num_seqs, recorded per-run since this rule can't
    tell "the scheduler is under-packing" from "there was never enough
    concurrent load to test it" without knowing what the cap actually
    was)."""
    if not max_num_seqs or not samples:
        return None
    peak_running = max((s.get("vllm:num_requests_running") or 0) for s in samples)
    peak_waiting = max((s.get("vllm:num_requests_waiting") or 0) for s in samples)
    if peak_waiting == 0:
        return None  # no queueing pressure at all -- nothing to explain
    if peak_running >= max_num_seqs:
        return None  # scheduler actually reached its configured cap -- that's good packing, not this failure mode
    return (
        f"Concurrency never reached the configured cap (--max-num-seqs {max_num_seqs}) -- "
        f"peaked at {peak_running:.0f} running while {peak_waiting:.0f} requests queued, "
        f"so KV cache capacity was the real bottleneck, not the scheduler's sequence-count cap"
    )


def annotate_request(request: dict, all_requests: list[dict], samples: list[dict]) -> list[str]:
    """All per-request correlation rules for one request, in priority order."""
    return [
        c for c in [
            attribute_queue_delay(request, all_requests),
            preemption_during_request(request, samples),
        ] if c
    ]


def annotate_run(samples: list[dict], max_num_seqs: int | None = None) -> list[str]:
    """All run-level correlation rules."""
    return [
        c for c in [
            preemption_kv_threshold(samples),
            poor_batching_packing(samples, max_num_seqs),
        ] if c
    ]
