# Inference Stack Observability Platform — Design Plan

An open-source, self-hosted observability platform for the **model-serving layer**
of the inference stack (vLLM, TGI, SGLang, Triton, etc.) — curated views and
cross-signal correlation, purpose-built for inference-serving internals rather
than generic metrics/logs/traces.

Status: personal project, pre-code, design phase.

## 1. Landscape check (what already exists)

Raw metric scraping and dashboarding for model servers is already solved and
commodity — not worth rebuilding:

- vLLM ships a `/metrics` Prometheus endpoint (queue depth, KV cache %,
  running/waiting/swapped counts, TTFT, etc.) and native OTel request tracing
  via `--otlp-traces-endpoint`.
  ([vLLM metrics docs](https://docs.vllm.ai/en/stable/design/metrics/))
- Grafana, Parseable, and OpenObserve all ship ready-made "point at vLLM's
  `/metrics`" dashboards.
  ([Grafana vLLM dashboard](https://grafana.com/grafana/dashboards/25263-vllm-metrics/))

## 2. The actual gap / wedge

Metrics and traces are collected but never correlated — and nobody diagnoses
*why* an agentic, multi-session workload behaves the way it does. Today's
inference-serving deployments aren't single-request benchmarks — they're many
concurrent sessions/clients (agent loops, multi-turn chats, batch jobs)
sharing a scheduler, a KV cache, and a fixed pool of GPUs. vLLM exposes both
`kv_cache_usage_perc` and per-request OTel spans, but today you'd view them in
two different tools (a metrics dashboard, a trace viewer) and eyeball the
correlation yourself. Nobody answers questions like:

- "Why did this short request stall — was it queued behind a half-million-
  token context request hogging the batch?"
- "Why did this MoE request spike in latency — was its expert evicted from
  GPU memory and reloaded?"
- "Is our KV cache actually being used well, or are a few hot experts/paths
  dominating while the rest sits idle?"
- "Are tensor-parallel / expert-parallel shards actually balanced, or is one
  GPU a straggler?"
- "Did throughput drop because request mix changed, or because a GPU quietly
  throttled (power/thermal) and tokens/sec dropped independent of load?"

**Positioning:** diagnose *why* a multi-session inference workload is
slow/inefficient, by correlating traces and metrics purpose-built for
model-serving internals.

**Why this matters:** throughput (tokens/sec) and latency (TTFT, ITL) on a
fixed GPU fleet are a direct cost lever — a 10% efficiency gain is a 10%
GPU-hour saving at scale. Failure modes compound quietly (e.g. verbose model
output filling context windows faster than expected, degrading batching and
cache efficiency in ways a plain "requests/sec" dashboard won't show).

## 3. Diagnostic catalog: failure modes this should surface

Parking lot for now — better to derive the actual failure-mode list from what
metrics/traces really get surfaced once something is running, rather than
speculate upfront. Rough areas to revisit against real data:

- Scheduling & batching fairness (e.g. long-context requests starving short
  ones)
- MoE expert routing/eviction/load balance
- Tensor/expert parallelism health (comms overhead, stragglers)
- Cache reuse across multi-turn sessions
- GPU hardware health (throttling, power draw)
- Workload ↔ infra mismatch (agentic loops, autoscaling lag, wrong routing)

## 4. Proposed architecture

```
┌─────────────┐   scrape /metrics (Prometheus)   ┌──────────────┐
│ vLLM/TGI/... │ ───────────────────────────────▶ │              │
│  instance    │   push OTLP traces               │  Collector   │
│              │ ───────────────────────────────▶ │  (per-engine │
└─────────────┘                                   │   adapter)   │
                                                   └──────┬───────┘
                                                          │ normalize to
                                                          │ canonical schema
                                                          ▼
                                          ┌───────────────────────────┐
                                          │  Storage                  │
                                          │  - metrics: embedded TSDB │
                                          │    (e.g. poll every 5s)   │
                                          │  - traces/events: SQLite  │
                                          │    or DuckDB              │
                                          └─────────────┬─────────────┘
                                                         │
                                                         ▼
                                          ┌───────────────────────────┐
                                          │  Correlation engine        │
                                          │  (join traces↔metrics by   │
                                          │   time+request_id, detect  │
                                          │   failure-mode patterns)   │
                                          └─────────────┬─────────────┘
                                                         ▼
                                          ┌───────────────────────────┐
                                          │  API + Web UI              │
                                          │  fleet view / engine       │
                                          │  deep-dive / request       │
                                          │  waterfall                 │
                                          └───────────────────────────┘
```

The ingest pipeline, concretely:
1. **Metrics** land in the TSDB on a fixed poll interval (e.g. every 5s).
2. **Traces** land in the trace store as spans arrive (event-driven, not
   polled).
3. **Correlation** joins the two by time window + `request_id`/session — this
   can also mean metric-to-metric correlation (e.g. power draw vs.
   tokens/sec), not only trace-to-metric.
4. Findings surface as annotations on the waterfall/dashboard views, not just
   raw alerts.

Concrete tech choices (collection protocol, storage engine, backend language,
UI framework) are TBD — better decided once the high-level design above is
settled, not locked in now.

## 5. Phased roadmap

1. **Phase 0 — vLLM-only.** Scrape `/metrics`, ingest OTLP traces, canonical
   schema v0, one "golden signals" dashboard (TTFT, ITL, tok/s, queue depth,
   KV cache %, GPU util via nvidia-smi/DCGM).
2. **Phase 1 — Request waterfall view.** Click a slow request, see its full
   span tree plus the metric timeline around it (e.g. "KV cache hit 96%,
   preempted twice"). Use this to start filling in the diagnostic catalog
   (§3) from real signal.
3. **Phase 2 — Second engine adapter** (TGI or SGLang) to pressure-test the
   canonical schema.
4. **Phase 3 — Correlation rules / alerting** for the failure modes found in
   §3, rather than generic threshold alerts.
5. **Phase 4 — Fleet view.** Multi-replica/multi-model view, cost/token
   overlay.

## Files

Under `poc-vllm-metal/` (run from that directory):

- `run.sh` — activates the vllm-metal venv and runs `basic_test.py`.
- `run_metrics.sh` — activates the venv, records a run with `metrics_test.py`,
  and builds `dashboard.html`. Flags: `[--stress] [--max-num-seqs N]`.
- `run_dashboard.sh` — activates the venv and re-renders `dashboard.html` from
  recorded data via `build_dashboard.py`, without recording. Flag: `[--run-id ID]`.
