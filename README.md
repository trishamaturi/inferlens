# Inference Stack Observability Platform — Design Plan

An open-source, self-hosted observability platform for the **model-serving layer**
of the inference stack (vLLM, TGI, SGLang, Triton, etc.) — curated views and
cross-signal correlation, purpose-built for inference-serving internals rather
than generic metrics/logs/traces.

Status: personal project. Phase 0 done, Phase 1 mostly done, Phase 3 started
(ahead of Phase 2, which is blocked on GPU access). See §5.

## 1. Landscape check (what already exists)

Raw metric scraping/dashboarding for model servers is commodity — not worth
rebuilding:

- vLLM ships a `/metrics` Prometheus endpoint and native OTel tracing via
  `--otlp-traces-endpoint`.
  ([vLLM metrics docs](https://docs.vllm.ai/en/stable/design/metrics/))
- Grafana, Parseable, and OpenObserve all ship ready-made vLLM dashboards.

## 2. The actual gap / wedge

Metrics and traces are collected but never correlated — nobody diagnoses *why*
a multi-session workload behaves the way it does. Concurrent sessions share a
scheduler, a KV cache, and a fixed GPU pool; vLLM exposes the signals, but
today you'd eyeball the correlation yourself across two separate tools.
Example questions nobody answers:

- Was this short request queued behind a huge-context request hogging the batch?
- Is KV cache well-utilized, or are a few hot paths dominating while the rest idles?
- Did throughput drop because of request mix, or a throttling GPU?

**Positioning:** diagnose *why* a multi-session inference workload is
slow/inefficient, by correlating traces and metrics purpose-built for
model-serving internals — not another raw-metrics dashboard.

## 3. Diagnostic catalog

Four of seven failure modes are confirmed against real recorded runs
(`vllm-metal/`, Qwen3-0.6B on Apple Silicon/Metal):

- **Scheduling & batching fairness** ✅ — short requests stagger in admission
  waves behind concurrent load, purely from queueing, not prefill cost.
- **Preemption** ✅ — only reproducible once KV cache is deliberately shrunk
  (`--num-gpu-blocks-override`); the default pool is oversized for this model.
- **Cache reuse across multi-turn sessions** ✅ — needs sequential firing to
  observe (concurrent firing races every prefill before a block is cached);
  hit rate climbs from 0% cold to ~99% on verbatim repeats.
- **Workload ↔ infra mismatch (agentic loops)** ✅ — concurrent sessions of
  *dependent* sequential turns reproduce real cross-agent queueing and show
  organic (not synthetic) cache reuse from growing context.
- MoE load balance, tensor/expert parallelism, GPU hardware health — not
  testable on this single-GPU, dense-model, Metal setup.

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

Concrete tech choices (collection protocol, storage engine, backend language,
UI framework) are TBD — better decided once the high-level design above is
settled, not locked in now.

## 5. Phased roadmap

1. **Phase 0 — vLLM-only.** ✅ Done. `/metrics` scraping, OTLP trace ingest,
   golden-signals dashboard. Gaps: no GPU-util metric (no Metal-native
   nvidia-smi/DCGM equivalent); storage schema is vLLM-shaped only so far.
2. **Phase 1 — Request waterfall view.** 🚧 Mostly done. Per-request
   queue/prefill/decode waterfall, preemption-stall heuristic, prefix-cache
   hit rate. 4 of 7 §3 catalog entries confirmed; the rest need hardware this
   setup doesn't have.
3. **Phase 2 — Second engine adapter** (TGI/SGLang). ⏸ Blocked — both are
   CUDA-only; this box only runs vLLM via its Metal backend. Needs real GPU
   access first (e.g. a university compute cluster or hourly rentals).
4. **Phase 3 — Correlation rules.** 🚧 Started out of order (ahead of the
   blocked Phase 2). `correlate.py` has 3 rules: large-context starvation
   attribution, preemption confirmed against a KV threshold, and poor
   batching packing (KV-bound vs. seq-count-bound concurrency). 2 more rules
   (MoE, pipeline bubbles) stay deferred — need hardware this setup doesn't
   have.
5. **Phase 4 — Fleet view.** Not started. Multi-replica/multi-model view,
   cost/token overlay.

## 6. Observing an already-running instance

Every script below starts its own `vllm serve` and fires synthetic traffic.
`observe.py` instead points at an instance you already have running with real
traffic — no server start, no synthetic requests:

```
./run_observe.sh --host <host> --port <port> [--otlp-port 4318] [--duration SECONDS]
```

Requires the target to have been started with `--otlp-traces-endpoint`
pointing at `observe.py`'s receiver (can't attach after the fact — needs a
restart). Without it: metrics-only, no waterfall. Observed requests also have
no prompt text and no stall heuristic (both need being the streaming client).

## Files

Under `vllm-metal/` (run from that directory):

- `basic_test.py` — offline smoke test (no server): raw-text and chat
  completions via `.generate()`/`.chat()`. Defines `MODEL`, `RAW_PROMPTS`,
  `CONVERSATIONS`, reused by the other scripts.
- `metrics_test.py` — starts `vllm serve`, fires requests, records
  `/metrics` samples, per-request timing, and OTel spans into
  `vllm_metrics.db`, then renders `dashboard.html`. `record_run()` is the
  shared recorder used by every script below.
- `stress_test.py` — scheduler-overloading workload (short requests + long
  "hogs") for real queueing/KV-pressure. `--num-gpu-blocks-override` forces
  real preemption.
- `cache_test.py` / `cache_workload.py` — fires a shared prompt prefix
  sequentially to get real prefix-cache hit-rate signal.
- `agentic_test.py` / `agentic_workload.py` — concurrent sessions, each a
  sequential chain of chat turns depending on the model's real prior reply.
- `correlate.py` — Phase 3 correlation rules, folded into the dashboard
  payload. See §5.
- `observe.py` — observes an already-running instance instead of starting
  one. See §6.
- `run.sh` / `run_metrics.sh` / `run_stress.sh` / `run_cache.sh` /
  `run_agentic.sh` / `run_observe.sh` — venv-activating wrappers for the
  scripts above; see each script's `--help`.
- `run_dashboard.sh` — re-renders `dashboard.html` from recorded data
  without recording. Flag: `[--run-id ID]`.
