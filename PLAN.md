# Inference Stack Observability Platform — Design Plan

An open-source, self-hosted observability platform for the **model-serving layer**
of the inference stack (vLLM, TGI, SGLang, Triton, etc.) — Datadog-style curated
views and cross-signal correlation, purpose-built for inference-serving internals
rather than generic metrics/logs/traces.

Status: personal project, pre-code, design phase.

## 1. Landscape check (what already exists)

Raw metric scraping for model servers is a **solved, crowded problem** — don't
rebuild it:

- vLLM ships a `/metrics` Prometheus endpoint: queue depth, `kv_cache_usage_perc`,
  running/waiting/swapped request counts, TTFT, etc.
  ([vLLM metrics docs](https://docs.vllm.ai/en/stable/design/metrics/))
- vLLM also ships native OpenTelemetry **request tracing** via
  `--otlp-traces-endpoint` — per-request spans through the scheduler/prefill/decode
  path. ([vLLM OTel docs](https://docs.vllm.ai/en/stable/examples/online_serving/opentelemetry/))
- Grafana, Parseable, and OpenObserve all ship off-the-shelf "point me at vLLM's
  `/metrics`" dashboards.
  ([Grafana vLLM dashboard](https://grafana.com/grafana/dashboards/25263-vllm-metrics/),
  [Parseable + OTel](https://www.parseable.com/blog/vllm-inference-metrics-otel))

## 2. The actual gap / wedge

Two things nobody has shipped as a real open-source product yet:

1. **Metrics and traces are collected but never correlated.** vLLM exposes both
   `kv_cache_usage_perc` and per-request OTel spans, but today you'd view them in
   two different tools (Grafana for one, Jaeger/Tempo for the other) and eyeball
   the correlation yourself — "did this request's latency spike because KV cache
   was under pressure and it got preempted?" This is exactly the kind of stitching
   Datadog does natively (APM trace ↔ infra metric ↔ host), and nobody has built
   it for the inference-serving layer specifically.
2. **No cross-engine canonical schema.** vLLM, TGI, SGLang, and Triton each
   name/shape their metrics differently. There's no "integration" model — the
   thing that makes Datadog Datadog — where you get the same golden-signal
   dashboard regardless of which serving engine sits underneath.

**Positioning:** not "generic observability platform," but "Datadog-style curated
views + trace/metric correlation, purpose-built for model-serving internals,
normalized across engines."

## 3. Scope decisions (already made)

- **Primary layer:** model-serving internals (vLLM/TGI/SGLang/Triton), not
  hardware-only or app/prompt-level LLM tracing.
- **v1 scale:** single box, self-hosted. Point it at one machine running
  vLLM/etc. with 1–8 GPUs. No multi-tenant/SaaS concerns yet.
- **Motivation:** personal project / portfolio-shaped, but aimed at a real gap,
  not purely an exercise.

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
                                          │  - traces/events: SQLite  │
                                          │    or DuckDB              │
                                          └─────────────┬─────────────┘
                                                         │
                                                         ▼
                                          ┌───────────────────────────┐
                                          │  Correlation engine        │
                                          │  (join traces↔metrics by   │
                                          │   time+request_id, detect  │
                                          │   preemption/cache-pressure│
                                          │   patterns)                │
                                          └─────────────┬─────────────┘
                                                         ▼
                                          ┌───────────────────────────┐
                                          │  API + Web UI              │
                                          │  fleet view / engine       │
                                          │  deep-dive / request       │
                                          │  waterfall                 │
                                          └───────────────────────────┘
```

### Key design decisions

- **Collection protocol: OpenTelemetry.** vLLM already speaks OTLP for traces,
  and Prometheus-format scraping is trivial to wrap in an OTel collector
  pipeline. This also means anyone running a real OTel collector can feed data
  in without a bespoke agent.
- **Per-engine adapter model.** A thin plugin per serving engine maps its
  metric names/labels onto a canonical schema (see §5). Start with vLLM only;
  add TGI/SGLang once the schema proves out, to confirm it's actually
  engine-agnostic and not secretly vLLM-shaped.
- **Storage: embeddable, zero external services.** SQLite or DuckDB for both
  time-series metrics and trace spans (DuckDB especially, since correlation
  queries are analytical joins over time windows). No separate
  Prometheus + Jaeger + Postgres stack to run for a single-box install.
- **Backend/agent language: Go.** Single static binary, cheap to run alongside
  a GPU-hogging inference process, solid OTel SDK/collector libraries. Python
  is tempting since the ML ecosystem lives there, but the agent shouldn't fight
  the same box's CUDA/torch dependency hell.
- **UI: React + a charting library.** Nothing exotic — the differentiation is
  in the correlation logic and curated views, not the frontend framework.

## 5. Canonical metric schema (starter sketch, vLLM-derived)

To be validated/adjusted once a second engine adapter is built.

| Canonical name | Description | vLLM source |
|---|---|---|
| `inference.queue_depth` | Requests waiting to be scheduled | `vllm:num_requests_waiting` |
| `inference.running_requests` | Requests currently in a batch | `vllm:num_requests_running` |
| `inference.swapped_requests` | Requests swapped out under memory pressure | `vllm:num_requests_swapped` |
| `inference.kv_cache_util_pct` | KV cache block utilization | `vllm:kv_cache_usage_perc` |
| `inference.ttft_ms` | Time to first token | vLLM TTFT histogram |
| `inference.itl_ms` | Inter-token latency | vLLM ITL histogram |
| `inference.tokens_per_sec` | Throughput | derived from token counters |
| `inference.preemption_count` | Preemption events (cache-pressure signal) | vLLM preemption counter |

Request-level trace attributes to preserve: `request_id`, prefill/decode span
boundaries, preemption events tied to a specific request.

## 6. Phased roadmap

1. **Phase 0 — vLLM-only.** Scrape `/metrics`, ingest OTLP traces, canonical
   schema v0, one "golden signals" dashboard (TTFT, ITL, tok/s, queue depth,
   KV cache %, GPU util via nvidia-smi/DCGM).
2. **Phase 1 — Request waterfall view.** Click a slow request, see its full
   span tree plus the metric timeline around it (e.g. "KV cache hit 96%,
   preempted twice").
3. **Phase 2 — Second engine adapter** (TGI or SGLang) to pressure-test the
   canonical schema.
4. **Phase 3 — Correlation rules / alerting** tuned to inference failure modes
   (cache-pressure-induced preemption storms, throughput cliffs, queue
   starvation) rather than generic threshold alerts.
5. **Phase 4 — Fleet view.** Multi-replica/multi-model view, cost/token
   overlay.

## 7. Open questions before scaffolding

- Start from a bare repo, or fork/embed an existing OTel collector rather than
  writing the scrape+ingest path from scratch?
- Is there a GPU box with vLLM already running to develop against, or should
  Phase 0 include a docker-compose that spins up a small vLLM instance for dev?
- Repo/package naming, license (likely Apache-2.0 or MIT given the OSS goal).
