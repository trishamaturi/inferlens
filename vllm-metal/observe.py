"""
Observes an already-running vLLM instance -- for supplementing your own
vLLM run with more information, instead of starting a server and firing
synthetic traffic like the other scripts here.

Requires vllm serve to have been started with --otlp-traces-endpoint
pointing at this script's OTel receiver (needs a restart if not already
on). Without it: metrics only, no waterfall or per-request correlation.

Observed requests have no prompt text and no stall heuristic -- both need
being the streaming client, which observed traffic isn't.

Run inside the vllm-metal venv:
    python observe.py --host HOST --port PORT [--otlp-port 4318]
                       [--duration SECONDS] [--model NAME]

Or: ./run_observe.sh --host HOST --port PORT [...]
"""

import argparse
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

import metrics_db
from build_dashboard import render_dashboard
from metrics_test import OTEL_FLUSH_GRACE_SEC, POLL_INTERVAL_SEC, MetricsPoller, span_to_request
from otel_receiver import OtelSpanReceiver


def fetch_model_name(base_url: str) -> str:
    """GET /v1/models is standard on any OpenAI-compatible server -- lets
    us skip asking the user to type the model name themselves."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=3)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        return data[0]["id"] if data else "unknown"
    except requests.exceptions.RequestException:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="Host of the already-running vLLM instance.")
    parser.add_argument("--port", type=int, required=True, help="Port of the already-running vLLM instance.")
    parser.add_argument(
        "--otlp-host", default="127.0.0.1",
        help="Host this script's OTel receiver listens on -- point the target's --otlp-traces-endpoint here.",
    )
    parser.add_argument("--otlp-port", type=int, default=4318, help="Port for the OTel receiver.")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Seconds to observe. Default: run until Ctrl+C.",
    )
    parser.add_argument("--model", default=None, help="Model name to record. Default: auto-detected via /v1/models.")
    parser.add_argument("--out", default=str(Path(__file__).parent / "dashboard.html"), help="Output HTML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    otel_receiver = OtelSpanReceiver(args.otlp_host, args.otlp_port)
    otel_receiver.start()
    poller = MetricsPoller(POLL_INTERVAL_SEC, base_url=base_url)

    # Captured before anything that makes a network call (fetch_model_name
    # below) -- otherwise that call's latency delays this reference past
    # when real traffic could already be arriving, producing a negative
    # submitted_t for any request that started in the gap.
    run_start_monotonic = time.monotonic()
    run_start_wall_ns = time.time_ns()
    poller.start(run_start_monotonic)

    model = args.model or fetch_model_name(base_url)
    print(f"Observing {base_url} (model={model}). OTel receiver listening at {otel_receiver.endpoint}.")
    print(
        "If the target wasn't started with --otlp-traces-endpoint pointing here, metrics are still "
        "recorded but there will be no per-request waterfall (see this script's own docstring)."
    )
    try:
        if args.duration:
            print(f"Observing for {args.duration:.0f}s ...")
            time.sleep(args.duration)
        else:
            print("Observing until Ctrl+C ...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        poller.stop()
        # Give vLLM's BatchSpanProcessor a moment to flush any spans still
        # in flight -- same reasoning as record_run's matching sleep.
        time.sleep(OTEL_FLUSH_GRACE_SEC)
        otel_receiver.stop()

    results = []
    for span in otel_receiver.spans:
        if span["name"] != "llm_request":
            continue
        timing = span_to_request(span, run_start_wall_ns)
        server_request_id = timing["server_request_id"] or "unknown"
        results.append({
            "request_id": str(uuid.uuid4()),
            "kind": "observed",
            "label": f"request {server_request_id[-8:]}",
            **timing,
        })

    duration_s = max(
        [t for t, _ in poller.samples] + [r["completed_t"] for r in results],
        default=0.0,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    conn = metrics_db.connect()
    with conn:
        metrics_db.insert_run(conn, run_id, model, datetime.now(timezone.utc).isoformat(), duration_s)
        for result in results:
            metrics_db.insert_request(conn, run_id, result)
            metrics_db.insert_request_tokens(conn, result["request_id"], result["token_times"])
        metrics_db.insert_samples(conn, run_id, poller.samples)
        metrics_db.insert_otel_spans(conn, run_id, otel_receiver.spans)
    conn.close()

    print(f"\nCaptured {len(results)} requests, {len(poller.samples)} metric samples, over {duration_s:.1f}s.")
    if not results:
        print(
            "No llm_request spans captured -- either no traffic arrived, or the target wasn't "
            "started with --otlp-traces-endpoint pointing at this receiver."
        )
    print(f"Wrote run {run_id} to {metrics_db.DB_PATH}")

    out_path = Path(args.out)
    render_dashboard(run_id, metrics_db.DB_PATH, out_path)
    print(f"Wrote dashboard to {out_path}")


if __name__ == "__main__":
    main()
