"""
Phase 0, step 2: run vLLM as an OpenAI-compatible server, fire every prompt
from basic_test.py at it *concurrently* as its own streamed request, and
record what happened -- both the engine-wide Prometheus /metrics timeline
and the true per-request timing (TTFT/ITL/E2E measured client-side from the
SSE stream, not approximated from engine aggregates) -- into vllm_metrics.db
for build_dashboard.py to render.

Also renders the run straight to dashboard.html afterwards (see
build_dashboard.py) so one command gets you from a cold server to a
viewable page.

Run inside the vllm-metal venv:
    source ~/.venv-vllm-metal/bin/activate
    python metrics_test.py

Or via the helper script from this directory:
    ./run_metrics.sh
"""

import json
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

import metrics_db
from basic_test import CONVERSATIONS, MODEL, RAW_PROMPTS
from build_dashboard import render_dashboard

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
STARTUP_TIMEOUT_S = 300
POLL_INTERVAL_S = 0.2

# Golden-signal metrics (see PLAN.md phase 0): queue depth, KV cache %,
# token throughput, and the latency histograms behind TTFT/ITL/e2e.
WANTED_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
}


def parse_metrics(text: str) -> dict[str, float]:
    """Pull the wanted gauge/counter/histogram sum+count values out of a
    Prometheus text-exposition payload, ignoring bucket lines and labels."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        if brace == -1:
            name, _, value = line.partition(" ")
        else:
            name = line[:brace]
            close = line.find("}", brace)
            value = line[close + 1 :].strip()
        if name in WANTED_METRICS:
            try:
                values[name] = float(value)
            except ValueError:
                pass
    return values


class MetricsPoller:
    """Scrapes /metrics on a background thread until stopped."""

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[tuple[float, dict]] = []
        self._start_time = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                resp = requests.get(f"{BASE_URL}/metrics", timeout=2)
                resp.raise_for_status()
                values = parse_metrics(resp.text)
                t = round(time.monotonic() - self._start_time, 3)
                self.samples.append((t, values))
            except requests.exceptions.RequestException:
                pass
            self._stop.wait(self._interval_s)

    def start(self, start_time: float) -> None:
        self._start_time = start_time
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def wait_for_server(proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited early with code {proc.returncode}")
        try:
            if requests.get(f"{BASE_URL}/health", timeout=1).ok:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vllm serve did not become healthy within {STARTUP_TIMEOUT_S}s")


def truncate(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def stream_request(path: str, payload: dict, start_time: float) -> dict:
    """POST a streamed OpenAI-compatible request, recording the arrival
    time of *every* token (not just the first), since that's what lets the
    dashboard draw this specific request's own latency/throughput curve
    instead of a single averaged number."""
    submitted_t = time.monotonic() - start_time
    token_times: list[float] = []
    prompt_tokens = completion_tokens = None
    is_chat = "chat" in path

    with requests.post(f"{BASE_URL}{path}", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices:
                delta_text = (
                    (choices[0].get("delta") or {}).get("content") if is_chat else choices[0].get("text")
                )
                if delta_text:
                    token_times.append(time.monotonic() - start_time)
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"]["prompt_tokens"]
                completion_tokens = chunk["usage"]["completion_tokens"]

    completed_t = time.monotonic() - start_time
    first_token_t = token_times[0] if token_times else None
    ttft_ms = (first_token_t - submitted_t) * 1000 if first_token_t is not None else None
    e2e_ms = (completed_t - submitted_t) * 1000
    itl_ms = (
        (completed_t - first_token_t) * 1000 / (completion_tokens - 1)
        if first_token_t is not None and completion_tokens and completion_tokens > 1
        else None
    )
    return {
        "submitted_t": submitted_t,
        "first_token_t": first_token_t,
        "completed_t": completed_t,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "e2e_ms": e2e_ms,
        "token_times": token_times,
    }


def completion_job(prompt: str) -> tuple[str, str, str, dict]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return "completion", truncate(prompt), "/v1/completions", payload


def chat_job(messages: list[dict]) -> tuple[str, str, str, dict]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    extra_turns = len(messages) - 1
    label = truncate(first_user) + (f" (+{extra_turns} more turn{'s' if extra_turns != 1 else ''})" if extra_turns else "")
    return "chat", label, "/v1/chat/completions", payload


def run_all_queries(start_time: float) -> list[dict]:
    """Fires every prompt/conversation as its own concurrent streamed
    request, so the server actually sees the multi-session load PLAN.md is
    about, and each query gets its own real (not engine-averaged) timing."""
    jobs = [completion_job(p) for p in RAW_PROMPTS] + [chat_job(c) for c in CONVERSATIONS]
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(stream_request, path, payload, start_time): (kind, label)
            for kind, label, path, payload in jobs
        }
        for future in as_completed(futures):
            kind, label = futures[future]
            timing = future.result()
            results.append({"request_id": str(uuid.uuid4()), "kind": kind, "label": label, **timing})
            ttft = f"ttft={timing['ttft_ms']:.0f}ms " if timing["ttft_ms"] is not None else ""
            print(f"  [{kind}] {label!r} -- {ttft}e2e={timing['e2e_ms']:.0f}ms")
    return results


def main() -> None:
    proc = subprocess.Popen(["vllm", "serve", MODEL, "--host", HOST, "--port", str(PORT)])
    poller = MetricsPoller(POLL_INTERVAL_S)
    start_time = time.monotonic()
    try:
        print(f"Waiting for vllm serve to come up on {BASE_URL} ...")
        wait_for_server(proc)
        print("Server is healthy. Starting metrics poller and firing all queries concurrently.")
        poller.start(start_time)

        results = run_all_queries(start_time)

        # A few extra samples so the tail of the run (post-request settling)
        # shows up in the chart too.
        time.sleep(POLL_INTERVAL_S * 4)
    finally:
        poller.stop()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    duration_s = max((t for t, _ in poller.samples), default=0.0)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    conn = metrics_db.connect()
    with conn:
        metrics_db.insert_run(conn, run_id, MODEL, datetime.now(timezone.utc).isoformat(), duration_s)
        for result in results:
            metrics_db.insert_request(conn, run_id, result)
            metrics_db.insert_request_tokens(conn, result["request_id"], result["token_times"])
        metrics_db.insert_samples(conn, run_id, poller.samples)
    conn.close()

    print(
        f"\nWrote run {run_id} ({len(poller.samples)} metric samples, {len(results)} requests) "
        f"to {metrics_db.DB_PATH}"
    )

    dashboard_path = Path(__file__).parent / "dashboard.html"
    render_dashboard(run_id, metrics_db.DB_PATH, dashboard_path)
    print(f"Wrote dashboard to {dashboard_path}")


if __name__ == "__main__":
    main()
