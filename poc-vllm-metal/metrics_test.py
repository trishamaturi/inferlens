"""
Phase 0, step 2: run vLLM as an OpenAI-compatible server and scrape its
Prometheus /metrics endpoint -- the piece basic_test.py's offline-batch mode
doesn't exercise (no server process there, so nothing to scrape).

Starts `vllm serve`, waits for it to report healthy, fires the same
prompts/conversations from basic_test.py at it over HTTP, then dumps
/metrics so we can see what real requests do to queue depth, KV cache
usage, etc.

Run inside the vllm-metal venv:
    source ~/.venv-vllm-metal/bin/activate
    python metrics_test.py

Or via the helper script from this directory:
    ./run.sh metrics_test.py
"""

import subprocess
import time

import requests

from basic_test import CONVERSATIONS, MODEL, RAW_PROMPTS

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
STARTUP_TIMEOUT_S = 300


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


def run_completions() -> None:
    print("\n### Batched raw completions (/v1/completions) ###")
    resp = requests.post(
        f"{BASE_URL}/v1/completions",
        json={"model": MODEL, "prompt": RAW_PROMPTS, "temperature": 0.7, "max_tokens": 64},
        timeout=120,
    )
    resp.raise_for_status()
    for prompt, choice in zip(RAW_PROMPTS, resp.json()["choices"]):
        print("=" * 60)
        print(f"Prompt:     {prompt!r}")
        print(f"Completion: {choice['text']!r}")


def run_chats() -> None:
    print("\n### Conversations (/v1/chat/completions) ###")
    for messages in CONVERSATIONS:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        resp.raise_for_status()
        print("=" * 60)
        print(f"Prompt:     {messages!r}")
        print(f"Completion: {resp.json()['choices'][0]['message']['content']!r}")


def dump_metrics() -> None:
    print("\n### /metrics (Prometheus text format) ###")
    resp = requests.get(f"{BASE_URL}/metrics", timeout=10)
    resp.raise_for_status()
    print(resp.text)


def main() -> None:
    proc = subprocess.Popen(["vllm", "serve", MODEL, "--host", HOST, "--port", str(PORT)])
    try:
        print(f"Waiting for vllm serve to come up on {BASE_URL} ...")
        wait_for_server(proc)
        print("Server is healthy.")
        run_completions()
        run_chats()
        dump_metrics()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
