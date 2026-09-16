"""
Pulls a run recorded by metrics_test.py out of vllm_metrics.db and stamps
it into dashboard_template.html, producing a self-contained dashboard.html
with a per-query dropdown plus the run-wide golden-signal charts.

Run inside the vllm-metal venv:
    source ~/.venv-vllm-metal/bin/activate
    python build_dashboard.py [--run-id RUN_ID] [--out dashboard.html]

Or via the helper script from this directory:
    ./run_metrics.sh build_dashboard.py
"""

import argparse
import json
from pathlib import Path

import metrics_db

TEMPLATE_PATH = Path(__file__).parent / "dashboard_template.html"


def pivot_samples(sample_rows: list[tuple[float, str, float]]) -> list[dict]:
    by_t: dict[float, dict] = {}
    for t, name, value in sample_rows:
        by_t.setdefault(t, {"t": t})[name] = value
    return [by_t[t] for t in sorted(by_t)]


def build_payload(run_id: str | None, db_path: Path) -> dict:
    conn = metrics_db.connect(db_path)
    try:
        run_id = run_id or metrics_db.latest_run_id(conn)
        if run_id is None:
            raise SystemExit(f"No runs recorded yet in {db_path} -- run metrics_test.py first.")
        run, request_rows, sample_rows, tokens_by_request = metrics_db.fetch_run(conn, run_id)
    finally:
        conn.close()

    _, model, started_at, duration_s = run
    requests_ = [
        {
            "request_id": r[0], "kind": r[1], "label": r[2], "submitted_t": r[3],
            "first_token_t": r[4], "completed_t": r[5], "prompt_tokens": r[6],
            "completion_tokens": r[7], "ttft_ms": r[8], "itl_ms": r[9], "e2e_ms": r[10],
            "token_times": tokens_by_request.get(r[0], []),
        }
        for r in request_rows
    ]
    return {
        "run": {"run_id": run_id, "model": model, "started_at": started_at, "duration_s": duration_s},
        "samples": pivot_samples(sample_rows),
        "requests": requests_,
    }


def render_dashboard(run_id: str | None, db_path: Path, out_path: Path) -> dict:
    """Build dashboard.html for a run and return its payload dict (used by
    metrics_test.py to render immediately after recording, and by this
    script's own CLI to (re)render any run on demand)."""
    payload = build_payload(run_id, db_path)
    template = TEMPLATE_PATH.read_text()
    html = template.replace("__RUN_DATA__", json.dumps(payload))
    out_path.write_text(html)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Run to render (defaults to the most recent).")
    parser.add_argument("--db", default=str(metrics_db.DB_PATH), help="Path to the SQLite DB.")
    parser.add_argument("--out", default=str(Path(__file__).parent / "dashboard.html"), help="Output HTML file.")
    args = parser.parse_args()

    payload = render_dashboard(args.run_id, Path(args.db), Path(args.out))
    print(
        f"Wrote dashboard for run {payload['run']['run_id']} "
        f"({len(payload['samples'])} samples, {len(payload['requests'])} requests) to {args.out}"
    )


if __name__ == "__main__":
    main()
