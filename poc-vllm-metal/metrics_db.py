"""
SQLite storage for vLLM run data: one row per /metrics scrape sample
(long format: run/t/metric_name/value) and one row per query fired at the
server, with real per-request timing measured client-side over streaming
responses (see metrics_test.py).

Multiple runs accumulate in the same DB file (vllm_metrics.db, gitignored)
so build_dashboard.py can render whichever run you want.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "vllm_metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_s REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    submitted_t REAL NOT NULL,
    first_token_t REAL,
    completed_t REAL NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    ttft_ms REAL,
    itl_ms REAL,
    e2e_ms REAL
);

CREATE TABLE IF NOT EXISTS metric_samples (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    t REAL NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL
);

-- Every token's arrival time (seconds since run start), per request -- the
-- only way to draw a real per-query latency/throughput curve, since a
-- single ttft_ms/itl_ms/e2e_ms triple is just three numbers, not a series.
CREATE TABLE IF NOT EXISTS request_tokens (
    request_id TEXT NOT NULL REFERENCES requests(request_id),
    idx INTEGER NOT NULL,
    t REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS idx_samples_run_t ON metric_samples(run_id, t);
CREATE INDEX IF NOT EXISTS idx_tokens_request ON request_tokens(request_id);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_run(conn: sqlite3.Connection, run_id: str, model: str, started_at: str, duration_s: float) -> None:
    conn.execute(
        "INSERT INTO runs (run_id, model, started_at, duration_s) VALUES (?, ?, ?, ?)",
        (run_id, model, started_at, duration_s),
    )


def insert_request(conn: sqlite3.Connection, run_id: str, req: dict) -> None:
    conn.execute(
        """INSERT INTO requests
           (request_id, run_id, kind, label, submitted_t, first_token_t, completed_t,
            prompt_tokens, completion_tokens, ttft_ms, itl_ms, e2e_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            req["request_id"], run_id, req["kind"], req["label"], req["submitted_t"],
            req["first_token_t"], req["completed_t"], req["prompt_tokens"], req["completion_tokens"],
            req["ttft_ms"], req["itl_ms"], req["e2e_ms"],
        ),
    )


def insert_samples(conn: sqlite3.Connection, run_id: str, samples: list[tuple[float, dict]]) -> None:
    rows = [(run_id, t, name, value) for t, metrics in samples for name, value in metrics.items()]
    conn.executemany("INSERT INTO metric_samples (run_id, t, metric_name, value) VALUES (?, ?, ?, ?)", rows)


def insert_request_tokens(conn: sqlite3.Connection, request_id: str, token_times: list[float]) -> None:
    rows = [(request_id, idx, t) for idx, t in enumerate(token_times, start=1)]
    conn.executemany("INSERT INTO request_tokens (request_id, idx, t) VALUES (?, ?, ?)", rows)


def latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def fetch_run(conn: sqlite3.Connection, run_id: str):
    run = conn.execute(
        "SELECT run_id, model, started_at, duration_s FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise ValueError(f"No run {run_id!r} found")

    request_rows = conn.execute(
        """SELECT request_id, kind, label, submitted_t, first_token_t, completed_t,
                  prompt_tokens, completion_tokens, ttft_ms, itl_ms, e2e_ms
           FROM requests WHERE run_id = ? ORDER BY submitted_t""",
        (run_id,),
    ).fetchall()

    sample_rows = conn.execute(
        "SELECT t, metric_name, value FROM metric_samples WHERE run_id = ? ORDER BY t",
        (run_id,),
    ).fetchall()

    token_rows = conn.execute(
        """SELECT rt.request_id, rt.t FROM request_tokens rt
           JOIN requests r ON r.request_id = rt.request_id
           WHERE r.run_id = ? ORDER BY rt.request_id, rt.idx""",
        (run_id,),
    ).fetchall()
    tokens_by_request: dict[str, list[float]] = {}
    for request_id, t in token_rows:
        tokens_by_request.setdefault(request_id, []).append(t)

    return run, request_rows, sample_rows, tokens_by_request
