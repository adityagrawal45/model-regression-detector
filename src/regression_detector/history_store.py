"""SQLite index of eval-run summaries, for fast trend/drift queries.

`eval_<version>_<timestamp>.json` (written by eval_runner.py) stays the
source of truth for a run's full per-case results - regression.py always
loads baseline/candidate straight from those files. This module only stores
the summary columns needed for trend charts and drift detection (accuracy,
latency, tokens, ...), keyed back to the JSON path for drill-down, so
"give me the last N runs" is a query instead of a directory glob + parse of
every file in it (what `drift.load_eval_history` still does, and continues
to support as a fallback).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from regression_detector.eval_runner import EvalReport

DEFAULT_DB_PATH = "reports/history.db"

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_version          TEXT NOT NULL,
    model                   TEXT NOT NULL,
    dataset_version         TEXT,
    ran_at                  TEXT NOT NULL,
    total                   INTEGER NOT NULL,
    category_accuracy       REAL NOT NULL,
    avg_summary_score       REAL NOT NULL,
    avg_summary_judge_score REAL,
    avg_latency_ms          REAL,
    total_tokens            INTEGER,
    report_path             TEXT NOT NULL UNIQUE,
    recorded_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_ran_at ON eval_runs(ran_at);
"""

_SELECT_COLUMNS = (
    "prompt_version, model, dataset_version, ran_at, total, category_accuracy, "
    "avg_summary_score, avg_summary_judge_score, avg_latency_ms, total_tokens, report_path"
)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the DB (creating parent dirs + table if needed). Caller closes it."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA_DDL)
    conn.commit()
    return conn


def record_run(report: EvalReport, report_path: str | Path, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Insert one summary row for `report`.

    Idempotent on `report_path` (normalized to an absolute path): recording
    the same report twice - e.g. a retried CI step - is a no-op rather than
    a duplicate row.
    """
    resolved_path = str(Path(report_path).resolve())
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT OR IGNORE INTO eval_runs ({_SELECT_COLUMNS}, recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                report.prompt_version,
                report.model,
                report.dataset_version,
                report.ran_at.isoformat(),
                report.total,
                report.category_accuracy,
                report.avg_summary_score,
                report.avg_summary_judge_score,
                report.avg_latency_ms,
                report.total_tokens,
                resolved_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def _row_to_report(row: tuple) -> EvalReport:
    (
        prompt_version,
        model,
        dataset_version,
        ran_at,
        total,
        category_accuracy,
        avg_summary_score,
        avg_summary_judge_score,
        avg_latency_ms,
        total_tokens,
        _report_path,
    ) = row
    return EvalReport(
        prompt_version=prompt_version,
        model=model,
        dataset_version=dataset_version,
        ran_at=datetime.fromisoformat(ran_at),
        total=total,
        category_accuracy=category_accuracy,
        avg_summary_score=avg_summary_score,
        avg_summary_judge_score=avg_summary_judge_score,
        avg_latency_ms=avg_latency_ms,
        total_tokens=total_tokens,
        results=[],
    )


def load_history(db_path: str | Path = DEFAULT_DB_PATH, limit: int | None = None) -> list[EvalReport]:
    """Load stored runs, oldest first, reconstructed with `results=[]`.

    Returns `[]` if the DB file doesn't exist yet - mirrors
    `drift.load_eval_history`'s behavior on a missing history directory, so
    this is a drop-in replacement at that call site.
    """
    if not Path(db_path).exists():
        return []
    query = f"SELECT {_SELECT_COLUMNS} FROM eval_runs ORDER BY ran_at ASC"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_report(row) for row in rows]


def load_recent_run_rows(n: int, db_path: str | Path = DEFAULT_DB_PATH) -> list[tuple[EvalReport, str]]:
    """The `n` most recent runs, most-recent-first, paired with their report path.

    Kept separate from `load_history` (rather than adding a `report_path`
    field to `EvalReport`, which has no such field) since only the dashboard's
    runs table needs the drill-down link.
    """
    if not Path(db_path).exists():
        return []
    query = f"SELECT {_SELECT_COLUMNS} FROM eval_runs ORDER BY ran_at DESC LIMIT ?"
    with _connect(db_path) as conn:
        rows = conn.execute(query, (n,)).fetchall()
    return [(_row_to_report(row), row[-1]) for row in rows]


__all__ = [
    "DEFAULT_DB_PATH",
    "record_run",
    "load_history",
    "load_recent_run_rows",
]
