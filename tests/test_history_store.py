from datetime import datetime, timedelta, timezone
from pathlib import Path

from regression_detector.eval_runner import EvalReport
from regression_detector.history_store import load_history, load_recent_run_rows, record_run

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _report(i: int, accuracy: float = 0.9) -> EvalReport:
    return EvalReport(
        prompt_version=f"v{i}",
        model="test-model",
        dataset_version="ds-v1",
        ran_at=BASE_TIME + timedelta(days=i),
        total=20,
        category_accuracy=accuracy,
        avg_summary_score=0.8,
        avg_summary_judge_score=4.0,
        avg_latency_ms=120.0,
        total_tokens=500,
        results=[],
    )


def test_load_history_missing_db_returns_empty(tmp_path: Path):
    assert load_history(tmp_path / "nope.db") == []


def test_load_recent_run_rows_missing_db_returns_empty(tmp_path: Path):
    assert load_recent_run_rows(5, tmp_path / "nope.db") == []


def test_record_and_load_round_trips_summary_fields(tmp_path: Path):
    db_path = tmp_path / "history.db"
    report = _report(0)
    report_path = tmp_path / "eval_v0_20260101T000000Z.json"
    report_path.write_text("{}", encoding="utf-8")

    record_run(report, report_path, db_path=db_path)
    history = load_history(db_path)

    assert len(history) == 1
    loaded = history[0]
    assert loaded.prompt_version == report.prompt_version
    assert loaded.model == report.model
    assert loaded.dataset_version == report.dataset_version
    assert loaded.ran_at == report.ran_at
    assert loaded.total == report.total
    assert loaded.category_accuracy == report.category_accuracy
    assert loaded.avg_summary_score == report.avg_summary_score
    assert loaded.avg_summary_judge_score == report.avg_summary_judge_score
    assert loaded.avg_latency_ms == report.avg_latency_ms
    assert loaded.total_tokens == report.total_tokens
    assert loaded.results == []
    assert loaded.accuracy_by_difficulty == {}
    assert loaded.accuracy_by_category == {}


def test_record_run_same_report_path_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "history.db"
    report_path = tmp_path / "eval_v0.json"
    report_path.write_text("{}", encoding="utf-8")

    record_run(_report(0, accuracy=0.5), report_path, db_path=db_path)
    record_run(_report(0, accuracy=0.9), report_path, db_path=db_path)  # same path, different report

    history = load_history(db_path)
    assert len(history) == 1


def test_load_history_sorted_oldest_first(tmp_path: Path):
    db_path = tmp_path / "history.db"
    for i in (2, 0, 1):
        path = tmp_path / f"eval_v{i}.json"
        path.write_text("{}", encoding="utf-8")
        record_run(_report(i), path, db_path=db_path)

    history = load_history(db_path)

    assert [r.prompt_version for r in history] == ["v0", "v1", "v2"]


def test_load_recent_run_rows_most_recent_first_with_paths(tmp_path: Path):
    db_path = tmp_path / "history.db"
    paths = []
    for i in range(3):
        path = tmp_path / f"eval_v{i}.json"
        path.write_text("{}", encoding="utf-8")
        paths.append(str(path.resolve()))
        record_run(_report(i), path, db_path=db_path)

    rows = load_recent_run_rows(10, db_path=db_path)

    assert [r.prompt_version for r, _ in rows] == ["v2", "v1", "v0"]
    assert [p for _, p in rows] == list(reversed(paths))


def test_load_recent_run_rows_respects_limit(tmp_path: Path):
    db_path = tmp_path / "history.db"
    for i in range(5):
        path = tmp_path / f"eval_v{i}.json"
        path.write_text("{}", encoding="utf-8")
        record_run(_report(i), path, db_path=db_path)

    rows = load_recent_run_rows(2, db_path=db_path)

    assert len(rows) == 2
    assert [r.prompt_version for r, _ in rows] == ["v4", "v3"]


def test_record_run_creates_db_and_parent_dirs(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "history.db"
    report_path = tmp_path / "eval_v0.json"
    report_path.write_text("{}", encoding="utf-8")

    record_run(_report(0), report_path, db_path=db_path)

    assert db_path.exists()
