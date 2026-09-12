from datetime import datetime, timedelta, timezone
from pathlib import Path

from regression_detector.dashboard import render_dashboard_html, write_dashboard_html
from regression_detector.drift import detect_drift
from regression_detector.eval_runner import EvalReport

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _report(i: int, accuracy: float = 0.9, latency: float = 100.0, tokens: int = 500) -> EvalReport:
    return EvalReport(
        prompt_version=f"v{i}",
        model="test-model",
        ran_at=BASE_TIME + timedelta(days=i),
        total=20,
        category_accuracy=accuracy,
        avg_summary_score=0.8,
        avg_latency_ms=latency,
        total_tokens=tokens,
        results=[],
    )


def test_render_dashboard_html_empty_history():
    html_out = render_dashboard_html([], [])

    assert "<!doctype html>" in html_out.lower()
    assert "Not enough history" in html_out
    assert "No runs recorded yet" in html_out


def test_render_dashboard_html_single_run():
    history = [_report(0)]
    rows = [(history[0], "/does/not/exist/eval_v0.json")]

    html_out = render_dashboard_html(history, rows)

    assert "Not enough history" in html_out  # charts need >= 2 points
    assert "v0" in html_out  # table still shows the one row


def test_render_dashboard_html_many_runs_shows_charts_and_table():
    history = [_report(i, accuracy=0.8 + i * 0.01, latency=100 + i, tokens=500 + i * 10) for i in range(5)]
    rows = [(r, f"/does/not/exist/eval_v{i}.json") for i, r in enumerate(history)]

    html_out = render_dashboard_html(history, rows)

    assert html_out.count("<svg") == 3
    for i in range(5):
        assert f"v{i}" in html_out
    assert "5 run(s) of history" in html_out


def test_render_dashboard_html_includes_drift_status_when_provided():
    history = [_report(i, accuracy=1.0 if i % 3 else 0.5) for i in range(10)]
    drift = detect_drift(history, window=7)

    html_out = render_dashboard_html(history, [], drift=drift)

    assert "Drift status" in html_out
    assert drift.severity in html_out


def test_write_dashboard_html_creates_file(tmp_path: Path):
    out_path = tmp_path / "nested" / "dashboard.html"

    result_path = write_dashboard_html([_report(0), _report(1)], [], out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert "<!doctype html>" in out_path.read_text(encoding="utf-8").lower()


def test_runs_table_links_to_report_path_when_file_exists(tmp_path: Path):
    report_path = tmp_path / "eval_v0.json"
    report_path.write_text("{}", encoding="utf-8")
    history = [_report(0), _report(1)]
    rows = [(history[0], str(report_path)), (history[1], str(tmp_path / "missing.json"))]

    html_out = render_dashboard_html(history, rows)

    assert report_path.as_uri() in html_out
    assert "missing.json" not in html_out  # no link generated for a missing file, but no crash either
