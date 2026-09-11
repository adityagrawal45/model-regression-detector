from datetime import datetime, timedelta, timezone
from pathlib import Path

from regression_detector.drift import detect_drift
from regression_detector.eval_runner import EvalReport, ExampleResult
from regression_detector.regression import diff_reports
from regression_detector.report_html import render_diff_report_html, write_diff_report_html

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _result(id_, category_match, expected_category="billing", actual_category="billing", summary="a summary"):
    return ExampleResult(
        id=id_,
        email=f"email body for {id_}",
        expected_category=expected_category,
        actual_category=actual_category,
        category_match=category_match,
        expected_summary="expected summary",
        actual_summary=summary,
        summary_score=0.8,
        summary_judge_score=4,
        latency_ms=120.0,
        prompt_tokens=50,
        completion_tokens=10,
        total_tokens=60,
    )


def _report(i: int, results):
    return EvalReport(
        prompt_version=f"v{i}",
        model="test-model",
        ran_at=BASE_TIME + timedelta(days=i),
        total=len(results),
        category_accuracy=sum(r.category_match for r in results) / len(results),
        avg_summary_score=0.8,
        avg_summary_judge_score=4.0,
        avg_latency_ms=120.0,
        total_tokens=sum(r.total_tokens for r in results),
        results=results,
    )


def _baseline_and_candidate():
    baseline_results = [_result("a", True), _result("b", True)]
    candidate_results = [_result("a", True), _result("b", False, actual_category="general", summary="wrong summary")]
    return _report(1, baseline_results), _report(2, candidate_results)


def test_render_diff_report_html_contains_metadata_and_scorecard():
    baseline, candidate = _baseline_and_candidate()
    diff = diff_reports(baseline, candidate)

    html_out = render_diff_report_html(baseline, candidate, diff)

    assert "<!doctype html>" in html_out.lower()
    assert baseline.prompt_version in html_out
    assert candidate.prompt_version in html_out
    assert baseline.model in html_out
    assert "Scorecard" in html_out
    assert "Category accuracy" in html_out


def test_render_diff_report_html_shows_regressed_case_old_vs_new():
    baseline, candidate = _baseline_and_candidate()
    diff = diff_reports(baseline, candidate)

    html_out = render_diff_report_html(baseline, candidate, diff)

    assert "wrong summary" in html_out  # candidate (new) output
    assert "a summary" in html_out  # baseline (old) output still shown
    assert ">b<" in html_out  # the regressed case id


def test_render_diff_report_html_no_regressions_message():
    baseline, candidate = _baseline_and_candidate()
    # Make candidate identical to baseline so nothing regresses.
    identical = _report(2, [_result("a", True), _result("b", True)])
    diff = diff_reports(baseline, identical)

    html_out = render_diff_report_html(baseline, identical, diff)

    assert "No regressed cases" in html_out


def test_render_diff_report_html_includes_trend_chart_with_history():
    baseline, candidate = _baseline_and_candidate()
    diff = diff_reports(baseline, candidate)
    history = [baseline, candidate, _report(3, [_result("a", True), _result("b", True)])]

    html_out = render_diff_report_html(baseline, candidate, diff, history=history)

    assert "<svg" in html_out
    assert "Not enough history" not in html_out


def test_render_diff_report_html_shows_drift_section_when_provided():
    baseline, candidate = _baseline_and_candidate()
    diff = diff_reports(baseline, candidate)
    history = [_report(i, [_result("a", True), _result("b", i % 3 != 0)]) for i in range(10)]
    drift = detect_drift(history, window=7)

    html_out = render_diff_report_html(baseline, candidate, diff, history=history, drift=drift)

    assert "Drift detection" in html_out


def test_write_diff_report_html_creates_file(tmp_path: Path):
    baseline, candidate = _baseline_and_candidate()
    diff = diff_reports(baseline, candidate)
    out_path = tmp_path / "nested" / "report.html"

    result_path = write_diff_report_html(baseline, candidate, diff, out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert "<!doctype html>" in out_path.read_text(encoding="utf-8").lower()
