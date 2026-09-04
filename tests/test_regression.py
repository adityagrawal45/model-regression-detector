from datetime import datetime, timezone

from regression_detector.eval_runner import EvalReport, ExampleResult
from regression_detector.regression import diff_reports


def _result(id_, category_match, summary_score=0.8, actual_category="billing", error=None):
    return ExampleResult(
        id=id_,
        email="some email",
        expected_category="billing",
        actual_category=actual_category,
        category_match=category_match,
        expected_summary="expected summary",
        actual_summary="actual summary",
        summary_score=summary_score,
        error=error,
    )


def _report(version, results, model="test-model"):
    total = len(results)
    accuracy = sum(r.category_match for r in results) / total
    avg_summary = sum(r.summary_score for r in results) / total
    return EvalReport(
        prompt_version=version,
        model=model,
        ran_at=datetime.now(timezone.utc),
        total=total,
        category_accuracy=round(accuracy, 3),
        avg_summary_score=round(avg_summary, 3),
        results=results,
    )


def test_no_regression_when_reports_identical():
    results = [_result("a", True), _result("b", True), _result("c", False, actual_category="general")]
    baseline = _report("v1", results)
    candidate = _report("v2", results)

    diff = diff_reports(baseline, candidate)

    assert diff.is_regression is False
    assert diff.regressed_examples == []
    assert diff.accuracy_delta == 0.0
    assert diff.reasons == []


def test_flags_example_that_flips_from_correct_to_incorrect():
    baseline = _report("v1", [_result("a", True), _result("b", True)])
    candidate = _report("v2", [_result("a", True), _result("b", False, actual_category="general")])

    diff = diff_reports(baseline, candidate)

    assert diff.is_regression is True
    assert diff.regressed_examples == ["b"]
    assert diff.accuracy_delta < 0
    assert any("accuracy dropped" in r.lower() for r in diff.reasons)


def test_improvement_is_not_flagged_as_regression():
    baseline = _report("v1", [_result("a", False, actual_category="general"), _result("b", True)])
    candidate = _report("v2", [_result("a", True), _result("b", True)])

    diff = diff_reports(baseline, candidate)

    assert diff.is_regression is False
    assert diff.improved_examples == ["a"]
    assert diff.regressed_examples == []
    assert diff.accuracy_delta > 0


def test_summary_score_drop_beyond_threshold_flags_regression():
    baseline = _report("v1", [_result("a", True, summary_score=0.9)])
    candidate = _report("v2", [_result("a", True, summary_score=0.5)])

    diff = diff_reports(baseline, candidate, summary_score_drop_threshold=0.05)

    assert diff.is_regression is True
    assert any("summary score dropped" in r.lower() for r in diff.reasons)


def test_summary_score_drop_within_threshold_is_not_flagged():
    baseline = _report("v1", [_result("a", True, summary_score=0.90)])
    candidate = _report("v2", [_result("a", True, summary_score=0.87)])

    diff = diff_reports(baseline, candidate, summary_score_drop_threshold=0.05)

    assert diff.is_regression is False


def test_new_error_is_flagged():
    baseline = _report("v1", [_result("a", True)])
    candidate_results = [_result("a", False, actual_category=None, summary_score=0.0, error="boom")]
    candidate = _report("v2", candidate_results)

    diff = diff_reports(baseline, candidate)

    assert diff.is_regression is True
    assert diff.new_errors == ["a"]
    assert any("started erroring" in r.lower() for r in diff.reasons)


def test_ids_only_in_one_report_are_ignored():
    baseline = _report("v1", [_result("a", True), _result("shared", True)])
    candidate = _report("v2", [_result("shared", True), _result("only-in-candidate", True)])

    diff = diff_reports(baseline, candidate)

    assert len(diff.example_diffs) == 1
    assert diff.example_diffs[0].id == "shared"
