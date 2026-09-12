from datetime import datetime, timezone

from regression_detector.eval_runner import EvalReport, ExampleResult
from regression_detector.regression import classify_severity, diff_reports


def _result(
    id_, category_match, summary_score=0.8, actual_category="billing", error=None, expected_category="billing"
):
    return ExampleResult(
        id=id_,
        email="some email",
        expected_category=expected_category,
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


def test_classify_severity_thresholds():
    assert classify_severity(0.0) == "none"
    assert classify_severity(+0.20) == "none"  # improvements are never escalated
    assert classify_severity(-0.02) == "none"  # below the 3% warning threshold
    assert classify_severity(-0.03) == "warning"  # at the warning threshold
    assert classify_severity(-0.05) == "warning"
    assert classify_severity(-0.08) == "critical"  # at the critical threshold
    assert classify_severity(-0.20) == "critical"


def test_classify_severity_thresholds_are_configurable():
    assert classify_severity(-0.10, warning_threshold=0.15, critical_threshold=0.30) == "none"
    assert classify_severity(-0.20, warning_threshold=0.15, critical_threshold=0.30) == "warning"
    assert classify_severity(-0.35, warning_threshold=0.15, critical_threshold=0.30) == "critical"


def test_per_category_deltas_flag_critical_drop_in_one_category():
    # 5 billing cases (all correct in both) + 5 technical cases, 1 of which
    # flips from correct to incorrect: overall delta is small (-2%, "none"),
    # but the technical category alone drops 20% ("critical") - this is the
    # "2 out of 80 cases flipped, is that signal or noise" case from a
    # per-category angle instead of a per-run angle.
    billing = [_result(f"b{i}", True, expected_category="billing") for i in range(5)]
    baseline_tech = [_result(f"t{i}", True, expected_category="technical") for i in range(5)]
    candidate_tech = [_result(f"t{i}", i != 0, expected_category="technical") for i in range(5)]

    baseline = _report("v1", billing + baseline_tech)
    candidate = _report("v2", billing + candidate_tech)

    diff = diff_reports(baseline, candidate)

    by_category = {cd.category: cd for cd in diff.category_deltas}
    assert by_category["billing"].severity == "none"
    assert by_category["technical"].delta == -0.2
    assert by_category["technical"].severity == "critical"
    assert diff.is_regression is True
    assert any("technical" in r.lower() and "critical" in r.lower() for r in diff.reasons)


def test_category_delta_thresholds_are_configurable_via_diff_reports():
    baseline = [_result(f"t{i}", True, expected_category="technical") for i in range(20)]
    candidate = [_result(f"t{i}", i != 0, expected_category="technical") for i in range(20)]  # 1/20 = 5% drop

    diff_default = diff_reports(_report("v1", baseline), _report("v2", candidate))
    diff_loose = diff_reports(
        _report("v1", baseline),
        _report("v2", candidate),
        warning_delta_threshold=0.10,
        critical_delta_threshold=0.25,
    )

    assert diff_default.category_deltas[0].severity == "warning"  # 5% is between 3% and 8%
    assert diff_loose.category_deltas[0].severity == "none"  # 5% is below the loosened 10% warning bar
