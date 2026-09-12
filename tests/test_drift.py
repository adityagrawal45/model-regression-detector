from datetime import datetime, timedelta, timezone

from regression_detector.drift import detect_drift
from regression_detector.eval_runner import EvalReport

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _report(i: int, accuracy: float) -> EvalReport:
    return EvalReport(
        prompt_version=f"v{i}",
        model="test-model",
        ran_at=BASE_TIME + timedelta(days=i),
        total=20,
        category_accuracy=accuracy,
        avg_summary_score=0.8,
        results=[],
    )


def test_not_enough_history_reports_none_severity():
    history = [_report(i, 0.90) for i in range(3)]

    drift = detect_drift(history, window=7)

    assert drift.current_rolling_avg is None
    assert drift.severity == "none"
    assert drift.is_drift is False
    assert "not enough history" in drift.message.lower()


def test_stable_scores_no_drift():
    history = [_report(i, 0.90) for i in range(20)]

    drift = detect_drift(history, window=7)

    assert drift.current_rolling_avg == 0.90
    assert drift.drift_delta == 0.0
    assert drift.severity == "none"
    assert drift.is_drift is False


def test_slow_decline_below_no_single_run_but_flagged_as_drift():
    # Each run only drops ~1pp - well under the 3% per-run warning bar - but
    # over 14 runs the rolling average has drifted down 13pp, which the
    # trend comparison (this window's avg vs the window before it) should
    # catch even though no single-run diff would.
    accuracies = [0.95 - 0.01 * i for i in range(14)]
    history = [_report(i, acc) for i, acc in enumerate(accuracies)]

    drift = detect_drift(history, window=7)

    assert drift.current_rolling_avg is not None
    assert drift.reference_rolling_avg is not None
    assert drift.drift_delta is not None and drift.drift_delta < 0
    assert drift.severity in {"warning", "critical"}
    assert drift.is_drift is True
    assert "slow drift" in drift.message.lower()


def test_sudden_improvement_is_never_flagged_as_drift():
    accuracies = [0.70] * 7 + [0.99] * 7
    history = [_report(i, acc) for i, acc in enumerate(accuracies)]

    drift = detect_drift(history, window=7)

    assert drift.drift_delta > 0
    assert drift.severity == "none"
    assert drift.is_drift is False


def test_absolute_floor_flags_even_without_a_clear_downward_trend():
    history = [_report(i, 0.60) for i in range(10)]  # flat, but low

    drift = detect_drift(history, window=7, absolute_floor=0.80)

    assert drift.below_absolute_floor is True
    assert drift.is_drift is True
    assert drift.severity != "none"


def test_thresholds_are_configurable():
    accuracies = [0.90 - 0.005 * i for i in range(14)]  # gentle decline
    history = [_report(i, acc) for i, acc in enumerate(accuracies)]

    strict = detect_drift(history, window=7, warning_threshold=0.01, critical_threshold=0.02)
    lenient = detect_drift(history, window=7, warning_threshold=0.20, critical_threshold=0.40)

    assert strict.severity != "none"
    assert lenient.severity == "none"
