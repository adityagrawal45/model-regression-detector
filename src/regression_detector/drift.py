"""Rolling-average drift detection across a history of eval runs.

`regression.py` catches a single bad run: prompt v1 -> v2 drops accuracy by
10pp, that's an obvious regression. What it can't catch is a *slow* decline -
each individual run only drifts down a point or two (well under any per-run
threshold), but ten runs later the model has quietly gotten meaningfully
worse. This module tracks a rolling average of category accuracy across a
run history and flags it when the trend itself crosses a threshold, even
though no single run in that history would have tripped `diff_reports`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from regression_detector.eval_runner import EvalReport
from regression_detector.severity import (
    DEFAULT_CRITICAL_DELTA_THRESHOLD,
    DEFAULT_WARNING_DELTA_THRESHOLD,
    Severity,
    classify_severity,
)

DEFAULT_DRIFT_WINDOW = 7


class DriftPoint(BaseModel):
    ran_at: datetime
    prompt_version: str
    category_accuracy: float
    rolling_avg: float | None = None  # None until `window` runs of history exist


class DriftReport(BaseModel):
    window: int
    history_length: int
    points: list[DriftPoint] = Field(default_factory=list)
    current_rolling_avg: float | None = None
    reference_rolling_avg: float | None = None  # rolling avg as of one window earlier
    drift_delta: float | None = None  # current - reference; negative is decline
    severity: Severity = "none"
    absolute_floor: float | None = None
    below_absolute_floor: bool = False
    is_drift: bool = False
    message: str


def load_eval_history(history_dir: str | Path) -> list[EvalReport]:
    """Load every `eval_*.json` report in a directory, oldest first."""
    reports = []
    for path in sorted(Path(history_dir).glob("eval_*.json")):
        try:
            reports.append(EvalReport.model_validate_json(path.read_text(encoding="utf-8")))
        except ValueError:
            continue  # skip anything that isn't a well-formed EvalReport
    reports.sort(key=lambda r: r.ran_at)
    return reports


def _rolling_averages(history: list[EvalReport], window: int) -> list[DriftPoint]:
    points: list[DriftPoint] = []
    accuracies = [r.category_accuracy for r in history]
    for i, report in enumerate(history):
        rolling_avg = None
        if i + 1 >= window:
            recent = accuracies[i + 1 - window : i + 1]
            rolling_avg = round(sum(recent) / len(recent), 4)
        points.append(
            DriftPoint(
                ran_at=report.ran_at,
                prompt_version=report.prompt_version,
                category_accuracy=report.category_accuracy,
                rolling_avg=rolling_avg,
            )
        )
    return points


def detect_drift(
    history: list[EvalReport],
    window: int = DEFAULT_DRIFT_WINDOW,
    warning_threshold: float = DEFAULT_WARNING_DELTA_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_DELTA_THRESHOLD,
    absolute_floor: float | None = None,
) -> DriftReport:
    """Detect slow drift in a chronological history of eval runs.

    Compares the current `window`-run rolling average of category accuracy
    against the rolling average as of one window earlier. A decline that
    crosses `warning_threshold`/`critical_threshold` (the same scale used for
    per-run/per-category severity in regression.py) is flagged even if no
    single run in the history triggered a per-run regression.

    `absolute_floor`, if given, independently flags when the current rolling
    average itself drops below a fixed bar - useful once you know what "good"
    looks like for a given feature, regardless of trend.
    """
    history = sorted(history, key=lambda r: r.ran_at)
    points = _rolling_averages(history, window)

    rolling_values = [p.rolling_avg for p in points if p.rolling_avg is not None]
    current_rolling_avg = rolling_values[-1] if rolling_values else None

    reference_rolling_avg = None
    if current_rolling_avg is not None and len(rolling_values) > window:
        reference_rolling_avg = rolling_values[-1 - window]
    elif current_rolling_avg is not None and len(rolling_values) >= 2:
        # Not a full window of history behind the current one yet, but at
        # least two rolling-average points exist - compare against the
        # earliest available rolling average rather than nothing at all.
        reference_rolling_avg = rolling_values[0]

    drift_delta = None
    severity: Severity = "none"
    if current_rolling_avg is not None and reference_rolling_avg is not None:
        drift_delta = round(current_rolling_avg - reference_rolling_avg, 4)
        severity = classify_severity(drift_delta, warning_threshold, critical_threshold)

    below_absolute_floor = (
        absolute_floor is not None
        and current_rolling_avg is not None
        and current_rolling_avg < absolute_floor
    )
    if below_absolute_floor and severity == "none":
        severity = "warning"

    is_drift = severity != "none"

    if current_rolling_avg is None:
        message = f"Not enough history yet: need {window} runs, have {len(history)}."
    elif drift_delta is None:
        message = (
            f"Current {window}-run rolling average is {current_rolling_avg * 100:.1f}%; "
            "not enough prior history to compare a trend yet."
        )
    elif not is_drift:
        message = f"No drift: {window}-run rolling average moved {drift_delta * 100:+.1f}pp."
    else:
        message = (
            f"[{severity.upper()}] Slow drift detected: {window}-run rolling average moved "
            f"{drift_delta * 100:+.1f}pp ({reference_rolling_avg * 100:.1f}% -> {current_rolling_avg * 100:.1f}%) "
            "even though no single run may have triggered a per-run regression."
        )
    if below_absolute_floor:
        message += f" Also below the absolute floor of {absolute_floor * 100:.1f}%."

    return DriftReport(
        window=window,
        history_length=len(history),
        points=points,
        current_rolling_avg=current_rolling_avg,
        reference_rolling_avg=reference_rolling_avg,
        drift_delta=drift_delta,
        severity=severity,
        absolute_floor=absolute_floor,
        below_absolute_floor=below_absolute_floor,
        is_drift=is_drift,
        message=message,
    )


def detect_drift_from_dir(
    history_dir: str | Path,
    window: int = DEFAULT_DRIFT_WINDOW,
    warning_threshold: float = DEFAULT_WARNING_DELTA_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_DELTA_THRESHOLD,
    absolute_floor: float | None = None,
) -> DriftReport:
    """Convenience wrapper: load every eval report in a directory, then detect_drift."""
    history = load_eval_history(history_dir)
    return detect_drift(history, window, warning_threshold, critical_threshold, absolute_floor)


__all__ = [
    "DEFAULT_DRIFT_WINDOW",
    "DriftPoint",
    "DriftReport",
    "detect_drift",
    "detect_drift_from_dir",
    "load_eval_history",
]
