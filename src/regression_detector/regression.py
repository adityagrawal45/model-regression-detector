"""Compares two eval reports and flags quality regressions.

This is the core "did this prompt/model change make things worse" logic.
It's deliberately decoupled from how the two reports were produced — feed it
any two `EvalReport` JSON files (baseline vs candidate) and it tells you
what changed, per example and in aggregate, and whether that change crosses
a regression threshold.

Two different kinds of threshold are in play here:
  - `accuracy_drop_threshold` / `summary_score_drop_threshold`: the "is this
    even allowed" gate used by `is_regression` (a CI-style pass/fail).
  - `warning_delta_threshold` / `critical_delta_threshold`: a severity scale
    for how big a change is, on the overall pass rate and on each category
    individually. With a 60-case dataset, 2 cases flipping is a ~3.3% delta
    and might be noise; 5 cases flipping is ~8.3% and probably isn't. These
    default to 3%/8% and are meant to be tuned to the size of your own
    golden dataset.

Usage:
    python -m regression_detector.regression \\
        --baseline reports/eval_v1_....json \\
        --candidate reports/eval_v2_....json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from regression_detector.alerts import SlackAlertError, send_slack_alert
from regression_detector.drift import DEFAULT_DRIFT_WINDOW, detect_drift, load_eval_history
from regression_detector.eval_runner import EvalReport, ExampleResult
from regression_detector.report_html import write_diff_report_html
from regression_detector.severity import (
    DEFAULT_CRITICAL_DELTA_THRESHOLD,
    DEFAULT_WARNING_DELTA_THRESHOLD,
    Severity,
    classify_severity,
)

# Default thresholds: how much worse the candidate is allowed to be before
# we call it a regression. Kept small since the golden dataset is small
# (a single flipped example already moves accuracy several points).
DEFAULT_ACCURACY_DROP_THRESHOLD = 0.0  # any drop in category accuracy fails
DEFAULT_SUMMARY_SCORE_DROP_THRESHOLD = 0.05


class CategoryDelta(BaseModel):
    category: str
    baseline_accuracy: float
    candidate_accuracy: float
    delta: float
    severity: Severity


class ExampleDiff(BaseModel):
    id: str
    email: str
    expected_category: str
    baseline_category: str | None
    candidate_category: str | None
    category_regressed: bool  # was correct in baseline, wrong in candidate
    category_improved: bool  # was wrong in baseline, correct in candidate
    baseline_summary: str | None = None
    candidate_summary: str | None = None
    baseline_summary_score: float
    candidate_summary_score: float
    summary_score_delta: float


class RegressionReport(BaseModel):
    baseline_version: str
    candidate_version: str
    baseline_model: str
    candidate_model: str
    compared_at: datetime
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float  # overall pass rate delta
    accuracy_severity: Severity
    avg_summary_score_delta: float
    category_deltas: list[CategoryDelta] = Field(default_factory=list)
    warning_delta_threshold: float
    critical_delta_threshold: float
    regressed_examples: list[str] = Field(default_factory=list)
    improved_examples: list[str] = Field(default_factory=list)
    new_errors: list[str] = Field(default_factory=list)  # ids that started erroring
    is_regression: bool
    reasons: list[str] = Field(default_factory=list)
    example_diffs: list[ExampleDiff] = Field(default_factory=list)


def _load_report(path: str | Path) -> EvalReport:
    return EvalReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _category_accuracy(results: list[ExampleResult]) -> dict[str, float]:
    by_category: dict[str, list[ExampleResult]] = {}
    for r in results:
        by_category.setdefault(r.expected_category, []).append(r)
    return {
        category: sum(r.category_match for r in group) / len(group)
        for category, group in by_category.items()
    }


def _category_deltas(
    baseline: EvalReport,
    candidate: EvalReport,
    warning_threshold: float,
    critical_threshold: float,
) -> list[CategoryDelta]:
    baseline_acc = _category_accuracy(baseline.results)
    candidate_acc = _category_accuracy(candidate.results)
    shared_categories = sorted(set(baseline_acc) & set(candidate_acc))

    deltas = []
    for category in shared_categories:
        b_acc = baseline_acc[category]
        c_acc = candidate_acc[category]
        delta = round(c_acc - b_acc, 3)
        deltas.append(
            CategoryDelta(
                category=category,
                baseline_accuracy=round(b_acc, 3),
                candidate_accuracy=round(c_acc, 3),
                delta=delta,
                severity=classify_severity(delta, warning_threshold, critical_threshold),
            )
        )
    return deltas


def diff_reports(
    baseline: EvalReport,
    candidate: EvalReport,
    accuracy_drop_threshold: float = DEFAULT_ACCURACY_DROP_THRESHOLD,
    summary_score_drop_threshold: float = DEFAULT_SUMMARY_SCORE_DROP_THRESHOLD,
    warning_delta_threshold: float = DEFAULT_WARNING_DELTA_THRESHOLD,
    critical_delta_threshold: float = DEFAULT_CRITICAL_DELTA_THRESHOLD,
) -> RegressionReport:
    baseline_by_id = {r.id: r for r in baseline.results}
    candidate_by_id = {r.id: r for r in candidate.results}
    shared_ids = [r.id for r in baseline.results if r.id in candidate_by_id]

    example_diffs: list[ExampleDiff] = []
    regressed_examples: list[str] = []
    improved_examples: list[str] = []
    new_errors: list[str] = []

    for example_id in shared_ids:
        b = baseline_by_id[example_id]
        c = candidate_by_id[example_id]

        category_regressed = b.category_match and not c.category_match
        category_improved = (not b.category_match) and c.category_match

        if category_regressed:
            regressed_examples.append(example_id)
        if category_improved:
            improved_examples.append(example_id)
        if c.error and not b.error:
            new_errors.append(example_id)

        example_diffs.append(
            ExampleDiff(
                id=example_id,
                email=c.email,
                expected_category=c.expected_category,
                baseline_category=b.actual_category,
                candidate_category=c.actual_category,
                category_regressed=category_regressed,
                category_improved=category_improved,
                baseline_summary=b.actual_summary,
                candidate_summary=c.actual_summary,
                baseline_summary_score=b.summary_score,
                candidate_summary_score=c.summary_score,
                summary_score_delta=round(c.summary_score - b.summary_score, 3),
            )
        )

    accuracy_delta = round(candidate.category_accuracy - baseline.category_accuracy, 3)
    avg_summary_score_delta = round(candidate.avg_summary_score - baseline.avg_summary_score, 3)
    accuracy_severity = classify_severity(accuracy_delta, warning_delta_threshold, critical_delta_threshold)
    category_deltas = _category_deltas(baseline, candidate, warning_delta_threshold, critical_delta_threshold)

    reasons: list[str] = []
    if accuracy_delta < -accuracy_drop_threshold:
        reasons.append(
            f"Category accuracy dropped by {abs(accuracy_delta) * 100:.1f}pp "
            f"({baseline.category_accuracy * 100:.1f}% -> {candidate.category_accuracy * 100:.1f}%)."
        )
    if avg_summary_score_delta < -summary_score_drop_threshold:
        reasons.append(
            f"Avg summary score dropped by {abs(avg_summary_score_delta)} "
            f"({baseline.avg_summary_score} -> {candidate.avg_summary_score})."
        )
    if regressed_examples:
        reasons.append(
            f"{len(regressed_examples)} example(s) flipped from correct to incorrect: "
            f"{', '.join(regressed_examples)}."
        )
    if new_errors:
        reasons.append(f"{len(new_errors)} example(s) started erroring: {', '.join(new_errors)}.")
    for cd in category_deltas:
        if cd.severity != "none":
            reasons.append(
                f"[{cd.severity.upper()}] '{cd.category}' accuracy moved by {cd.delta * 100:+.1f}pp "
                f"({cd.baseline_accuracy * 100:.1f}% -> {cd.candidate_accuracy * 100:.1f}%)."
            )

    return RegressionReport(
        baseline_version=baseline.prompt_version,
        candidate_version=candidate.prompt_version,
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        compared_at=datetime.now(timezone.utc),
        baseline_accuracy=baseline.category_accuracy,
        candidate_accuracy=candidate.category_accuracy,
        accuracy_delta=accuracy_delta,
        accuracy_severity=accuracy_severity,
        avg_summary_score_delta=avg_summary_score_delta,
        category_deltas=category_deltas,
        warning_delta_threshold=warning_delta_threshold,
        critical_delta_threshold=critical_delta_threshold,
        regressed_examples=regressed_examples,
        improved_examples=improved_examples,
        new_errors=new_errors,
        is_regression=bool(reasons),
        reasons=reasons,
        example_diffs=example_diffs,
    )


def _print_summary(report: RegressionReport) -> None:
    print(f"\nBaseline:  {report.baseline_version} ({report.baseline_model})")
    print(f"Candidate: {report.candidate_version} ({report.candidate_model})")
    print(f"Compared at: {report.compared_at.isoformat()}")
    print(f"\nOverall pass rate delta: {report.accuracy_delta:+.3f}  [{report.accuracy_severity}]")
    print(f"Avg summary score delta: {report.avg_summary_score_delta:+.3f}")

    if report.category_deltas:
        print("\nPer-category accuracy delta:")
        for cd in report.category_deltas:
            print(
                f"  - {cd.category}: {cd.baseline_accuracy * 100:.1f}% -> {cd.candidate_accuracy * 100:.1f}% "
                f"({cd.delta * 100:+.1f}pp) [{cd.severity}]"
            )

    if report.regressed_examples:
        print(f"\nRegressed examples ({len(report.regressed_examples)}):")
        for d in report.example_diffs:
            if d.category_regressed:
                print(f"  - {d.id}: {d.baseline_category} -> {d.candidate_category}")

    if report.improved_examples:
        print(f"\nImproved examples ({len(report.improved_examples)}):")
        for d in report.example_diffs:
            if d.category_improved:
                print(f"  - {d.id}: {d.baseline_category} -> {d.candidate_category}")

    if report.is_regression:
        print("\nREGRESSION DETECTED:")
        for reason in report.reasons:
            print(f"  - {reason}")
    else:
        print("\nNo regression detected.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two eval reports and flag regressions between them."
    )
    parser.add_argument("--baseline", required=True, help="Path to the baseline EvalReport JSON.")
    parser.add_argument("--candidate", required=True, help="Path to the candidate EvalReport JSON.")
    parser.add_argument(
        "--accuracy-drop-threshold",
        type=float,
        default=DEFAULT_ACCURACY_DROP_THRESHOLD,
        help="Max allowed drop in category accuracy before it's flagged as a regression (default: 0.0).",
    )
    parser.add_argument(
        "--summary-score-drop-threshold",
        type=float,
        default=DEFAULT_SUMMARY_SCORE_DROP_THRESHOLD,
        help="Max allowed drop in avg summary score before it's flagged as a regression (default: 0.05).",
    )
    parser.add_argument(
        "--warning-delta-threshold",
        type=float,
        default=DEFAULT_WARNING_DELTA_THRESHOLD,
        help="Accuracy delta magnitude (overall or per-category) that triggers a WARNING severity (default: 0.03).",
    )
    parser.add_argument(
        "--critical-delta-threshold",
        type=float,
        default=DEFAULT_CRITICAL_DELTA_THRESHOLD,
        help="Accuracy delta magnitude (overall or per-category) that triggers a CRITICAL severity (default: 0.08).",
    )
    parser.add_argument("--out-dir", default="reports", help="Directory to write the diff report JSON to.")
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Directory of past eval_*.json reports to build the trend chart and drift check from "
        "(default: same directory as --baseline).",
    )
    parser.add_argument(
        "--drift-window", type=int, default=DEFAULT_DRIFT_WINDOW, help="Rolling-average window size in runs (default: 7)."
    )
    parser.add_argument(
        "--drift-absolute-floor",
        type=float,
        default=None,
        help="Optional fixed floor for the rolling average category accuracy; below it is always at least a warning.",
    )
    parser.add_argument("--html-out", default=None, help="Path to write the HTML diff report to (default: alongside the JSON diff report).")
    parser.add_argument("--no-html", action="store_true", help="Skip generating the HTML diff report.")
    parser.add_argument(
        "--slack", action="store_true", help="Send a Slack alert via SLACK_WEBHOOK_URL (or --slack-webhook-url)."
    )
    parser.add_argument("--slack-webhook-url", default=None, help="Slack incoming webhook URL (default: $SLACK_WEBHOOK_URL).")
    parser.add_argument(
        "--report-url", default=None, help="Public URL of the HTML report to link to from the Slack alert."
    )
    args = parser.parse_args(argv)

    baseline = _load_report(args.baseline)
    candidate = _load_report(args.candidate)

    report = diff_reports(
        baseline,
        candidate,
        accuracy_drop_threshold=args.accuracy_drop_threshold,
        summary_score_drop_threshold=args.summary_score_drop_threshold,
        warning_delta_threshold=args.warning_delta_threshold,
        critical_delta_threshold=args.critical_delta_threshold,
    )
    _print_summary(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.compared_at.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"diff_{report.baseline_version}_vs_{report.candidate_version}_{timestamp}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nDiff report written to {out_path}")

    history_dir = Path(args.history_dir) if args.history_dir else Path(args.baseline).resolve().parent
    history = load_eval_history(history_dir) if history_dir.exists() else []
    drift = detect_drift(
        history,
        window=args.drift_window,
        warning_threshold=args.warning_delta_threshold,
        critical_threshold=args.critical_delta_threshold,
        absolute_floor=args.drift_absolute_floor,
    )
    if drift.is_drift:
        print(f"\nDRIFT: {drift.message}")

    html_path = None
    if not args.no_html:
        html_path = Path(args.html_out) if args.html_out else out_dir / f"{out_path.stem}.html"
        write_diff_report_html(baseline, candidate, report, html_path, history=history, drift=drift)
        print(f"HTML diff report written to {html_path}")

    if args.slack:
        report_url = args.report_url or (str(html_path) if html_path else None)
        try:
            send_slack_alert(report, webhook_url=args.slack_webhook_url, report_url=report_url, drift=drift)
            print("Slack alert sent.")
        except SlackAlertError as exc:
            print(f"WARNING: Slack alert not sent: {exc}", file=sys.stderr)

    return 1 if report.is_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
