"""Compares two eval reports and flags quality regressions.

This is the core "did this prompt/model change make things worse" logic.
It's deliberately decoupled from how the two reports were produced — feed it
any two `EvalReport` JSON files (baseline vs candidate) and it tells you
what changed, per example and in aggregate, and whether that change crosses
a regression threshold.

Usage:
    python -m regression_detector.regression \\
        --baseline reports/eval_v1_....json \\
        --candidate reports/eval_v2_....json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from regression_detector.eval_runner import EvalReport

# Default thresholds: how much worse the candidate is allowed to be before
# we call it a regression. Kept small since the golden dataset is small
# (a single flipped example already moves accuracy several points).
DEFAULT_ACCURACY_DROP_THRESHOLD = 0.0  # any drop in category accuracy fails
DEFAULT_SUMMARY_SCORE_DROP_THRESHOLD = 0.05


class ExampleDiff(BaseModel):
    id: str
    email: str
    baseline_category: str | None
    candidate_category: str | None
    category_regressed: bool  # was correct in baseline, wrong in candidate
    category_improved: bool  # was wrong in baseline, correct in candidate
    baseline_summary_score: float
    candidate_summary_score: float
    summary_score_delta: float


class RegressionReport(BaseModel):
    baseline_version: str
    candidate_version: str
    baseline_model: str
    candidate_model: str
    compared_at: datetime
    accuracy_delta: float
    avg_summary_score_delta: float
    regressed_examples: list[str] = Field(default_factory=list)
    improved_examples: list[str] = Field(default_factory=list)
    new_errors: list[str] = Field(default_factory=list)  # ids that started erroring
    is_regression: bool
    reasons: list[str] = Field(default_factory=list)
    example_diffs: list[ExampleDiff] = Field(default_factory=list)


def _load_report(path: str | Path) -> EvalReport:
    return EvalReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def diff_reports(
    baseline: EvalReport,
    candidate: EvalReport,
    accuracy_drop_threshold: float = DEFAULT_ACCURACY_DROP_THRESHOLD,
    summary_score_drop_threshold: float = DEFAULT_SUMMARY_SCORE_DROP_THRESHOLD,
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
                baseline_category=b.actual_category,
                candidate_category=c.actual_category,
                category_regressed=category_regressed,
                category_improved=category_improved,
                baseline_summary_score=b.summary_score,
                candidate_summary_score=c.summary_score,
                summary_score_delta=round(c.summary_score - b.summary_score, 3),
            )
        )

    accuracy_delta = round(candidate.category_accuracy - baseline.category_accuracy, 3)
    avg_summary_score_delta = round(candidate.avg_summary_score - baseline.avg_summary_score, 3)

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

    return RegressionReport(
        baseline_version=baseline.prompt_version,
        candidate_version=candidate.prompt_version,
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        compared_at=datetime.now(timezone.utc),
        accuracy_delta=accuracy_delta,
        avg_summary_score_delta=avg_summary_score_delta,
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
    print(f"\nAccuracy delta:          {report.accuracy_delta:+.3f}")
    print(f"Avg summary score delta: {report.avg_summary_score_delta:+.3f}")

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
    parser.add_argument("--out-dir", default="reports", help="Directory to write the diff report JSON to.")
    args = parser.parse_args(argv)

    baseline = _load_report(args.baseline)
    candidate = _load_report(args.candidate)

    report = diff_reports(
        baseline,
        candidate,
        accuracy_drop_threshold=args.accuracy_drop_threshold,
        summary_score_drop_threshold=args.summary_score_drop_threshold,
    )
    _print_summary(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.compared_at.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"diff_{report.baseline_version}_vs_{report.candidate_version}_{timestamp}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nDiff report written to {out_path}")

    return 1 if report.is_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
