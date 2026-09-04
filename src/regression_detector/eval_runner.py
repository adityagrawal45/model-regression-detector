"""Runs the classifier against a golden dataset and scores the results.

Usage:
    python -m regression_detector.eval_runner \\
        --prompt prompts/classifier_v1.yaml \\
        --dataset data/golden_dataset.json

If GROQ_API_KEY is not set, falls back to MockClient (with a warning) so the
whole pipeline stays demoable offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from regression_detector.classifier import classify_email
from regression_detector.llm_client import GroqClient, LLMClient, MockClient
from regression_detector.prompts import load_prompt_config

REPO_ROOT = Path(__file__).resolve().parents[2]


class ExampleResult(BaseModel):
    id: str
    email: str
    expected_category: str
    actual_category: str | None
    category_match: bool
    expected_summary: str
    actual_summary: str | None
    summary_score: float  # 0..1 rough keyword-overlap heuristic
    error: str | None = None


class EvalReport(BaseModel):
    prompt_version: str
    model: str
    ran_at: datetime
    total: int
    category_accuracy: float
    avg_summary_score: float
    results: list[ExampleResult] = Field(default_factory=list)


def _summary_score(actual: str | None, expected: str) -> float:
    """Rough keyword-overlap proxy for summary quality (not a judge model).

    Deliberately simple for this phase: fraction of expected-summary content
    words that also appear in the actual summary. Flags obviously empty or
    wildly off-topic summaries without needing another LLM call.
    """
    if not actual:
        return 0.0
    stopwords = {
        "the", "a", "an", "to", "for", "and", "or", "is", "was", "of", "on",
        "in", "with", "their", "they", "customer", "wants", "asks",
    }
    expected_words = {w.strip(".,!?").lower() for w in expected.split()} - stopwords
    actual_words = {w.strip(".,!?").lower() for w in actual.split()}
    if not expected_words:
        return 1.0
    overlap = expected_words & actual_words
    return round(len(overlap) / len(expected_words), 3)


def run_eval(prompt_path: str | Path, dataset_path: str | Path, client: LLMClient) -> EvalReport:
    prompt_config = load_prompt_config(prompt_path)
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))

    results: list[ExampleResult] = []
    for example in dataset:
        try:
            result = classify_email(example["email"], prompt_config, client)
            results.append(
                ExampleResult(
                    id=example["id"],
                    email=example["email"],
                    expected_category=example["expected_category"],
                    actual_category=result.category,
                    category_match=result.category == example["expected_category"],
                    expected_summary=example["expected_summary"],
                    actual_summary=result.summary,
                    summary_score=_summary_score(result.summary, example["expected_summary"]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - record and continue so one bad example doesn't abort the run
            results.append(
                ExampleResult(
                    id=example["id"],
                    email=example["email"],
                    expected_category=example["expected_category"],
                    actual_category=None,
                    category_match=False,
                    expected_summary=example["expected_summary"],
                    actual_summary=None,
                    summary_score=0.0,
                    error=str(exc),
                )
            )

    total = len(results)
    category_accuracy = round(sum(r.category_match for r in results) / total, 3) if total else 0.0
    avg_summary_score = round(sum(r.summary_score for r in results) / total, 3) if total else 0.0

    return EvalReport(
        prompt_version=prompt_config.version,
        model=prompt_config.model,
        ran_at=datetime.now(timezone.utc),
        total=total,
        category_accuracy=category_accuracy,
        avg_summary_score=avg_summary_score,
        results=results,
    )


def _print_summary(report: EvalReport) -> None:
    print(f"\nPrompt version: {report.prompt_version}  Model: {report.model}")
    print(f"Ran at: {report.ran_at.isoformat()}")
    print(f"{'ID':<10}{'Expected':<12}{'Actual':<12}{'Match':<8}{'Summary':<8}")
    for r in report.results:
        actual = r.actual_category or "ERROR"
        mark = "PASS" if r.category_match else "FAIL"
        print(f"{r.id:<10}{r.expected_category:<12}{actual:<12}{mark:<8}{r.summary_score:<8}")
    print(f"\nCategory accuracy: {report.category_accuracy * 100:.1f}%  "
          f"({sum(r.category_match for r in report.results)}/{report.total})")
    print(f"Avg summary score: {report.avg_summary_score}")

    failures = [r for r in report.results if not r.category_match]
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for r in failures:
            reason = r.error or f"expected={r.expected_category} actual={r.actual_category}"
            print(f"  - {r.id}: {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-dataset eval for the email classifier.")
    parser.add_argument("--prompt", default="prompts/classifier_v1.yaml", help="Path to the prompt YAML file.")
    parser.add_argument("--dataset", default="data/golden_dataset.json", help="Path to the golden dataset JSON file.")
    parser.add_argument("--mock", action="store_true", help="Force MockClient even if GROQ_API_KEY is set.")
    parser.add_argument("--out-dir", default="reports", help="Directory to write the report JSON to.")
    args = parser.parse_args(argv)

    if args.mock or not os.environ.get("GROQ_API_KEY"):
        if not args.mock:
            print("WARNING: GROQ_API_KEY not set, falling back to MockClient (offline mode).", file=sys.stderr)
        client: LLMClient = MockClient()
    else:
        client = GroqClient()

    report = run_eval(args.prompt, args.dataset, client)
    _print_summary(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.ran_at.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"eval_{report.prompt_version}_{timestamp}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    return 0 if report.category_accuracy == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
