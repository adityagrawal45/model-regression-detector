"""Runs the classifier against a golden dataset and scores the results.

Every case is scored on multiple, independent dimensions rather than a single
pass/fail bit:
  - category match (binary - the category is either right or it isn't)
  - summary relevance (1-5, via an LLM-as-judge - see judge.py)
  - latency per request (ms)
  - token usage per request (prompt/completion/total)

Cases run concurrently (bounded by `--concurrency`) via asyncio so a 60+ case
golden dataset doesn't mean 60+ sequential round trips to the model provider.

Usage:
    python -m regression_detector.eval_runner \\
        --prompt prompts/classifier_v1.yaml \\
        --dataset data/golden_dataset.json

If GROQ_API_KEY is not set, falls back to MockClient (with a warning) so the
whole pipeline stays demoable offline.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from regression_detector.classifier import classify_email_detailed_async
from regression_detector.config import PromptConfig
from regression_detector.dataset import GoldenExample, load_golden_dataset
from regression_detector.judge import judge_summary_async
from regression_detector.llm_client import GroqClient, LLMClient, MockClient
from regression_detector.prompts import load_prompt_config

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONCURRENCY = 5


class ExampleResult(BaseModel):
    id: str
    email: str
    expected_category: str
    actual_category: str | None
    category_match: bool
    expected_summary: str
    actual_summary: str | None
    summary_score: float  # 0..1 keyword-overlap heuristic - cheap sanity signal
    summary_judge_score: int | None = None  # 1..5, from the LLM-as-judge
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    difficulty: str | None = None
    error: str | None = None


class EvalReport(BaseModel):
    prompt_version: str
    model: str
    dataset_version: str | None = None
    ran_at: datetime
    total: int
    category_accuracy: float
    avg_summary_score: float
    avg_summary_judge_score: float | None = None
    avg_latency_ms: float | None = None
    total_tokens: int | None = None
    accuracy_by_difficulty: dict[str, float] = Field(default_factory=dict)
    accuracy_by_category: dict[str, float] = Field(default_factory=dict)
    results: list[ExampleResult] = Field(default_factory=list)


def _summary_score(actual: str | None, expected: str) -> float:
    """Rough keyword-overlap proxy for summary quality (not a judge model).

    Deliberately simple: fraction of expected-summary content words that also
    appear in the actual summary. Cheap enough to run with zero extra LLM
    calls, so it stays around alongside the judge score as a sanity check.
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


def _accuracy_by(results: list[ExampleResult], key: str) -> dict[str, float]:
    groups: dict[str, list[ExampleResult]] = {}
    for r in results:
        value = getattr(r, key)
        if value is not None:
            groups.setdefault(value, []).append(r)
    return {
        value: round(sum(r.category_match for r in group) / len(group), 3)
        for value, group in sorted(groups.items())
    }


async def _eval_one(
    example: GoldenExample,
    prompt_config: PromptConfig,
    client: LLMClient,
    judge_client: LLMClient,
    semaphore: asyncio.Semaphore,
) -> ExampleResult:
    async with semaphore:
        try:
            outcome = await classify_email_detailed_async(example.email, prompt_config, client)
        except Exception as exc:  # noqa: BLE001 - one bad case shouldn't abort the run
            return ExampleResult(
                id=example.id,
                email=example.email,
                expected_category=example.expected_category,
                actual_category=None,
                category_match=False,
                expected_summary=example.expected_summary,
                actual_summary=None,
                summary_score=0.0,
                difficulty=example.expected_difficulty,
                error=str(exc),
            )

        result = outcome.result
        judge_score: int | None = None
        try:
            judge_score = await judge_summary_async(
                judge_client, example.email, example.expected_summary, result.summary
            )
        except Exception:  # noqa: BLE001 - a judge failure shouldn't fail the whole case
            judge_score = None

        usage = outcome.usage
        return ExampleResult(
            id=example.id,
            email=example.email,
            expected_category=example.expected_category,
            actual_category=result.category,
            category_match=result.category == example.expected_category,
            expected_summary=example.expected_summary,
            actual_summary=result.summary,
            summary_score=_summary_score(result.summary, example.expected_summary),
            summary_judge_score=judge_score,
            latency_ms=round(outcome.latency_ms, 1),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            difficulty=example.expected_difficulty,
        )


async def run_eval_async(
    prompt_path: str | Path,
    dataset_path: str | Path,
    client: LLMClient,
    judge_client: LLMClient | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> EvalReport:
    """Run every golden-dataset case through the classifier, batched concurrently."""
    prompt_config = load_prompt_config(prompt_path)
    dataset = load_golden_dataset(dataset_path)
    judge_client = judge_client or client
    semaphore = asyncio.Semaphore(max(1, concurrency))

    results = await asyncio.gather(
        *(_eval_one(example, prompt_config, client, judge_client, semaphore) for example in dataset.cases)
    )
    results = list(results)

    total = len(results)
    category_accuracy = round(sum(r.category_match for r in results) / total, 3) if total else 0.0
    avg_summary_score = round(sum(r.summary_score for r in results) / total, 3) if total else 0.0

    judge_scores = [r.summary_judge_score for r in results if r.summary_judge_score is not None]
    avg_summary_judge_score = round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None

    token_totals = [r.total_tokens for r in results if r.total_tokens is not None]
    total_tokens = sum(token_totals) if token_totals else None

    return EvalReport(
        prompt_version=prompt_config.version,
        model=prompt_config.model,
        dataset_version=dataset.version,
        ran_at=datetime.now(timezone.utc),
        total=total,
        category_accuracy=category_accuracy,
        avg_summary_score=avg_summary_score,
        avg_summary_judge_score=avg_summary_judge_score,
        avg_latency_ms=avg_latency_ms,
        total_tokens=total_tokens,
        accuracy_by_difficulty=_accuracy_by(results, "difficulty"),
        accuracy_by_category=_accuracy_by(results, "expected_category"),
        results=results,
    )


def run_eval(
    prompt_path: str | Path,
    dataset_path: str | Path,
    client: LLMClient,
    judge_client: LLMClient | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> EvalReport:
    """Sync entry point (used by the CLI and by tests) around `run_eval_async`."""
    return asyncio.run(run_eval_async(prompt_path, dataset_path, client, judge_client, concurrency))


def _print_summary(report: EvalReport) -> None:
    print(f"\nPrompt version: {report.prompt_version}  Model: {report.model}")
    if report.dataset_version:
        print(f"Dataset version: {report.dataset_version}")
    print(f"Ran at: {report.ran_at.isoformat()}")
    print(f"{'ID':<10}{'Expected':<12}{'Actual':<12}{'Match':<8}{'Summary':<8}{'Judge':<7}{'Latency':<10}")
    for r in report.results:
        actual = r.actual_category or "ERROR"
        mark = "PASS" if r.category_match else "FAIL"
        judge = r.summary_judge_score if r.summary_judge_score is not None else "-"
        latency = f"{r.latency_ms:.0f}ms" if r.latency_ms is not None else "-"
        print(f"{r.id:<10}{r.expected_category:<12}{actual:<12}{mark:<8}{r.summary_score:<8}{judge:<7}{latency:<10}")
    print(f"\nCategory accuracy: {report.category_accuracy * 100:.1f}%  "
          f"({sum(r.category_match for r in report.results)}/{report.total})")
    print(f"Avg summary score (heuristic): {report.avg_summary_score}")
    if report.avg_summary_judge_score is not None:
        print(f"Avg summary score (LLM judge, 1-5): {report.avg_summary_judge_score}")
    if report.avg_latency_ms is not None:
        print(f"Avg latency: {report.avg_latency_ms}ms")
    if report.total_tokens is not None:
        print(f"Total tokens used: {report.total_tokens}")

    if report.accuracy_by_difficulty:
        breakdown = "  ".join(
            f"{difficulty}={acc * 100:.0f}%" for difficulty, acc in report.accuracy_by_difficulty.items()
        )
        print(f"Accuracy by difficulty: {breakdown}")

    if report.accuracy_by_category:
        breakdown = "  ".join(
            f"{category}={acc * 100:.0f}%" for category, acc in report.accuracy_by_category.items()
        )
        print(f"Accuracy by category: {breakdown}")

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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max number of golden-dataset cases to evaluate concurrently (default: 5).",
    )
    args = parser.parse_args(argv)

    if args.mock or not os.environ.get("GROQ_API_KEY"):
        if not args.mock:
            print("WARNING: GROQ_API_KEY not set, falling back to MockClient (offline mode).", file=sys.stderr)
        client: LLMClient = MockClient()
    else:
        client = GroqClient()

    report = run_eval(args.prompt, args.dataset, client, concurrency=args.concurrency)
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
