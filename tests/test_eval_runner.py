import json
from pathlib import Path

from regression_detector.eval_runner import run_eval
from regression_detector.llm_client import MockClient

FIXTURE_DATASET = {
    "version": "test-fixture",
    "created_at": "2026-01-01T00:00:00Z",
    "cases": [
        {
            "id": "t-001",
            "email": "I was charged twice for my subscription, please issue a refund.",
            "expected_category": "billing",
            "expected_summary": "Customer was double-charged and wants a refund.",
            "expected_difficulty": "easy",
            "notes": "Fixture case for eval runner tests.",
        },
        {
            "id": "t-002",
            "email": "I can't log in, my password reset link isn't working.",
            "expected_category": "account",
            "expected_summary": "Customer cannot log in and their password reset link fails.",
            "expected_difficulty": "easy",
            "notes": "Fixture case for eval runner tests.",
        },
        {
            "id": "t-003",
            "email": "The checkout page throws a 500 error every time I submit an order.",
            "expected_category": "technical",
            "expected_summary": "Customer reports a 500 error on the checkout page.",
            "expected_difficulty": "medium",
            "notes": "Fixture case for eval runner tests.",
        },
    ],
}


def test_run_eval_end_to_end(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(FIXTURE_DATASET), encoding="utf-8")

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "classifier_v1.yaml"

    report = run_eval(prompt_path, dataset_path, MockClient())

    assert report.total == 3
    assert 0.0 <= report.category_accuracy <= 1.0
    assert len(report.results) == 3
    assert report.dataset_version == "test-fixture"
    # MockClient's keyword matching should get all three of these unambiguous examples right.
    assert report.category_accuracy == 1.0
    assert report.accuracy_by_difficulty == {"easy": 1.0, "medium": 1.0}
    assert report.accuracy_by_category == {"account": 1.0, "billing": 1.0, "technical": 1.0}
    assert report.avg_latency_ms is not None and report.avg_latency_ms >= 0.0
    assert report.avg_summary_judge_score is not None
    assert 1.0 <= report.avg_summary_judge_score <= 5.0
    assert report.total_tokens is not None and report.total_tokens > 0
    for result in report.results:
        assert result.error is None
        assert 0.0 <= result.summary_score <= 1.0
        assert result.difficulty in {"easy", "medium", "hard"}
        assert result.summary_judge_score is not None
        assert 1 <= result.summary_judge_score <= 5
        assert result.latency_ms is not None and result.latency_ms >= 0.0
        assert result.total_tokens is not None and result.total_tokens > 0


def test_run_eval_batches_with_bounded_concurrency(tmp_path: Path):
    """concurrency=1 (fully serial) should score identically to the default batched run."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(FIXTURE_DATASET), encoding="utf-8")
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "classifier_v1.yaml"

    serial_report = run_eval(prompt_path, dataset_path, MockClient(), concurrency=1)
    batched_report = run_eval(prompt_path, dataset_path, MockClient(), concurrency=10)

    assert serial_report.category_accuracy == batched_report.category_accuracy
    assert {r.id for r in serial_report.results} == {r.id for r in batched_report.results}


def test_run_eval_scoring_math(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(FIXTURE_DATASET), encoding="utf-8")
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "classifier_v1.yaml"

    report = run_eval(prompt_path, dataset_path, MockClient())

    expected_accuracy = sum(r.category_match for r in report.results) / report.total
    assert report.category_accuracy == round(expected_accuracy, 3)
