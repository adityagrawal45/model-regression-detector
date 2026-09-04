from datetime import datetime

import pytest
from pydantic import ValidationError

from regression_detector.config import ClassificationResult, PromptConfig


def test_prompt_config_valid():
    cfg = PromptConfig(
        version="v1",
        created_at=datetime(2026, 1, 1),
        model="llama-3.1-8b-instant",
        system_prompt="Classify the email.",
        few_shot_examples=[],
    )
    assert cfg.version == "v1"
    assert cfg.few_shot_examples == []


def test_prompt_config_rejects_bad_category_in_few_shot():
    with pytest.raises(ValidationError):
        PromptConfig(
            version="v1",
            created_at=datetime(2026, 1, 1),
            model="llama-3.1-8b-instant",
            system_prompt="Classify the email.",
            few_shot_examples=[
                {"email": "hi", "category": "not-a-real-category", "summary": "test"}
            ],
        )


def test_classification_result_valid():
    result = ClassificationResult(category="billing", summary="Customer wants a refund.")
    assert result.category == "billing"


def test_classification_result_rejects_bad_category():
    with pytest.raises(ValidationError):
        ClassificationResult(category="unknown", summary="test")


def test_classification_result_rejects_overlong_summary():
    with pytest.raises(ValidationError):
        ClassificationResult(category="billing", summary="x" * 201)
