import asyncio
from datetime import datetime

import pytest

from regression_detector.classifier import (
    ClassificationError,
    ClassificationOutcome,
    classify_email,
    classify_email_detailed,
    classify_email_detailed_async,
)
from regression_detector.config import ClassificationResult, PromptConfig
from regression_detector.llm_client import LLMResponse, MockClient


@pytest.fixture
def prompt_config():
    return PromptConfig(
        version="v1",
        created_at=datetime(2026, 1, 1),
        model="llama-3.1-8b-instant",
        system_prompt="Classify the email.",
        few_shot_examples=[],
    )


def test_classify_email_returns_classification_result(prompt_config):
    result = classify_email(
        "I was charged twice for my subscription, please refund me.",
        prompt_config,
        MockClient(),
    )
    assert isinstance(result, ClassificationResult)
    assert result.category == "billing"
    assert result.summary


def test_classify_email_detects_technical_keywords(prompt_config):
    result = classify_email(
        "The API keeps returning a 500 error on every request.",
        prompt_config,
        MockClient(),
    )
    assert result.category == "technical"


def test_classify_email_falls_back_to_general(prompt_config):
    result = classify_email(
        "Just wanted to say thanks for the great support!",
        prompt_config,
        MockClient(),
    )
    assert result.category == "general"


def test_classify_email_raises_on_bad_response(prompt_config):
    class BadClient:
        def complete_json(self, system_prompt, user_prompt, model):
            return LLMResponse(data={"category": "not-a-category", "summary": "test"})

    with pytest.raises(ClassificationError):
        classify_email("some email", prompt_config, BadClient())


def test_classify_email_detailed_reports_latency_and_usage(prompt_config):
    outcome = classify_email_detailed(
        "I was charged twice for my subscription, please refund me.",
        prompt_config,
        MockClient(),
    )
    assert isinstance(outcome, ClassificationOutcome)
    assert outcome.result.category == "billing"
    assert outcome.latency_ms >= 0.0
    assert outcome.usage is not None
    assert outcome.usage.prompt_tokens > 0
    assert outcome.usage.completion_tokens > 0
    assert outcome.usage.total_tokens == outcome.usage.prompt_tokens + outcome.usage.completion_tokens


def test_classify_email_detailed_async_matches_sync(prompt_config):
    email = "The API keeps returning a 500 error on every request."
    sync_outcome = classify_email_detailed(email, prompt_config, MockClient())
    async_outcome = asyncio.run(classify_email_detailed_async(email, prompt_config, MockClient()))

    assert sync_outcome.result == async_outcome.result
