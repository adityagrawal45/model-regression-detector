"""The LLM feature under test: classify a customer support email.

Input: raw email text. Output: a structured ClassificationResult (category +
one-sentence summary). The prompt is fully configurable via `PromptConfig` so
the eval pipeline can run the same function against different prompt
versions/models.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ValidationError

from regression_detector.config import ClassificationResult, PromptConfig
from regression_detector.llm_client import LLMClient, TokenUsage


class ClassificationError(Exception):
    """Raised when the model's response can't be parsed into ClassificationResult."""


class ClassificationOutcome(BaseModel):
    """A classification result plus the request-level metrics the eval pipeline scores."""

    result: ClassificationResult
    latency_ms: float
    usage: TokenUsage | None = None


def _build_user_prompt(email_text: str, prompt_config: PromptConfig) -> str:
    parts: list[str] = []
    if prompt_config.few_shot_examples:
        parts.append("Examples:")
        for ex in prompt_config.few_shot_examples:
            parts.append(
                f'Email: {ex.email.strip()}\n'
                f'Response: {{"category": "{ex.category}", "summary": "{ex.summary}"}}'
            )
        parts.append("")
    parts.append(f"Email: {email_text.strip()}")
    return "\n".join(parts)


def _parse(raw: dict) -> ClassificationResult:
    try:
        return ClassificationResult.model_validate(raw)
    except ValidationError as exc:
        raise ClassificationError(
            f"Model response did not match ClassificationResult schema: {raw!r}\n{exc}"
        ) from exc


def classify_email_detailed(
    email_text: str,
    prompt_config: PromptConfig,
    client: LLMClient,
) -> ClassificationOutcome:
    """Classify a single email, also capturing latency and token usage. Sync."""
    user_prompt = _build_user_prompt(email_text, prompt_config)

    start = time.perf_counter()
    response = client.complete_json(
        system_prompt=prompt_config.system_prompt,
        user_prompt=user_prompt,
        model=prompt_config.model,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return ClassificationOutcome(result=_parse(response.data), latency_ms=latency_ms, usage=response.usage)


async def classify_email_detailed_async(
    email_text: str,
    prompt_config: PromptConfig,
    client: LLMClient,
) -> ClassificationOutcome:
    """Classify a single email, also capturing latency and token usage. Async, for batching."""
    user_prompt = _build_user_prompt(email_text, prompt_config)

    start = time.perf_counter()
    response = await client.acomplete_json(
        system_prompt=prompt_config.system_prompt,
        user_prompt=user_prompt,
        model=prompt_config.model,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return ClassificationOutcome(result=_parse(response.data), latency_ms=latency_ms, usage=response.usage)


def classify_email(
    email_text: str,
    prompt_config: PromptConfig,
    client: LLMClient,
) -> ClassificationResult:
    """Classify a single email using the given prompt version and LLM client."""
    return classify_email_detailed(email_text, prompt_config, client).result
