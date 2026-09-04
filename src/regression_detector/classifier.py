"""The LLM feature under test: classify a customer support email.

Input: raw email text. Output: a structured ClassificationResult (category +
one-sentence summary). The prompt is fully configurable via `PromptConfig` so
the eval pipeline can run the same function against different prompt
versions/models.
"""

from __future__ import annotations

from pydantic import ValidationError

from regression_detector.config import ClassificationResult, PromptConfig
from regression_detector.llm_client import LLMClient


class ClassificationError(Exception):
    """Raised when the model's response can't be parsed into ClassificationResult."""


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


def classify_email(
    email_text: str,
    prompt_config: PromptConfig,
    client: LLMClient,
) -> ClassificationResult:
    """Classify a single email using the given prompt version and LLM client."""
    user_prompt = _build_user_prompt(email_text, prompt_config)

    raw = client.complete_json(
        system_prompt=prompt_config.system_prompt,
        user_prompt=user_prompt,
        model=prompt_config.model,
    )

    try:
        return ClassificationResult.model_validate(raw)
    except ValidationError as exc:
        raise ClassificationError(
            f"Model response did not match ClassificationResult schema: {raw!r}\n{exc}"
        ) from exc
