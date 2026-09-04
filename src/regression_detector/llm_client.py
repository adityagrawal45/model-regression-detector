"""LLM client abstraction.

`classifier.py` and `eval_runner.py` depend only on the `LLMClient` protocol,
never on a specific provider's SDK, so swapping providers later (OpenAI,
Anthropic, a different Groq model, ...) is a one-line change at the call site.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient(Protocol):
    """Anything that can turn a system + user prompt into a parsed JSON dict."""

    def complete_json(self, system_prompt: str, user_prompt: str, model: str) -> dict:
        ...


class GroqClient:
    """Calls Groq's OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_key: str | None = None) -> None:
        # Imported lazily so environments without a key (and without `openai`
        # installed for some reason) can still use MockClient.
        from openai import OpenAI

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it in your environment or .env file, "
                "or use MockClient for offline runs."
            )
        self._client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)

    def complete_json(self, system_prompt: str, user_prompt: str, model: str) -> dict:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        return json.loads(content)


class MockClient:
    """Deterministic, offline stand-in for testing and demoing without an API key.

    Uses simple keyword matching against the email text so behavior is
    predictable in tests and in `--mock` eval runs.
    """

    _KEYWORDS: dict[str, list[str]] = {
        "billing": ["charge", "invoice", "refund", "payment", "subscription", "billed", "price"],
        "technical": ["error", "bug", "crash", "api", "500", "not working", "broken", "integration"],
        "account": ["login", "log in", "password", "locked out", "2fa", "account access", "reset"],
    }

    def complete_json(self, system_prompt: str, user_prompt: str, model: str) -> dict:
        # classifier.py appends any few-shot examples before the actual email
        # being classified, each as its own "Email: ...\nResponse: ..." block.
        # The target email is always the last "Email:" block, with no
        # "Response:" after it - pull that one out so few-shot example
        # content doesn't leak into the classification.
        matches = re.findall(r"Email:\s*(.*?)(?=\nResponse:|\Z)", user_prompt, re.DOTALL)
        email_text = matches[-1].strip() if matches else user_prompt.strip()

        text = email_text.lower()
        category = "general"
        for cat, keywords in self._KEYWORDS.items():
            if any(kw in text for kw in keywords):
                category = cat
                break

        first_line = email_text.splitlines()[0].strip()
        summary = first_line[:197] + "..." if len(first_line) > 200 else first_line

        return {"category": category, "summary": summary or "No summary available."}
