"""LLM client abstraction.

`classifier.py` and `eval_runner.py` depend only on the `LLMClient` protocol,
never on a specific provider's SDK, so swapping providers later (OpenAI,
Anthropic, a different Groq model, ...) is a one-line change at the call site.

Every call returns an `LLMResponse` (parsed JSON body + token usage, when the
provider reports it) rather than a bare dict, so the eval pipeline can score
token usage per test case alongside category/summary correctness. Both a sync
and an async entry point are required so the eval runner can batch requests
with `asyncio.gather` instead of running the golden dataset serially.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from pydantic import BaseModel

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Sentinel marker only ever present in judge-style prompts (see judge.py), so
# MockClient can special-case them without needing to know about the judge
# module (avoids a circular import).
JUDGE_PROMPT_MARKER = "Candidate summary:"


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(BaseModel):
    """A single completion: the parsed JSON body plus token usage, if known."""

    data: dict
    usage: TokenUsage | None = None


class LLMClient(Protocol):
    """Anything that can turn a system + user prompt into a parsed JSON response."""

    def complete_json(self, system_prompt: str, user_prompt: str, model: str) -> LLMResponse:
        ...

    async def acomplete_json(self, system_prompt: str, user_prompt: str, model: str) -> LLMResponse:
        ...


class GroqClient:
    """Calls Groq's OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_key: str | None = None) -> None:
        # Imported lazily so environments without a key (and without `openai`
        # installed for some reason) can still use MockClient.
        from openai import AsyncOpenAI, OpenAI

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it in your environment or .env file, "
                "or use MockClient for offline runs."
            )
        self._client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        self._async_client = AsyncOpenAI(api_key=key, base_url=GROQ_BASE_URL)

    @staticmethod
    def _to_response(response) -> LLMResponse:
        content = response.choices[0].message.content
        usage = None
        if response.usage is not None:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return LLMResponse(data=json.loads(content), usage=usage)

    def complete_json(self, system_prompt: str, user_prompt: str, model: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return self._to_response(response)

    async def acomplete_json(self, system_prompt: str, user_prompt: str, model: str) -> LLMResponse:
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return self._to_response(response)


class MockClient:
    """Deterministic, offline stand-in for testing and demoing without an API key.

    Uses simple keyword matching against the email text so behavior is
    predictable in tests and in `--mock` eval runs. Also answers judge-style
    prompts (see judge.py) with a keyword-overlap heuristic scaled to 1-5, so
    the whole pipeline - including LLM-as-judge scoring - stays demoable
    offline.
    """

    _KEYWORDS: dict[str, list[str]] = {
        "billing": ["charge", "invoice", "refund", "payment", "subscription", "billed", "price"],
        "technical": ["error", "bug", "crash", "api", "500", "not working", "broken", "integration"],
        "account": ["login", "log in", "password", "locked out", "2fa", "account access", "reset"],
    }

    def complete_json(self, system_prompt: str, user_prompt: str, model: str) -> LLMResponse:
        if JUDGE_PROMPT_MARKER in user_prompt:
            data = self._judge(user_prompt)
        else:
            data = self._classify(user_prompt)
        usage = TokenUsage(
            prompt_tokens=len(user_prompt.split()),
            completion_tokens=len(json.dumps(data).split()),
            total_tokens=len(user_prompt.split()) + len(json.dumps(data).split()),
        )
        return LLMResponse(data=data, usage=usage)

    async def acomplete_json(self, system_prompt: str, user_prompt: str, model: str) -> LLMResponse:
        # No real I/O to await; keep the same behavior as the sync path so
        # concurrent batched runs and serial runs score identically.
        return self.complete_json(system_prompt, user_prompt, model)

    def _classify(self, user_prompt: str) -> dict:
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

    @staticmethod
    def _judge(user_prompt: str) -> dict:
        expected_match = re.search(r"Expected summary:\s*(.*)", user_prompt)
        candidate_match = re.search(r"Candidate summary:\s*(.*)", user_prompt)
        expected = expected_match.group(1).strip() if expected_match else ""
        candidate = candidate_match.group(1).strip() if candidate_match else ""

        if not candidate:
            return {"score": 1, "reasoning": "No candidate summary provided."}

        stopwords = {"the", "a", "an", "to", "for", "and", "or", "is", "was", "of", "on", "in", "with"}
        expected_words = {w.strip(".,!?").lower() for w in expected.split()} - stopwords
        candidate_words = {w.strip(".,!?").lower() for w in candidate.split()} - stopwords

        if not expected_words:
            score = 5
        else:
            overlap = len(expected_words & candidate_words) / len(expected_words)
            # Map overlap fraction onto a 1-5 scale instead of clamping to the
            # extremes, so partial matches land in the middle of the range.
            score = max(1, min(5, round(1 + overlap * 4)))

        return {"score": score, "reasoning": f"Keyword overlap heuristic scored this {score}/5."}
