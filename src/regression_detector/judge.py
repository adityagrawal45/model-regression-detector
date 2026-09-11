"""LLM-as-judge scoring for summary relevance.

Category match is binary and needs no judgment call, but "is this summary a
good summary of the email" is inherently a matter of degree - so instead of a
brittle keyword-overlap proxy, we ask an LLM to rate the candidate summary
against the human-written expected summary on a 1-5 scale.
"""

from __future__ import annotations

from regression_detector.llm_client import LLMClient

JUDGE_SYSTEM_PROMPT = """\
You are grading a customer-support email summarizer. You will be given the
original email, a human-written expected summary, and a candidate summary
produced by the model under test.

Rate how well the candidate summary captures the same issue/request as the
expected summary, on a scale of 1 to 5:
  5 = captures the same core issue/request, no meaningful information lost
  4 = captures the core issue with a minor omission or imprecision
  3 = partially correct, missing or distorting a notable part of the issue
  2 = mostly misses the point, only tangentially related
  1 = unrelated to, or contradicts, the actual issue

Respond with ONLY a JSON object: {"score": <1-5 integer>, "reasoning": "<one sentence>"}.
"""

JUDGE_MODEL_FALLBACK = "llama-3.1-8b-instant"


class JudgeError(Exception):
    """Raised when the judge's response can't be parsed into a 1-5 score."""


def _build_judge_prompt(email: str, expected_summary: str, candidate_summary: str) -> str:
    return (
        f"Email: {email.strip()}\n"
        f"Expected summary: {expected_summary.strip()}\n"
        f"Candidate summary: {candidate_summary.strip()}"
    )


def _parse_score(data: dict) -> int:
    try:
        score = int(data["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgeError(f"Judge response missing a usable 'score': {data!r}") from exc
    if not 1 <= score <= 5:
        raise JudgeError(f"Judge score {score} out of the 1-5 range: {data!r}")
    return score


def judge_summary(
    client: LLMClient,
    email: str,
    expected_summary: str,
    candidate_summary: str,
    model: str = JUDGE_MODEL_FALLBACK,
) -> int:
    """Rate a candidate summary 1-5 against the expected summary. Sync."""
    if not candidate_summary:
        return 1
    response = client.complete_json(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=_build_judge_prompt(email, expected_summary, candidate_summary),
        model=model,
    )
    return _parse_score(response.data)


async def judge_summary_async(
    client: LLMClient,
    email: str,
    expected_summary: str,
    candidate_summary: str,
    model: str = JUDGE_MODEL_FALLBACK,
) -> int:
    """Rate a candidate summary 1-5 against the expected summary. Async, for batching."""
    if not candidate_summary:
        return 1
    response = await client.acomplete_json(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=_build_judge_prompt(email, expected_summary, candidate_summary),
        model=model,
    )
    return _parse_score(response.data)


__all__ = [
    "JudgeError",
    "JUDGE_SYSTEM_PROMPT",
    "judge_summary",
    "judge_summary_async",
]
