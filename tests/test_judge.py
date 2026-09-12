import asyncio

import pytest

from regression_detector.judge import JudgeError, judge_summary, judge_summary_async
from regression_detector.llm_client import LLMResponse, MockClient


def test_judge_summary_scores_close_match_highly():
    client = MockClient()
    score = judge_summary(
        client,
        email="I was charged twice for my subscription, please refund me.",
        expected_summary="Customer was double-charged and wants a refund.",
        candidate_summary="Customer was double-charged and wants a refund.",
    )
    assert 1 <= score <= 5
    assert score >= 4


def test_judge_summary_scores_unrelated_summary_low():
    client = MockClient()
    score = judge_summary(
        client,
        email="I was charged twice for my subscription, please refund me.",
        expected_summary="Customer was double-charged and wants a refund.",
        candidate_summary="The weather today is sunny with a light breeze.",
    )
    assert score <= 2


def test_judge_summary_empty_candidate_scores_one():
    client = MockClient()
    score = judge_summary(
        client,
        email="Some email",
        expected_summary="Expected thing",
        candidate_summary="",
    )
    assert score == 1


def test_judge_summary_raises_on_out_of_range_score():
    class BadJudgeClient:
        def complete_json(self, system_prompt, user_prompt, model):
            return LLMResponse(data={"score": 9})

    with pytest.raises(JudgeError):
        judge_summary(BadJudgeClient(), "email", "expected", "candidate")


def test_judge_summary_async_matches_sync():
    client = MockClient()
    sync_score = judge_summary(client, "email text", "expected summary here", "expected summary here")
    async_score = asyncio.run(
        judge_summary_async(client, "email text", "expected summary here", "expected summary here")
    )
    assert sync_score == async_score
