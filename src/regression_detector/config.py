"""Typed contract for the email classifier feature.

`PromptConfig` is the "code" the eval pipeline runs against (loaded from a
versioned YAML file in /prompts). `ClassificationResult` is the structured
output every prompt version must produce, regardless of which model or
provider is behind it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


class FewShotExample(BaseModel):
    """One labeled example used to steer the model, stored inline in the prompt YAML."""

    email: str
    category: Category
    summary: str = Field(max_length=200)


class PromptConfig(BaseModel):
    """A single versioned prompt definition, as loaded from prompts/*.yaml."""

    version: str
    created_at: datetime
    model: str
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    """The structured output contract for the email classifier."""

    category: Category
    summary: str = Field(max_length=200)
