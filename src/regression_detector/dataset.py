"""Typed contract + loader for the golden dataset.

The golden dataset is hand-curated, human-verified ground truth (never
LLM-generated) that the eval pipeline runs the classifier against. It's
versioned as a whole (`GoldenDataset.version`) so a change to the eval bar
itself - adding cases, relabeling one, retiring an ambiguous case - is a
visible, trackable event, distinct from a prompt version change.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from regression_detector.config import Category

Difficulty = Literal["easy", "medium", "hard"]


class GoldenExample(BaseModel):
    """One hand-labeled test case in the golden dataset."""

    id: str
    email: str
    expected_category: Category
    expected_summary: str = Field(max_length=200)
    expected_difficulty: Difficulty
    notes: str


class GoldenDataset(BaseModel):
    """The versioned golden dataset, as loaded from data/golden_dataset.json."""

    version: str
    created_at: datetime
    cases: list[GoldenExample] = Field(default_factory=list)

    @field_validator("cases")
    @classmethod
    def _ids_must_be_unique(cls, cases: list[GoldenExample]) -> list[GoldenExample]:
        seen = set()
        for case in cases:
            if case.id in seen:
                raise ValueError(f"Duplicate golden dataset id: {case.id!r}")
            seen.add(case.id)
        return cases


class DatasetLoadError(Exception):
    """Raised when the golden dataset file is missing or doesn't match GoldenDataset."""


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(f"Invalid JSON in {path}: {exc}") from exc

    try:
        return GoldenDataset.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError or the uniqueness check above
        raise DatasetLoadError(f"Dataset file {path} does not match GoldenDataset schema:\n{exc}") from exc
