"""Loading versioned prompt YAML files into `PromptConfig`."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from regression_detector.config import PromptConfig


class PromptLoadError(Exception):
    """Raised when a prompt YAML file is missing or doesn't match PromptConfig."""


def load_prompt_config(path: str | Path) -> PromptConfig:
    path = Path(path)
    if not path.exists():
        raise PromptLoadError(f"Prompt file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"Invalid YAML in {path}: {exc}") from exc

    try:
        return PromptConfig.model_validate(raw)
    except ValidationError as exc:
        raise PromptLoadError(f"Prompt file {path} does not match PromptConfig schema:\n{exc}") from exc
