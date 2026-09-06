import json
from pathlib import Path

import pytest

from regression_detector.dataset import DatasetLoadError, load_golden_dataset

VALID_DATASET = {
    "version": "test-v1",
    "created_at": "2026-01-01T00:00:00Z",
    "cases": [
        {
            "id": "t-001",
            "email": "I was charged twice, please refund me.",
            "expected_category": "billing",
            "expected_summary": "Customer was double-charged and wants a refund.",
            "expected_difficulty": "easy",
            "notes": "Fixture case.",
        },
    ],
}


def test_load_golden_dataset_valid(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(VALID_DATASET), encoding="utf-8")

    dataset = load_golden_dataset(dataset_path)

    assert dataset.version == "test-v1"
    assert len(dataset.cases) == 1
    assert dataset.cases[0].expected_category == "billing"
    assert dataset.cases[0].expected_difficulty == "easy"


def test_load_golden_dataset_missing_file(tmp_path: Path):
    with pytest.raises(DatasetLoadError, match="not found"):
        load_golden_dataset(tmp_path / "does_not_exist.json")


def test_load_golden_dataset_invalid_json(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Invalid JSON"):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_rejects_bad_category(tmp_path: Path):
    bad = json.loads(json.dumps(VALID_DATASET))
    bad["cases"][0]["expected_category"] = "not-a-real-category"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(DatasetLoadError):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_rejects_bad_difficulty(tmp_path: Path):
    bad = json.loads(json.dumps(VALID_DATASET))
    bad["cases"][0]["expected_difficulty"] = "impossible"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(DatasetLoadError):
        load_golden_dataset(dataset_path)


def test_load_golden_dataset_rejects_duplicate_ids(tmp_path: Path):
    dupe = json.loads(json.dumps(VALID_DATASET))
    dupe["cases"].append(dict(dupe["cases"][0]))  # same id twice
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dupe), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Duplicate"):
        load_golden_dataset(dataset_path)


def test_load_real_golden_dataset():
    """Sanity check that data/golden_dataset.json itself stays valid and sizable."""
    path = Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"

    dataset = load_golden_dataset(path)

    assert len(dataset.cases) >= 50
    difficulties = {c.expected_difficulty for c in dataset.cases}
    assert difficulties == {"easy", "medium", "hard"}
    categories = {c.expected_category for c in dataset.cases}
    assert categories == {"billing", "technical", "account", "general"}
