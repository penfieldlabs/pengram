# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests verifying vocabulary/relationships.json mirrors pengram.vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

from pengram import vocabulary as v

JSON_PATH = Path(__file__).resolve().parent.parent / "pengram" / "vocabulary" / "relationships.json"


def load_json() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


def test_json_file_exists() -> None:
    assert JSON_PATH.exists()


def test_semantic_types_match_python() -> None:
    data = load_json()
    json_types: set[str] = set()
    for group in data["semantic_types"].values():
        json_types.update(group["types"].keys())
    assert json_types == set(v.SEMANTIC_TYPES)


def test_semantic_categories_match_python() -> None:
    data = load_json()
    assert set(data["semantic_types"].keys()) == set(v.SEMANTIC_CATEGORIES.keys())
    for name, group in data["semantic_types"].items():
        assert set(group["types"].keys()) == set(v.SEMANTIC_CATEGORIES[name])


def test_structural_types_match_python() -> None:
    data = load_json()
    assert set(data["structural_types"]["types"].keys()) == set(v.STRUCTURAL_TYPES)


def test_confidence_labels_match_python() -> None:
    data = load_json()
    assert set(data["confidence_labels"].keys()) == v.CONFIDENCE_LABELS


def test_defaults_match_python() -> None:
    data = load_json()
    assert data["defaults"]["by_kind"] == dict(v.DEFAULTS)


def test_json_version_present() -> None:
    data = load_json()
    assert "version" in data
    assert isinstance(data["version"], str)
