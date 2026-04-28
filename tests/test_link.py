# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.link."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pengram import vocabulary as v
from pengram.link import Entity, link_all, link_entities
from pengram.llm import LLMError


def _source(kind: str = "concept") -> Entity:
    return Entity(id="src", name="Source", kind=kind, context="ctx")


def _targets() -> list[Entity]:
    return [
        Entity(id="t1", name="Target One", kind="concept"),
        Entity(id="t2", name="Target Two", kind="concept"),
    ]


def test_link_entities_valid_relation() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "links": [
                    {
                        "target_index": 0,
                        "relation": "supports",
                        "confidence": "EXTRACTED",
                        "reason": "x",
                    },
                    {
                        "target_index": 1,
                        "relation": "references",
                        "confidence": "INFERRED",
                        "reason": "y",
                    },
                ]
            }
        )

    decisions = link_entities(_source(), _targets(), llm=fake_llm)
    assert len(decisions) == 2
    assert decisions[0].relation == "supports"
    assert decisions[0].confidence == "EXTRACTED"
    assert decisions[1].relation == "references"


def test_link_entities_invalid_relation_falls_back() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "links": [
                    {
                        "target_index": 0,
                        "relation": "fakes",
                        "confidence": "EXTRACTED",
                        "reason": "x",
                    },
                ]
            }
        )

    decisions = link_entities(_source("concept"), _targets()[:1], llm=fake_llm)
    assert decisions[0].relation == v.DEFAULTS["concept"]
    assert decisions[0].confidence == v.CONFIDENCE_AMBIGUOUS


def test_link_entities_invalid_confidence_coerced() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "links": [
                    {
                        "target_index": 0,
                        "relation": "supports",
                        "confidence": "high",
                        "reason": "x",
                    },
                ]
            }
        )

    decisions = link_entities(_source(), _targets()[:1], llm=fake_llm)
    assert decisions[0].confidence == v.CONFIDENCE_INFERRED


def test_link_entities_llm_failure_uses_defaults() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        raise LLMError("offline")

    decisions = link_entities(_source("category"), _targets(), llm=fake_llm)
    assert all(d.relation == v.DEFAULTS["category"] for d in decisions)
    assert all(d.confidence == v.CONFIDENCE_AMBIGUOUS for d in decisions)


def test_link_entities_missing_link_entry_gets_default() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "links": [
                    {"target_index": 0, "relation": "supports", "confidence": "EXTRACTED"},
                ]
            }
        )

    decisions = link_entities(_source(), _targets(), llm=fake_llm)
    assert decisions[0].relation == "supports"
    assert decisions[1].confidence == v.CONFIDENCE_AMBIGUOUS


def test_link_entities_empty_targets() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        raise AssertionError("should not be called")

    assert link_entities(_source(), [], llm=fake_llm) == []


def test_link_all_caches_per_source(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps(
            {"links": [{"target_index": 0, "relation": "supports", "confidence": "EXTRACTED"}]}
        )

    pairs = [(_source(), _targets()[:1])]
    link_all(pairs, output_dir=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 1
    # Second run should hit the cache.
    link_all(pairs, output_dir=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 1


def test_link_all_writes_cache_json(tmp_path: Path) -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {"links": [{"target_index": 0, "relation": "supports", "confidence": "EXTRACTED"}]}
        )

    link_all([(_source(), _targets()[:1])], output_dir=tmp_path, workers=1, llm=fake_llm)
    files = list((tmp_path / "links").glob("*.json"))
    assert files
    data = json.loads(files[0].read_text())
    assert data[0]["relation"] == "supports"
