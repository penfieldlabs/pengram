# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.validate."""

from __future__ import annotations

from pengram.validate import validate_extraction


def test_valid_extraction() -> None:
    data = {
        "nodes": [
            {"id": "alice", "label": "Alice", "confidence": "EXTRACTED"},
            {"id": "bob", "label": "Bob"},
        ],
        "edges": [
            {
                "source": "alice",
                "target": "bob",
                "relation": "references",
                "confidence": "EXTRACTED",
            },
        ],
    }
    assert validate_extraction(data) == []


def test_not_a_mapping() -> None:
    errors = validate_extraction(["not", "a", "dict"])
    assert any("object" in e for e in errors)


def test_missing_nodes() -> None:
    errors = validate_extraction({"edges": []})
    assert any("nodes" in e for e in errors)


def test_missing_edges() -> None:
    errors = validate_extraction({"nodes": []})
    assert any("edges" in e for e in errors)


def test_node_missing_id() -> None:
    errors = validate_extraction(
        {
            "nodes": [{"label": "nameless"}],
            "edges": [],
        }
    )
    assert any("id" in e for e in errors)


def test_node_missing_label() -> None:
    errors = validate_extraction(
        {
            "nodes": [{"id": "x"}],
            "edges": [],
        }
    )
    assert any("label" in e for e in errors)


def test_node_invalid_confidence() -> None:
    errors = validate_extraction(
        {
            "nodes": [{"id": "x", "label": "X", "confidence": "MAYBE"}],
            "edges": [],
        }
    )
    assert any("confidence" in e for e in errors)


def test_edge_missing_source() -> None:
    errors = validate_extraction(
        {
            "nodes": [],
            "edges": [{"target": "x", "relation": "references"}],
        }
    )
    assert any("source" in e for e in errors)


def test_edge_missing_target() -> None:
    errors = validate_extraction(
        {
            "nodes": [],
            "edges": [{"source": "x", "relation": "references"}],
        }
    )
    assert any("target" in e for e in errors)


def test_edge_missing_relation() -> None:
    errors = validate_extraction(
        {
            "nodes": [],
            "edges": [{"source": "a", "target": "b"}],
        }
    )
    assert any("relation" in e for e in errors)


def test_edge_invalid_relation() -> None:
    errors = validate_extraction(
        {
            "nodes": [],
            "edges": [{"source": "a", "target": "b", "relation": "does_a_flip"}],
        }
    )
    assert any("relation" in e and "does_a_flip" in e for e in errors)


def test_edge_structural_relation_ok() -> None:
    errors = validate_extraction(
        {
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b", "relation": "calls"}],
        }
    )
    assert errors == []


def test_edge_invalid_confidence() -> None:
    errors = validate_extraction(
        {
            "nodes": [],
            "edges": [
                {"source": "a", "target": "b", "relation": "calls", "confidence": "probably"}
            ],
        }
    )
    assert any("confidence" in e for e in errors)


def test_non_object_node() -> None:
    errors = validate_extraction({"nodes": ["a"], "edges": []})
    assert any("nodes[0]" in e for e in errors)


def test_non_object_edge() -> None:
    errors = validate_extraction({"nodes": [], "edges": ["a"]})
    assert any("edges[0]" in e for e in errors)


def test_empty_extraction_valid() -> None:
    assert validate_extraction({"nodes": [], "edges": []}) == []
