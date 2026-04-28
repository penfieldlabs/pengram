# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.build."""

from __future__ import annotations

import networkx as nx
import pytest

from pengram.build import build, build_from_extraction, normalize_id


def test_normalize_id_basic() -> None:
    assert normalize_id("Foo/Bar.py") == "foo_bar_py"


def test_normalize_id_preserves_alphanumeric() -> None:
    assert normalize_id("ABC123") == "abc123"


def test_build_from_extraction_basic() -> None:
    data = {
        "nodes": [
            {"id": "A", "label": "A"},
            {"id": "B", "label": "B"},
        ],
        "edges": [
            {"source": "A", "target": "B", "relation": "references", "confidence": "EXTRACTED"},
        ],
    }
    g = build_from_extraction(data)
    assert isinstance(g, nx.DiGraph)
    assert "a" in g and "b" in g
    assert g.has_edge("a", "b")
    assert g["a"]["b"]["relation"] == "references"


def test_build_from_extraction_drops_dangling() -> None:
    data = {
        "nodes": [{"id": "A", "label": "A"}],
        "edges": [{"source": "A", "target": "ghost", "relation": "references"}],
    }
    g = build_from_extraction(data)
    assert not g.has_edge("a", "ghost")


def test_build_from_extraction_invalid_raises() -> None:
    with pytest.raises(ValueError):
        build_from_extraction({"nodes": [{"id": "A"}], "edges": []})  # missing label


def test_build_merges_duplicate_nodes() -> None:
    a = {
        "nodes": [{"id": "X", "label": "X", "kind": "file"}],
        "edges": [],
    }
    b = {
        "nodes": [{"id": "X", "label": "X", "kind": "class", "extra": 1}],
        "edges": [],
    }
    g = build([a, b])
    assert g.nodes["x"]["kind"] == "class"
    assert g.nodes["x"]["extra"] == 1


def test_build_drops_dangling_edges() -> None:
    a = {
        "nodes": [{"id": "X", "label": "X"}],
        "edges": [{"source": "X", "target": "Y", "relation": "references"}],
    }
    g = build([a])
    assert not g.has_edge("x", "y")


def test_build_deduplicates_edges() -> None:
    a = {
        "nodes": [
            {"id": "A", "label": "A"},
            {"id": "B", "label": "B"},
        ],
        "edges": [
            {"source": "A", "target": "B", "relation": "references", "confidence": "EXTRACTED"},
            {"source": "A", "target": "B", "relation": "references", "confidence": "INFERRED"},
        ],
    }
    g = build([a])
    assert len(g.edges) == 1
    assert g["a"]["b"]["confidence"] == "INFERRED"


def test_build_preserves_direction() -> None:
    a = {
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
        "edges": [{"source": "A", "target": "B", "relation": "causes"}],
    }
    g = build([a])
    assert g.has_edge("a", "b")
    assert not g.has_edge("b", "a")


def test_build_undirected() -> None:
    a = {
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
        "edges": [{"source": "A", "target": "B", "relation": "sibling_of"}],
    }
    g = build([a], directed=False)
    assert isinstance(g, nx.Graph) and not isinstance(g, nx.DiGraph)
    assert g.has_edge("a", "b")


def test_build_empty() -> None:
    g = build([])
    assert len(g.nodes) == 0
    assert len(g.edges) == 0


def test_build_warns_when_overwriting_edge_relation(capsys) -> None:
    """Different relations on the same (src, tgt) pair must warn via _ui."""
    a = {
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
        "edges": [
            {"source": "A", "target": "B", "relation": "supports", "confidence": "EXTRACTED"},
            {"source": "A", "target": "B", "relation": "contradicts", "confidence": "EXTRACTED"},
        ],
    }
    g = build([a])
    # Last write wins (contradicts overwrites supports).
    assert g["a"]["b"]["relation"] == "contradicts"
    # And we warned about the overwrite.
    assert "overwriting edge" in capsys.readouterr().out
