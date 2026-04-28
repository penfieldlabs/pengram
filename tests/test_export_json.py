# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.export_json."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import networkx as nx

from pengram.export_json import export_json, to_dict


def _make_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("a", label="Alice", kind="concept")
    g.add_node("b", label="Bob", kind="concept")
    g.add_edge("a", "b", relation="references", confidence="EXTRACTED")
    return g


def test_to_dict_structure() -> None:
    g = _make_graph()
    out = to_dict(g, {0: ["a", "b"]}, {"node_count": 2})
    assert out["meta"]["nodes"] == 2
    assert out["meta"]["edges"] == 1
    assert out["meta"]["directed"] is True
    assert len(out["nodes"]) == 2
    assert out["edges"][0]["relation"] == "references"
    assert out["communities"]["0"] == ["a", "b"]
    assert out["analysis"]["node_count"] == 2


def test_export_json_writes_file(tmp_path: Path) -> None:
    g = _make_graph()
    out = export_json(g, tmp_path)
    assert out.exists()
    assert out.name == "graph.json"
    data = json.loads(out.read_text())
    assert data["meta"]["nodes"] == 2
    assert data["meta"]["version"]


def test_export_json_includes_all_nodes_and_edges(tmp_path: Path) -> None:
    g = _make_graph()
    out = export_json(g, tmp_path, communities={0: ["a", "b"]})
    data = json.loads(out.read_text())
    ids = {n["id"] for n in data["nodes"]}
    assert ids == {"a", "b"}
    assert data["edges"][0]["source"] == "a"
    assert data["edges"][0]["target"] == "b"


def test_export_json_empty_graph(tmp_path: Path) -> None:
    g = nx.DiGraph()
    out = export_json(g, tmp_path)
    data = json.loads(out.read_text())
    assert data["meta"]["nodes"] == 0
    assert data["nodes"] == []
    assert data["edges"] == []


def test_export_json_undirected(tmp_path: Path) -> None:
    g = nx.Graph()
    g.add_node("x", label="X")
    out = export_json(g, tmp_path)
    data = json.loads(out.read_text())
    assert data["meta"]["directed"] is False


def test_to_dict_edges_have_relationship_key() -> None:
    g = _make_graph()
    out = to_dict(g)
    for edge in out["edges"]:
        assert "relationship" in edge, "edges must include 'relationship' key"
        assert edge["relationship"] == edge["relation"]


def test_to_dict_preserves_existing_relationship_key() -> None:
    g = nx.DiGraph()
    g.add_node("a", label="A")
    g.add_node("b", label="B")
    g.add_edge("a", "b", relation="supports", relationship="custom_val")
    out = to_dict(g)
    assert out["edges"][0]["relationship"] == "custom_val"


def test_export_json_handles_date_objects(tmp_path: Path) -> None:
    """Regression: datetime.date in node attrs must not crash json.dump."""
    g = nx.DiGraph()
    g.add_node("a", label="A", upload_date=datetime.date(2025, 12, 1))
    out = export_json(g, tmp_path)
    data = json.loads(out.read_text())
    assert data["nodes"][0]["upload_date"] == "2025-12-01"


def test_export_json_handles_path_objects(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("a", label="A", source_path=Path("/tmp/file.md"))
    out = export_json(g, tmp_path)
    data = json.loads(out.read_text())
    assert data["nodes"][0]["source_path"] == "/tmp/file.md"
