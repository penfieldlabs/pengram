# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.export_catalog."""

from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx

from pengram.export_catalog import export_catalog, to_rows


def _plain_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(
        "doc_notes",
        label="notes.md",
        kind="document",
        body="word " * 50,
    )
    g.add_node("concept_a", label="A", kind="concept")
    g.add_node("concept_b", label="B", kind="concept")
    g.add_edge("doc_notes", "concept_a", relation="references")
    g.add_edge("doc_notes", "concept_b", relation="references")
    return g


def test_base_columns_only_when_no_youtube_metadata() -> None:
    g = _plain_graph()
    columns, rows = to_rows(g)
    assert "video_id" not in columns
    assert "channel" not in columns
    # One row for the only content node.
    assert len(rows) == 1
    row = rows[0]
    assert row["filename"] == "notes.md"
    assert row["content_type"] == "document"
    assert row["word_count"] == 50
    assert row["entity_count"] == 2  # concept_a + concept_b
    # doc→concept_a + doc→concept_b + 0 incoming ⇒ 2.
    assert row["relationship_count"] == 2


def test_youtube_columns_included_when_any_node_has_video_id() -> None:
    g = _plain_graph()
    # Simulate a YouTube-sourced transcript node alongside the regular doc.
    g.add_node(
        "doc_vid",
        label="ep42.vtt",
        kind="transcript",
        body="hi " * 30,
        video_id="abc123",
        channel="Test Channel",
        upload_date="2026-04-19",
        duration=3600,
        view_count=1000,
        like_count=42,
    )
    columns, rows = to_rows(g)
    for col in ("video_id", "channel", "upload_date", "duration", "view_count", "like_count"):
        assert col in columns
    # Two content rows now.
    assert len(rows) == 2
    by_filename = {r["filename"]: r for r in rows}
    # The non-YouTube row still has the new columns (empty strings).
    assert by_filename["notes.md"]["video_id"] == ""
    # The YouTube row carries its metadata.
    assert by_filename["ep42.vtt"]["video_id"] == "abc123"
    assert by_filename["ep42.vtt"]["channel"] == "Test Channel"


def test_cluster_column_reflects_community_membership() -> None:
    g = _plain_graph()
    communities = {0: ["doc_notes"], 1: ["concept_a"]}
    _, rows = to_rows(g, communities)
    assert rows[0]["cluster"] == "0"


def test_non_content_nodes_are_excluded() -> None:
    g = _plain_graph()
    # Concepts and events shouldn't appear as catalog rows.
    _, rows = to_rows(g)
    assert all(r["content_type"] == "document" for r in rows)


def test_export_catalog_writes_csv(tmp_path: Path) -> None:
    g = _plain_graph()
    path = export_catalog(g, tmp_path)
    assert path.exists()
    assert path.name == "catalog.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        data = list(reader)
    assert "filename" in header
    assert len(data) == 1
    assert data[0]["filename"] == "notes.md"


def test_export_catalog_empty_graph(tmp_path: Path) -> None:
    g = nx.DiGraph()
    path = export_catalog(g, tmp_path)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Header present, no data rows.
    assert len(rows) == 1
    assert rows[0][0] == "filename"
