# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.report."""

from __future__ import annotations

import networkx as nx

from pengram.analyze import analyze
from pengram.report import render_report


def _make_graph() -> tuple[nx.DiGraph, dict[int, list[str]], dict]:
    g = nx.DiGraph()
    g.add_node("a", label="Alice")
    g.add_node("b", label="Bob")
    g.add_node("c", label="Carol")
    g.add_edge("a", "b", relation="references", confidence="EXTRACTED")
    g.add_edge("b", "c", relation="references", confidence="EXTRACTED")
    communities = {0: ["a", "b"], 1: ["c"]}
    return g, communities, analyze(g, communities)


def test_report_contains_summary_sections() -> None:
    g, communities, analysis = _make_graph()
    report = render_report(g, analysis, communities)
    for section in (
        "# Graph Report",
        "## Summary",
        "## God Nodes",
        "## Bridge Nodes",
        "## Communities",
        "## Surprising Connections",
        "## Suggested Questions",
    ):
        assert section in report


def test_report_includes_counts() -> None:
    g, communities, analysis = _make_graph()
    report = render_report(g, analysis, communities)
    assert "Nodes: **3**" in report
    assert "Edges: **2**" in report
    assert "Communities: **2**" in report


def test_report_empty_sections_are_marked() -> None:
    g = nx.Graph()
    analysis = {
        "node_count": 0,
        "edge_count": 0,
        "community_count": 0,
        "god_nodes": [],
        "bridge_nodes": [],
        "surprising_connections": [],
        "suggested_questions": [],
    }
    report = render_report(g, analysis, {})
    assert "_None._" in report


def test_report_lists_communities_with_previews() -> None:
    g, communities, analysis = _make_graph()
    report = render_report(g, analysis, communities)
    assert "Community 0" in report
    assert "Alice" in report


def test_report_is_valid_markdown_shape() -> None:
    g, communities, analysis = _make_graph()
    report = render_report(g, analysis, communities)
    # Table headers must have matching separator lines (4 columns in God Nodes).
    assert "| Label | Degree | Betweenness | Score |" in report
    assert "| --- | --- | --- | --- |" in report


def test_report_omits_single_node_communities() -> None:
    g = nx.DiGraph()
    g.add_node("a", label="Alice")
    g.add_node("b", label="Bob")
    g.add_node("c", label="Carol")
    g.add_edge("a", "b", relation="references")
    communities = {0: ["a", "b"], 1: ["c"]}
    analysis = analyze(g, communities)
    report = render_report(g, analysis, communities)
    assert "Community 0" in report
    assert "Community 1" not in report
    assert "1 single-node communities omitted" in report
