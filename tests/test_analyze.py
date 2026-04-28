# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.analyze."""

from __future__ import annotations

import networkx as nx

from pengram.analyze import (
    analyze,
    bridge_nodes,
    god_nodes,
    suggested_questions,
    surprising_connections,
)


def test_god_nodes_identifies_hub() -> None:
    g = nx.Graph()
    g.add_node("hub", label="Hub")
    for i in range(5):
        name = f"leaf{i}"
        g.add_node(name, label=name)
        g.add_edge("hub", name)
    gods = god_nodes(g, top_n=1)
    assert gods[0]["id"] == "hub"


def test_god_nodes_empty_graph() -> None:
    assert god_nodes(nx.Graph()) == []


def test_bridge_nodes_cross_community() -> None:
    g = nx.Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    communities = {0: ["a", "b"], 1: ["c", "d"]}
    for node in g.nodes:
        g.nodes[node]["label"] = node
    bridges = bridge_nodes(g, communities)
    ids = {b["id"] for b in bridges}
    assert "b" in ids or "c" in ids


def test_bridge_nodes_empty_communities() -> None:
    g = nx.Graph()
    g.add_edge("a", "b")
    assert bridge_nodes(g, {}) == []


def test_surprising_connections_requires_cross_community() -> None:
    g = nx.Graph()
    g.add_node("a", label="A")
    g.add_node("b", label="B")
    g.add_node("c", label="C")
    g.add_edge("a", "b", relation="references", confidence="EXTRACTED")
    g.add_edge("b", "c", relation="references", confidence="EXTRACTED")
    communities = {0: ["a", "b"], 1: ["c"]}
    result = surprising_connections(g, communities)
    pairs = {(s["source"], s["target"]) for s in result}
    assert ("b", "c") in pairs or ("c", "b") in pairs


def test_surprising_connections_filters_low_confidence() -> None:
    g = nx.Graph()
    g.add_node("a", label="A")
    g.add_node("b", label="B")
    g.add_edge("a", "b", relation="references", confidence="AMBIGUOUS")
    communities = {0: ["a"], 1: ["b"]}
    assert surprising_connections(g, communities) == []


def test_suggested_questions_from_gods() -> None:
    gods = [{"id": "x", "label": "X-Topic"}]
    questions = suggested_questions(nx.Graph(), gods, [])
    assert any("X-Topic" in q for q in questions)


def test_suggested_questions_fallback() -> None:
    g = nx.Graph()
    g.add_node("only", label="Only")
    questions = suggested_questions(g, [], [])
    assert questions
    assert any("Only" in q for q in questions)


def test_analyze_returns_summary_keys() -> None:
    g = nx.DiGraph()
    g.add_node("a", label="A")
    g.add_node("b", label="B")
    g.add_edge("a", "b", relation="references", confidence="EXTRACTED")
    summary = analyze(g, {0: ["a"], 1: ["b"]})
    assert {
        "node_count",
        "edge_count",
        "community_count",
        "god_nodes",
        "bridge_nodes",
        "surprising_connections",
        "suggested_questions",
    }.issubset(summary.keys())
    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
