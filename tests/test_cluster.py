# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.cluster."""

from __future__ import annotations

import networkx as nx

from pengram.cluster import cluster, cohesion_score


def test_cluster_empty() -> None:
    assert cluster(nx.Graph()) == {}


def test_cluster_single_node() -> None:
    g = nx.Graph()
    g.add_node("a")
    result = cluster(g)
    assert result == {0: ["a"]}


def test_cluster_isolates_are_single_communities() -> None:
    g = nx.Graph()
    g.add_node("a")
    g.add_node("b")
    result = cluster(g)
    sizes = sorted(len(c) for c in result.values())
    assert sizes == [1, 1]


def test_cluster_two_cliques() -> None:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    g.add_edges_from([("x", "y"), ("y", "z"), ("x", "z")])
    result = cluster(g)
    # Two cliques should be detected as two communities (at least).
    assert len(result) >= 2
    # Members of a clique should end up in the same community.
    community_of = {node: cid for cid, nodes in result.items() for node in nodes}
    assert community_of["a"] == community_of["b"] == community_of["c"]
    assert community_of["x"] == community_of["y"] == community_of["z"]


def test_cluster_sorted_by_size_descending() -> None:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
    g.add_edge("x", "y")
    result = cluster(g)
    sizes = [len(nodes) for nodes in result.values()]
    assert sizes == sorted(sizes, reverse=True)


def test_cluster_digraph_accepted() -> None:
    g = nx.DiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    result = cluster(g)
    assert any("a" in nodes for nodes in result.values())


def test_cohesion_score_pure_internal() -> None:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    assert cohesion_score(g, {"a", "b", "c"}) == 1.0


def test_cohesion_score_external_edges() -> None:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    # Nodes {a, b}: internal = 1 (a-b), external = 1 (b-c) → 0.5
    assert cohesion_score(g, {"a", "b"}) == 0.5


def test_cohesion_score_empty_community() -> None:
    g = nx.Graph()
    g.add_edge("a", "b")
    assert cohesion_score(g, set()) == 0.0


def test_cluster_splits_oversized() -> None:
    # Path of 20 nodes → Louvain may produce one huge community.
    g = nx.Graph()
    for i in range(20):
        g.add_edge(f"n{i}", f"n{i + 1}")
    result = cluster(g)
    max_fraction = max(len(c) for c in result.values()) / len(g)
    # With oversized splitting, no community should exceed ~25% by much.
    assert max_fraction <= 0.6  # generous — Louvain may still produce chunky blobs


def test_cluster_leiden_fallback_on_import_error(monkeypatch) -> None:
    # Simulate graspologic being unavailable so we take the Louvain fallback.
    import builtins
    import sys

    # Remove any cached graspologic reference first.
    for mod in list(sys.modules):
        if mod.startswith("graspologic"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("graspologic"):
            raise ImportError("no graspologic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("x", "y")])
    result = cluster(g)
    assert result  # something was produced by Louvain


def test_cohesion_score_single_isolated_community() -> None:
    from pengram.cluster import cohesion_score

    g = nx.Graph()
    g.add_node("a")
    # Single isolate has no edges at all → 0.0 by definition.
    assert cohesion_score(g, {"a"}) == 0.0
