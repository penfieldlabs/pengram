# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Graph analysis — god nodes, bridges, surprising connections, questions.

All metrics use only :mod:`networkx` so this module never pulls heavy
optional deps.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

_TOP_N_DEFAULT = 10
_SURPRISE_CONFIDENCE = {"EXTRACTED", "INFERRED"}
_SYNTHETIC_KINDS: frozenset[str] = frozenset({"category", "code"})


def _as_undirected(g: nx.Graph) -> nx.Graph:
    return g.to_undirected() if isinstance(g, nx.DiGraph) else g


def _label_of(g: nx.Graph, node: str) -> str:
    return g.nodes[node].get("label", node)


def god_nodes(g: nx.Graph, *, top_n: int = _TOP_N_DEFAULT) -> list[dict[str, Any]]:
    """Return the top-N nodes by a combined degree + betweenness score."""
    if len(g) == 0:
        return []
    undirected = _as_undirected(g)
    degree = dict(undirected.degree())
    if len(undirected) > 1:
        try:
            betweenness = nx.betweenness_centrality(undirected)
        except Exception:
            betweenness = {n: 0.0 for n in undirected}
    else:
        betweenness = {n: 0.0 for n in undirected}
    max_degree = max(degree.values()) or 1
    max_betweenness = max(betweenness.values()) or 1e-9
    scores: list[tuple[str, float]] = []
    for node in undirected:
        if g.nodes[node].get("kind") in _SYNTHETIC_KINDS:
            continue
        score = (degree[node] / max_degree) + (betweenness[node] / max_betweenness)
        scores.append((node, score))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [
        {
            "id": node,
            "label": _label_of(g, node),
            "degree": degree[node],
            "betweenness": round(betweenness[node], 4),
            "score": round(score, 4),
        }
        for node, score in scores[:top_n]
    ]


def bridge_nodes(
    g: nx.Graph,
    communities: dict[int, list[str]],
    *,
    top_n: int = _TOP_N_DEFAULT,
) -> list[dict[str, Any]]:
    """Return nodes connecting the most distinct communities."""
    if not communities or len(g) == 0:
        return []
    community_of: dict[str, int] = {}
    for cid, nodes in communities.items():
        for node in nodes:
            community_of[node] = cid
    undirected = _as_undirected(g)
    scored: list[tuple[str, int]] = []
    for node in undirected:
        if g.nodes[node].get("kind") in _SYNTHETIC_KINDS:
            continue
        neighbour_communities = {
            community_of.get(n)
            for n in undirected.neighbors(node)
            if community_of.get(n) is not None and community_of.get(n) != community_of.get(node)
        }
        if not neighbour_communities:
            continue
        scored.append((node, len(neighbour_communities)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [
        {"id": node, "label": _label_of(g, node), "bridges": count}
        for node, count in scored[:top_n]
    ]


def surprising_connections(
    g: nx.Graph,
    communities: dict[int, list[str]],
    *,
    top_n: int = _TOP_N_DEFAULT,
) -> list[dict[str, Any]]:
    """Return high-confidence edges that cross community boundaries.

    Edges whose source and target belong to different communities are
    surprising when both endpoints are relatively small communities.
    """
    community_of: dict[str, int] = {}
    for cid, nodes in communities.items():
        for node in nodes:
            community_of[node] = cid
    undirected = _as_undirected(g)
    surprises: list[dict[str, Any]] = []
    for u, v, data in undirected.edges(data=True):
        confidence = data.get("confidence")
        if confidence not in _SURPRISE_CONFIDENCE:
            continue
        cu, cv = community_of.get(u), community_of.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        surprises.append(
            {
                "source": u,
                "target": v,
                "relation": data.get("relation", ""),
                "confidence": confidence,
                "source_label": _label_of(g, u),
                "target_label": _label_of(g, v),
                "source_community": cu,
                "target_community": cv,
            }
        )
    surprises.sort(key=lambda s: (s["confidence"] != "EXTRACTED", s["source"]))
    return surprises[:top_n]


def suggested_questions(
    g: nx.Graph,
    gods: list[dict[str, Any]],
    surprises: list[dict[str, Any]],
) -> list[str]:
    """Generate natural-language questions from graph structure."""
    questions: list[str] = []
    for god in gods[:3]:
        questions.append(f"Why does {god['label']!r} appear so central in the corpus?")
    for surprise in surprises[:3]:
        questions.append(
            f"What is the connection between {surprise['source_label']!r} and "
            f"{surprise['target_label']!r} ({surprise['relation']})?"
        )
    if not questions and len(g) > 0:
        any_node = next(iter(g.nodes))
        questions.append(f"What does the graph reveal about {_label_of(g, any_node)!r}?")
    return questions


def analyze(g: nx.Graph, communities: dict[int, list[str]]) -> dict[str, Any]:
    """Run all analyses and return a single summary dict."""
    gods = god_nodes(g)
    bridges = bridge_nodes(g, communities)
    surprises = surprising_connections(g, communities)
    questions = suggested_questions(g, gods, surprises)
    return {
        "node_count": len(g),
        "edge_count": g.number_of_edges(),
        "community_count": len(communities),
        "god_nodes": gods,
        "bridge_nodes": bridges,
        "surprising_connections": surprises,
        "suggested_questions": questions,
    }


__all__ = [
    "analyze",
    "god_nodes",
    "bridge_nodes",
    "surprising_connections",
    "suggested_questions",
]
