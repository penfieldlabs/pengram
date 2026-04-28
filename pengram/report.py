# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Render GRAPH_REPORT.md from graph + analysis + communities."""

from __future__ import annotations

from typing import Any

import networkx as nx


def _label(g: nx.Graph, node: str) -> str:
    return g.nodes[node].get("label", node) if node in g.nodes else node


def render_report(
    g: nx.Graph,
    analysis: dict[str, Any],
    communities: dict[int, list[str]],
) -> str:
    """Return a Markdown report summarising the graph."""
    lines: list[str] = []
    lines.append("# Graph Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Nodes: **{analysis['node_count']}**")
    lines.append(f"- Edges: **{analysis['edge_count']}**")
    lines.append(f"- Communities: **{analysis['community_count']}**")
    lines.append("")

    lines.append("## God Nodes")
    lines.append("")
    gods = analysis.get("god_nodes") or []
    if gods:
        lines.append("| Label | Degree | Betweenness | Score |")
        lines.append("| --- | --- | --- | --- |")
        for god in gods:
            lines.append(
                f"| {god['label']} | {god['degree']} | {god['betweenness']} | {god['score']} |"
            )
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Bridge Nodes")
    lines.append("")
    bridges = analysis.get("bridge_nodes") or []
    if bridges:
        lines.append("| Label | Communities bridged |")
        lines.append("| --- | --- |")
        for b in bridges:
            lines.append(f"| {b['label']} | {b['bridges']} |")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Communities")
    lines.append("")
    multi = {cid: ns for cid, ns in communities.items() if len(ns) > 1} if communities else {}
    if multi:
        for cid, nodes in multi.items():
            preview = ", ".join(_label(g, n) for n in nodes[:8])
            more = f" (+{len(nodes) - 8} more)" if len(nodes) > 8 else ""
            lines.append(f"- **Community {cid}** ({len(nodes)} nodes): {preview}{more}")
        singles = len(communities) - len(multi)
        if singles:
            lines.append(f"- _{singles} single-node communities omitted_")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Surprising Connections")
    lines.append("")
    surprises = analysis.get("surprising_connections") or []
    if surprises:
        for s in surprises:
            lines.append(
                f"- {s['source_label']} → {s['target_label']} "
                f"(`{s['relation']}`, {s['confidence']})"
            )
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Suggested Questions")
    lines.append("")
    for q in analysis.get("suggested_questions", []):
        lines.append(f"- {q}")
    if not analysis.get("suggested_questions"):
        lines.append("_None._")
    lines.append("")

    return "\n".join(lines)


__all__ = ["render_report"]
