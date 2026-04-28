# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Optional ``catalog.csv`` export — a machine-readable manifest of what
PENgram processed.

One row per source file. Base columns are always present; YouTube-
specific columns appear only when at least one node in the graph
declares YouTube provenance (``video_id`` attribute set).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import networkx as nx

from .export_common import _CONTENT_KINDS

# Base columns for every catalog row.
_BASE_COLUMNS: tuple[str, ...] = (
    "filename",
    "content_type",
    "title",
    "word_count",
    "entity_count",
    "relationship_count",
    "cluster",
)

# YouTube columns only emitted when a content node carries any of these.
_YT_COLUMNS: tuple[str, ...] = (
    "video_id",
    "channel",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
)


def _has_youtube_metadata(g: nx.Graph) -> bool:
    """True when any node has a non-empty ``video_id`` attribute."""
    for node_id in g.nodes:
        if g.nodes[node_id].get("video_id"):
            return True
    return False


def _cluster_of(
    node_id: str,
    communities: Mapping[int, list[str]] | None,
) -> str:
    if not communities:
        return ""
    for cid, members in communities.items():
        if node_id in members:
            return str(cid)
    return ""


def _count_entities_in(
    node_id: str,
    g: nx.Graph,
    entity_kinds: frozenset[str],
) -> int:
    """Count concept nodes referenced from this content node."""
    if not isinstance(g, nx.DiGraph):
        # Undirected: iterate neighbours.
        return sum(1 for nbr in g.neighbors(node_id) if g.nodes[nbr].get("kind") in entity_kinds)
    return sum(
        1
        for _src, tgt, data in g.out_edges(node_id, data=True)
        if data.get("relation") == "references" and g.nodes[tgt].get("kind") in entity_kinds
    )


def _relationship_count_for(node_id: str, g: nx.Graph) -> int:
    """Total edges incident to the node (both directions)."""
    if isinstance(g, nx.DiGraph):
        return g.in_degree(node_id) + g.out_degree(node_id)
    return g.degree(node_id)


def to_rows(
    g: nx.Graph,
    communities: Mapping[int, list[str]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(columns, rows)`` for the catalog.

    YouTube columns are appended only when the graph has YouTube
    metadata; non-YouTube corpora get a narrow CSV with just the base
    columns.
    """
    entity_kinds = frozenset({"concept"})
    include_yt = _has_youtube_metadata(g)
    columns: list[str] = list(_BASE_COLUMNS)
    if include_yt:
        columns.extend(_YT_COLUMNS)

    rows: list[dict[str, Any]] = []
    for node_id in sorted(g.nodes):
        node = g.nodes[node_id]
        if node.get("kind") not in _CONTENT_KINDS:
            continue
        body = str(node.get("body") or node.get("summary") or "")
        row: dict[str, Any] = {
            "filename": str(node.get("label") or node_id),
            "content_type": str(node.get("kind") or ""),
            "title": str(node.get("label") or node_id),
            "word_count": len(body.split()),
            "entity_count": _count_entities_in(node_id, g, entity_kinds),
            "relationship_count": _relationship_count_for(node_id, g),
            "cluster": _cluster_of(node_id, communities),
        }
        if include_yt:
            for col in _YT_COLUMNS:
                val = node.get(col, "")
                row[col] = "" if val is None else val
        rows.append(row)
    return columns, rows


def export_catalog(
    g: nx.Graph,
    output_dir: Path,
    *,
    communities: Mapping[int, list[str]] | None = None,
    filename: str = "catalog.csv",
) -> Path:
    """Write ``output_dir/catalog.csv`` and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = to_rows(g, communities)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


__all__ = ["export_catalog", "to_rows"]
