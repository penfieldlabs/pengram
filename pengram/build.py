# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Assemble extraction results into a NetworkX graph.

Each extraction is a dict ``{"nodes": [...], "edges": [...]}`` that has
already passed :func:`pengram.validate.validate_extraction`. Build merges
them into a single :class:`networkx.DiGraph`:

- Node IDs are normalised (lowercased, non-alphanumeric replaced with ``_``).
- Duplicate nodes merge; later writes win (semantic extractions merging on
  top of AST extractions is the common case).
- Edges whose target does not exist in the final node set are dropped
  silently — they point to stdlib/external identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import networkx as nx

from ._ui import say as _ui_say
from .errors import ExtractionError
from .validate import validate_extraction

_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9]+")


def normalize_id(value: str) -> str:
    """Return a stable, filesystem-safe node id derived from ``value``."""
    cleaned = _ID_SAFE_RE.sub("_", value).strip("_").lower()
    return cleaned


def build_from_extraction(
    extraction: dict[str, Any],
    *,
    directed: bool = True,
) -> nx.DiGraph | nx.Graph:
    """Build a graph from a single extraction dict.

    Raises :class:`ValueError` if the extraction fails schema validation.
    """
    errors = validate_extraction(extraction)
    if errors:
        raise ExtractionError(f"Invalid extraction: {errors[0]}")
    graph_cls: type[nx.Graph] = nx.DiGraph if directed else nx.Graph
    g: nx.Graph = graph_cls()
    for node in extraction["nodes"]:
        nid = normalize_id(node["id"])
        attrs = {k: v for k, v in node.items() if k != "id"}
        g.add_node(nid, **attrs)
    for edge in extraction["edges"]:
        src = normalize_id(edge["source"])
        tgt = normalize_id(edge["target"])
        if src not in g or tgt not in g:
            continue
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        g.add_edge(src, tgt, **attrs)
    return g


def build(
    extractions: Iterable[dict[str, Any]],
    *,
    directed: bool = True,
) -> nx.DiGraph | nx.Graph:
    """Merge many extraction dicts into a single graph.

    Node attributes use last-write-wins. Duplicate ``source+target+relation``
    edges collapse to a single edge; other edge attributes keep the latest
    value.
    """
    graph_cls: type[nx.Graph] = nx.DiGraph if directed else nx.Graph
    merged: nx.Graph = graph_cls()

    # First pass: collect every node so edge endpoints can be validated.
    edge_conflicts: list[tuple[str, str, str, str]] = []
    for extraction in extractions:  # type: ignore[assignment]
        errors = validate_extraction(extraction)
        if errors:
            raise ExtractionError(f"Invalid extraction: {errors[0]}")
        for node in extraction["nodes"]:
            nid = normalize_id(node["id"])
            attrs = {k: v for k, v in node.items() if k != "id"}
            if nid in merged:
                merged.nodes[nid].update(attrs)
            else:
                merged.add_node(nid, **attrs)
        for edge in extraction["edges"]:
            src = normalize_id(edge["source"])
            tgt = normalize_id(edge["target"])
            if src not in merged or tgt not in merged:
                continue  # drop dangling
            relation = edge.get("relation")
            attrs = {k: val for k, val in edge.items() if k not in ("source", "target")}
            if merged.has_edge(src, tgt):
                existing_relation = merged[src][tgt].get("relation")
                if existing_relation == relation:
                    merged[src][tgt].update(attrs)
                    continue
                edge_conflicts.append((src, tgt, existing_relation or "", relation or ""))
            merged.add_edge(src, tgt, **attrs)
    if edge_conflicts:
        _ui_say(
            f"  Build: {len(edge_conflicts)} conflicting edge relation(s) "
            f"resolved (last-write wins)"
        )
    return merged


__all__ = ["normalize_id", "build", "build_from_extraction"]
