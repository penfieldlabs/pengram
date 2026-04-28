# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Export the graph as a single ``graph.json`` file.

Schema::

    {
      "meta": {"nodes": int, "edges": int, "communities": int,
               "directed": bool, "version": "0.1"},
      "nodes": [{"id": "...", "label": "...", ...}],
      "edges": [{"source": "...", "target": "...", "relation": "...", "relationship": "...", ...}],
      "communities": {"<cid>": ["id1", ...]},
      "analysis": {...}  # optional
    }
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import networkx as nx

GRAPH_JSON_VERSION = "0.1"


def to_dict(
    g: nx.Graph,
    communities: dict[int, list[str]] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable dict representation of the graph."""
    directed = isinstance(g, nx.DiGraph)
    nodes = [{"id": n, **g.nodes[n]} for n in g.nodes]
    edges = []
    for u, v, data in g.edges(data=True):
        edge: dict[str, Any] = {"source": u, "target": v, **data}
        # "relation" is canonical internally; "relationship" aliases it
        # for external consumers (e.g. Penfield import, graph viewers).
        if "relation" in edge and "relationship" not in edge:
            edge["relationship"] = edge["relation"]
        edges.append(edge)
    return {
        "meta": {
            "nodes": len(nodes),
            "edges": len(edges),
            "communities": len(communities or {}),
            "directed": directed,
            "version": GRAPH_JSON_VERSION,
        },
        "nodes": nodes,
        "edges": edges,
        "communities": {str(cid): members for cid, members in (communities or {}).items()},
        "analysis": analysis or {},
    }


def _json_default(obj: Any) -> str:
    """Fallback serialiser for types ``json.dump`` cannot handle natively."""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_json(
    g: nx.Graph,
    output_dir: Path,
    *,
    communities: dict[int, list[str]] | None = None,
    analysis: dict[str, Any] | None = None,
    filename: str = "graph.json",
) -> Path:
    """Write the graph to ``output_dir/graph.json`` and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = to_dict(g, communities, analysis)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
    return path


__all__ = ["export_json", "to_dict", "GRAPH_JSON_VERSION"]
