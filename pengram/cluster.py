# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Community detection on the pengram graph.

Tries Leiden (``graspologic``) first; falls back to Louvain (``networkx``)
when graspologic is unavailable. Oversized communities (>25% of the graph)
are re-clustered. Isolated nodes each get their own single-node community.

Outputs a dict ``{community_id: [node_ids]}`` sorted by size descending and
keyed by stable integer ids.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from ._ui import warn as _ui_warn

_OVERSIZED_FRACTION = 0.25


def _to_undirected(g: nx.Graph) -> nx.Graph:
    if isinstance(g, nx.DiGraph):
        return g.to_undirected()
    return g


def _leiden_communities(g: nx.Graph) -> list[list[str]] | None:
    try:
        from graspologic.partition import leiden  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        partition = leiden(g)
    except Exception as exc:  # pragma: no cover — graspologic runtime issues
        _ui_warn(f"Leiden failed, falling back to Louvain: {exc}")
        return None
    grouped: dict[int, list[str]] = {}
    for node, community_id in partition.items():
        grouped.setdefault(community_id, []).append(node)
    return list(grouped.values())


def _louvain_communities(g: nx.Graph) -> list[list[str]]:
    try:
        communities = nx.community.louvain_communities(g, seed=42)
    except Exception as exc:
        _ui_warn(f"Louvain failed, returning connected components: {exc}")
        communities = list(nx.connected_components(g))
    return [sorted(c) for c in communities]


def _cluster_once(g: nx.Graph) -> list[list[str]]:
    """Produce one clustering pass using Leiden-or-Louvain."""
    if len(g) == 0:
        return []
    if len(g) == 1:
        return [list(g.nodes)]
    leiden = _leiden_communities(g)
    if leiden is not None:
        return leiden
    return _louvain_communities(g)


def _split_oversized(
    g: nx.Graph,
    communities: list[list[str]],
    *,
    max_fraction: float = _OVERSIZED_FRACTION,
) -> list[list[str]]:
    """Recursively split communities that are larger than ``max_fraction``."""
    total = len(g)
    if total == 0:
        return communities
    threshold = max(1, int(total * max_fraction))
    refined: list[list[str]] = []
    for community in communities:
        if len(community) <= threshold or len(community) <= 2:
            refined.append(community)
            continue
        subgraph = g.subgraph(community).copy()
        sub_communities = _cluster_once(subgraph)
        if len(sub_communities) <= 1:
            refined.append(community)
            continue
        refined.extend(
            _split_oversized(
                subgraph,
                sub_communities,
                max_fraction=max_fraction,
            )
        )
    return refined


def cluster(g: nx.Graph) -> dict[int, list[str]]:
    """Partition ``g`` into communities.

    Returns a dict keyed by sequential community id (largest first). Empty
    graphs return ``{}``. Isolate nodes become single-node communities.
    """
    undirected = _to_undirected(g)
    if len(undirected) == 0:
        return {}

    isolates = [n for n, d in undirected.degree() if d == 0]
    main = undirected.copy()
    main.remove_nodes_from(isolates)

    communities = _cluster_once(main) if len(main) > 0 else []
    communities = _split_oversized(main, communities) if communities else []
    for node in isolates:
        communities.append([node])

    communities.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return {i: sorted(c) for i, c in enumerate(communities)}


def cohesion_score(g: nx.Graph, community_nodes: Iterable[str]) -> float:
    """Return the ratio of intra-community edges to total edges touching the community."""
    undirected = _to_undirected(g)
    nodes = set(community_nodes)
    if not nodes:
        return 0.0
    internal = 0
    external = 0
    for u, v in undirected.edges():
        if u in nodes and v in nodes:
            internal += 1
        elif u in nodes or v in nodes:
            external += 1
    total = internal + external
    if total == 0:
        return 0.0
    return internal / total


__all__ = ["cluster", "cohesion_score"]
