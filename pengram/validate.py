# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Schema validation for extraction results.

Every module that produces nodes and edges (extract_ast, extract_llm, link)
should produce output matching this schema:

.. code-block:: json

    {
      "nodes": [
        {"id": "str", "label": "str", "kind": "optional",
         "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", ...}
      ],
      "edges": [
        {"source": "str", "target": "str", "relation": "str",
         "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", ...}
      ]
    }

``validate_extraction`` returns an empty list on success, or a list of
human-readable error strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import vocabulary as v


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _validate_node(node: Any, idx: int, errors: list[str]) -> None:
    if not isinstance(node, Mapping):
        errors.append(f"nodes[{idx}] must be an object, got {type(node).__name__}")
        return
    if not _is_nonempty_str(node.get("id")):
        errors.append(f"nodes[{idx}] missing non-empty 'id'")
    if not _is_nonempty_str(node.get("label")):
        errors.append(f"nodes[{idx}] missing non-empty 'label'")
    confidence = node.get("confidence")
    if confidence is not None and not v.is_valid_confidence(confidence):
        errors.append(
            f"nodes[{idx}] has invalid confidence {confidence!r}; "
            f"valid labels: {sorted(v.CONFIDENCE_LABELS)}"
        )


def _validate_edge(edge: Any, idx: int, errors: list[str]) -> None:
    if not isinstance(edge, Mapping):
        errors.append(f"edges[{idx}] must be an object, got {type(edge).__name__}")
        return
    if not _is_nonempty_str(edge.get("source")):
        errors.append(f"edges[{idx}] missing non-empty 'source'")
    if not _is_nonempty_str(edge.get("target")):
        errors.append(f"edges[{idx}] missing non-empty 'target'")
    relation = edge.get("relation")
    if relation is None:
        errors.append(f"edges[{idx}] missing 'relation'")
    elif not v.is_valid(relation):
        errors.append(
            f"edges[{idx}] has invalid relation {relation!r}; valid types: {sorted(v.ALL_TYPES)}"
        )
    confidence = edge.get("confidence")
    if confidence is not None and not v.is_valid_confidence(confidence):
        errors.append(
            f"edges[{idx}] has invalid confidence {confidence!r}; "
            f"valid labels: {sorted(v.CONFIDENCE_LABELS)}"
        )


def validate_extraction(data: Any) -> list[str]:
    """Validate an extraction dict. Returns a list of errors (empty = valid)."""
    errors: list[str] = []
    if not isinstance(data, Mapping):
        return [f"extraction must be an object, got {type(data).__name__}"]

    nodes = data.get("nodes")
    edges = data.get("edges")

    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        errors.append("'nodes' must be a list")
    else:
        for i, node in enumerate(nodes):
            _validate_node(node, i, errors)

    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        errors.append("'edges' must be a list")
    else:
        for i, edge in enumerate(edges):
            _validate_edge(edge, i, errors)

    return errors


__all__ = ["validate_extraction"]
