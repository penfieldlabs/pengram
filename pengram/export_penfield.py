# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Penfield-compliant vault export.

Writes a directory of Markdown files ready for ``penfield-import``. Each note
has YAML frontmatter with:

- ``note_type`` — one of ``document``, ``transcript``, ``concept``,
  ``category``, ``code`` (vault-internal classification).
- Relationship keys from the 24 Penfield semantic types, each holding an
  array of wikilink strings, e.g. ``supports: ["[[Other Note]]"]``.
- Metadata (``tags``, ``source_file``, ``source_path``, ``mentions``, …).

There is deliberately **no** top-level ``type:`` field — that name is
reserved for the relationship vocabulary.

penfield-import reads relationships from frontmatter only, so the note body
never contains a ``## Relationships`` section in this export.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import networkx as nx

from .export_common import (
    DEFAULT_THRESHOLDS,
    NoteSpec,
    _group_edges_by_relation,
    _incoming_category_backlinks,
    _subdir_for,
    build_slug_map,
    extract_metadata,
    format_frontmatter,
    included_nodes,
    note_type_of,
    render_body,
)


def render_note(spec: NoteSpec) -> str:
    """Render a Penfield-compliant note. No inline Relationships section.

    The note does NOT emit a synthetic ``# {title}`` heading — Obsidian
    already renders the filename as the note title. And there is NO
    blank line between the closing ``---`` fence and the body.
    """
    parts = [format_frontmatter(spec)]
    if spec.body.strip():
        parts.append(spec.body.rstrip())
    return "\n".join(parts) + "\n"


def export_penfield(
    g: nx.Graph,
    output_dir: Path,
    *,
    thresholds: Mapping[str, int] | None = None,
    vault_name: str = "vault-penfield",
    min_confidence: str | None = None,
) -> Path:
    """Write a Penfield-compliant vault under ``output_dir/<vault_name>/``.

    Returns the vault root path. Idempotent: running twice on unchanged input
    produces identical files.
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    vault = Path(output_dir) / vault_name
    vault.mkdir(parents=True, exist_ok=True)
    included = included_nodes(g, thresholds)
    slug_map = build_slug_map(g, included)

    for node_id in g.nodes:
        if node_id not in included:
            continue
        node = g.nodes[node_id]
        note_type = note_type_of(node)
        metadata = extract_metadata(node, note_type, slug_map=slug_map, node_id=node_id, g=g)
        body = render_body(note_type, node, slug_map)

        relationships = _group_edges_by_relation(
            g,
            node_id,
            included_node_ids=included,
            slug_map=slug_map,
            min_confidence=min_confidence,
        )
        for rel, targets in _incoming_category_backlinks(
            g, node_id, included_node_ids=included, slug_map=slug_map
        ).items():
            relationships.setdefault(rel, []).extend(targets)
        spec = NoteSpec(
            node_id=node_id,
            note_type=note_type,
            body=body,
            metadata=metadata,
            relationships=relationships,
        )
        subdir = _subdir_for(vault, note_type, node)
        subdir.mkdir(parents=True, exist_ok=True)
        file_path = subdir / f"{slug_map[node_id]}.md"
        file_path.write_text(render_note(spec), encoding="utf-8")
    return vault


__all__ = [
    "NoteSpec",
    "export_penfield",
    "render_note",
    "format_frontmatter",
    "build_slug_map",
    "included_nodes",
    "DEFAULT_THRESHOLDS",
]
