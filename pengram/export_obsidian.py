# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Obsidian-compatible vault export.

Same frontmatter format as :mod:`pengram.export_penfield` (so the Penfield
import pipeline still works), **plus** an inline ``## Relationships`` section
using the Wikilink Types ``@type`` alias syntax.

Per the obsidian-wikilink-types SKILL rule 1, frontmatter and inline links
must match — both the YAML keys and the inline list are rendered from the
same underlying relationship dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import networkx as nx

from . import vocabulary as _vocab
from ._ui import step as _ui_step
from .export_common import (
    _IMAGE_EXTENSIONS,
    DEFAULT_THRESHOLDS,
    NoteSpec,
    _group_edges_by_relation,
    _incoming_category_backlinks,
    _subdir_for,
    build_slug_map,
    copy_image_attachment,
    extract_metadata,
    format_frontmatter,
    included_nodes,
    note_type_of,
    purge_stale_files,
    render_body,
    write_text_if_changed,
)


def _strip_wikilink(text: str) -> str:
    """Return the inner slug from a ``[[slug]]`` wikilink (or the raw string)."""
    s = text.strip()
    if s.startswith("[[") and s.endswith("]]"):
        return s[2:-2]
    return s


def render_relationships_section(relationships: Mapping[str, list[str]]) -> str:
    """Render the inline ``## Relationships`` block.

    Emits ``- → [[slug|slug @relation]]`` per the wikilink-types spec.
    """
    lines: list[str] = []
    any_lines = False
    for relation in sorted(relationships):
        if not _vocab.is_valid_semantic(relation):
            continue
        for target in relationships[relation]:
            slug = _strip_wikilink(target)
            lines.append(f"- → [[{slug}|{slug} @{relation}]]")
            any_lines = True
    if not any_lines:
        return ""
    return "## Relationships\n\n" + "\n".join(lines) + "\n"


def render_note(spec: NoteSpec) -> str:
    """Render a full Obsidian-style note: frontmatter + body + Relationships."""
    parts = [format_frontmatter(spec)]
    if spec.body.strip():
        parts.append(spec.body.rstrip())
    rel_block = render_relationships_section(spec.relationships)
    if rel_block:
        if len(parts) > 1:
            parts.append("")
        parts.append(rel_block.rstrip())
    return "\n".join(parts) + "\n"


def export_obsidian(
    g: nx.Graph,
    output_dir: Path,
    *,
    thresholds: Mapping[str, int] | None = None,
    vault_name: str = "vault-obsidian",
    min_confidence: str | None = None,
) -> Path:
    """Write an Obsidian-compatible vault under ``output_dir/<vault_name>/``."""
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    vault = Path(output_dir) / vault_name
    vault.mkdir(parents=True, exist_ok=True)
    included = included_nodes(g, thresholds)
    slug_map = build_slug_map(g, included)

    written: set[Path] = set()
    included_list = [nid for nid in g.nodes if nid in included]
    total = len(included_list)
    progress_step = max(1, total // 10) if total else 1
    for i, node_id in enumerate(included_list, 1):
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
        subdir = _subdir_for(vault, note_type, node)
        subdir.mkdir(parents=True, exist_ok=True)
        slug = slug_map[node_id]
        if note_type == "document":
            embed_filename = copy_image_attachment(node, subdir, slug)
            if embed_filename:
                body = body.rstrip()
                if body:
                    body += "\n\n"
                body += f"## Source\n\n![[{embed_filename}]]"
                written.add(subdir / embed_filename)
        spec = NoteSpec(
            node_id=node_id,
            note_type=note_type,
            body=body,
            metadata=metadata,
            relationships=relationships,
        )
        md_path = subdir / f"{slug}.md"
        write_text_if_changed(md_path, render_note(spec))
        written.add(md_path)
        if i % progress_step == 0 and i < total:
            _ui_step(i, total, "Export (Obsidian)")
    if total:
        _ui_step(total, total, "Export (Obsidian)")
    purge_stale_files(vault, written, extensions=frozenset({".md"}) | _IMAGE_EXTENSIONS)
    return vault


__all__ = ["export_obsidian", "render_note", "render_relationships_section"]
