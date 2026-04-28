# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Shared constants, data structures, and helpers for vault exporters.

Both :mod:`pengram.export_penfield` and :mod:`pengram.export_obsidian`
need the same filtering, slug-mapping, tag-computation, and body-rendering
logic. Centralising it here makes the dependency explicit and keeps the
exporters focused on format-specific rendering.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import networkx as nx

from . import vocabulary as _vocab
from .security import slugify

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Mapping[str, int] = {
    "concept": 3,
}

_NOTE_KIND_PLURAL: Mapping[str, str] = {
    "concept": "concepts",
    "category": "categories",
    "code": "code",
    "document": "documents",
    "image": "images",
    "transcript": "transcripts",
    "video_transcript": "transcripts",
    "audio_transcript": "transcripts",
    "file": "files",
}

_VAULT_SKIP_KINDS: frozenset[str] = frozenset(
    {
        "file",
        "class",
        "function",
    }
)

_CONTENT_KINDS: frozenset[str] = frozenset(
    {
        "document",
        "transcript",
        "video_transcript",
        "audio_transcript",
    }
)

_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "note_type",
        "tags",
        "source_file",
        "source_path",
        "mentions",
        "summary",
        "url",
        "date",
        "duration",
        "video_id",
        "channel",
        "language",
        "aliases",
    }
)

_MENTIONS_KINDS: frozenset[str] = frozenset({"concept"})

_CONFIDENCE_RANK: Mapping[str, int] = {
    "AMBIGUOUS": 0,
    "INFERRED": 1,
    "EXTRACTED": 2,
}


# ---------------------------------------------------------------------------
# NoteSpec
# ---------------------------------------------------------------------------


@dataclass
class NoteSpec:
    """The input to vault note renderers.

    No ``title`` field — Obsidian renders the filename as the heading and
    Penfield uses the filename as the identity.
    """

    node_id: str
    note_type: str
    body: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    relationships: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plural(note_type: str) -> str:
    return _NOTE_KIND_PLURAL.get(note_type, f"{note_type}s")


def _subdir_for(
    vault: Path,
    note_type: str,
    node: Mapping[str, Any],
) -> Path:
    """Return the directory the note should be written to.

    Content kinds preserve input folder hierarchy under their type
    subdirectory. Non-content notes stay at the top of their type dir.
    """
    base = vault / _plural(note_type)
    if note_type not in _CONTENT_KINDS:
        return base
    source_path = node.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        return base
    rel = PurePosixPath(source_path)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        return base
    if not rel.parts:
        return base
    return base.joinpath(*rel.parts)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in ':#-[]{}"\\'):
        escaped = text.replace("'", "''")
        return f"'{escaped}'"
    if not text.strip():
        return "''"
    return text


def _yaml_list(items: Iterable[str], *, quote: bool = True) -> str:
    rendered: list[str] = []
    for item in items:
        if quote:
            escaped = str(item).replace('"', '\\"')
            rendered.append(f'"{escaped}"')
        else:
            rendered.append(_yaml_scalar(item))
    return "[" + ", ".join(rendered) + "]"


def format_frontmatter(spec: NoteSpec) -> str:
    """Render a :class:`NoteSpec` as a YAML frontmatter block."""
    lines: list[str] = ["---"]
    lines.append(f"note_type: {_yaml_scalar(spec.note_type)}")

    for key in sorted(spec.metadata.keys()):
        if key not in _METADATA_KEYS or key == "note_type":
            continue
        value = spec.metadata[key]
        if isinstance(value, list):
            if not value:
                continue
            lines.append(f"{key}: {_yaml_list(value)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")

    for relation in sorted(spec.relationships):
        if not _vocab.is_valid_semantic(relation):
            continue
        targets = spec.relationships[relation]
        if not targets:
            continue
        lines.append(f"{relation}:")
        for target in targets:
            escaped = str(target).replace('"', '\\"')
            lines.append(f'  - "{escaped}"')

    lines.append("---")
    return "\n".join(lines)


def _wikilink(slug: str) -> str:
    return f"[[{slug}]]"


def _group_edges_by_relation(
    g: nx.Graph,
    node_id: str,
    *,
    included_node_ids: set[str] | None = None,
    slug_map: Mapping[str, str] | None = None,
    min_confidence: str | None = None,
) -> dict[str, list[str]]:
    """Return ``{relation: [wikilink, ...]}`` for outgoing edges of a node."""
    if node_id not in g:
        return {}
    floor = _CONFIDENCE_RANK.get(min_confidence or "", -1)
    result: dict[str, list[str]] = {}
    iterator = (
        g.out_edges(node_id, data=True)
        if isinstance(g, nx.DiGraph)
        else ((node_id, n, data) for n, data in g[node_id].items())
    )
    for _src, tgt, data in iterator:
        if _src == tgt:
            continue
        relation = data.get("relation", "")
        if not _vocab.is_valid_semantic(relation):
            continue
        if included_node_ids is not None and tgt not in included_node_ids:
            continue
        if floor >= 0:
            edge_rank = _CONFIDENCE_RANK.get(data.get("confidence", ""), 0)
            if edge_rank < floor:
                continue
        if slug_map is not None and tgt in slug_map:
            target_slug = slug_map[tgt]
        else:
            target_label = g.nodes[tgt].get("label", tgt) if tgt in g.nodes else tgt
            target_slug = slugify(str(target_label))
        result.setdefault(relation, []).append(_wikilink(target_slug))
    return {rel: sorted({*targets}) for rel, targets in result.items()}


def _incoming_category_backlinks(
    g: nx.Graph,
    node_id: str,
    *,
    included_node_ids: set[str] | None = None,
    slug_map: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Return ``child_of`` back-links for incoming ``parent_of`` edges from categories."""
    if not isinstance(g, nx.DiGraph) or node_id not in g:
        return {}
    targets: list[str] = []
    for src, _tgt, data in g.in_edges(node_id, data=True):
        if data.get("relation") != "parent_of":
            continue
        if g.nodes.get(src, {}).get("kind") != "category":
            continue
        if included_node_ids is not None and src not in included_node_ids:
            continue
        if slug_map is not None and src in slug_map:
            cat_slug = slug_map[src]
        else:
            cat_label = g.nodes[src].get("label", src) if src in g.nodes else src
            cat_slug = slugify(str(cat_label))
        targets.append(_wikilink(cat_slug))
    if not targets:
        return {}
    return {"child_of": sorted(set(targets))}


def _compute_tags(
    g: nx.Graph,
    node_id: str,
    note_type: str,
    slug_map: Mapping[str, str],
) -> list[str]:
    """Derive vault tags from graph relationships."""
    own_slug = slug_map.get(node_id)
    tags: list[str] = []
    seen: set[str] = set()

    def _add(tag: str) -> None:
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)

    if note_type == "concept":
        if own_slug:
            _add(own_slug)
        _add("concept-hub")
        return tags
    if note_type == "category":
        if own_slug:
            _add(own_slug)
        _add("category-hub")
        return tags
    if note_type == "code":
        if own_slug:
            _add(own_slug)
        _add("code-hub")
        return tags

    if node_id in g:
        out_iter = (
            g.out_edges(node_id, data=True)
            if isinstance(g, nx.DiGraph)
            else ((node_id, n, data) for n, data in g[node_id].items())
        )
        for _src, tgt, data in out_iter:
            if data.get("relation") != "references":
                continue
            if g.nodes.get(tgt, {}).get("kind") != "concept":
                continue
            slug = slug_map.get(tgt)
            if slug:
                _add(slug)
        if isinstance(g, nx.DiGraph):
            for src, _tgt, data in g.in_edges(node_id, data=True):
                if data.get("relation") != "parent_of":
                    continue
                if g.nodes.get(src, {}).get("kind") != "category":
                    continue
                slug = slug_map.get(src)
                if slug:
                    _add(slug)
    return tags[:10]


def _render_category_body(
    node: Mapping[str, Any],
    slug_map: Mapping[str, str],
) -> str:
    """Build the body for a synthetic category/cluster hub note."""
    members = node.get("members") or []
    link_targets: list[str] = []
    for member_id in members:
        slug = slug_map.get(member_id)
        if slug:
            link_targets.append(slug)
    if not link_targets:
        return ""
    lines = ["## Members", ""]
    for slug in link_targets:
        lines.append(f"- [[{slug}]]")
    lines.append("")
    return "\n".join(lines).rstrip()


def _render_concept_body(node: Mapping[str, Any], slug_map: Mapping[str, str]) -> str:
    """Build the body for a concept note."""
    lines: list[str] = []

    definition = str(node.get("definition") or "").strip()
    note = str(node.get("note") or "").strip()
    lead = definition or note
    if lead:
        lines.append(lead)
        lines.append("")

    if definition and note and definition != note:
        lines.append("## Original Note")
        lines.append("")
        lines.append(note)
        lines.append("")

    quotes = node.get("quotes") or []
    if isinstance(quotes, list) and quotes:
        lines.append("## Key Quotes")
        lines.append("")
        for q in quotes:
            if not isinstance(q, Mapping):
                continue
            text = str(q.get("text") or "").strip()
            if not text:
                continue
            source = str(q.get("source") or "").strip()
            suffix = f" — {source}" if source else ""
            lines.append(f"> {text}{suffix}")
        lines.append("")

    cross = str(node.get("cross_source_notes") or "").strip()
    if cross:
        lines.append("## Cross-Source Notes")
        lines.append("")
        lines.append(cross)
        lines.append("")

    appearances = node.get("appearances") or []
    link_targets: list[str] = []
    for appearance_id in appearances:
        slug = slug_map.get(appearance_id)
        if slug:
            link_targets.append(slug)
    if link_targets:
        lines.append("## Discussed in")
        lines.append("")
        for slug in link_targets:
            lines.append(f"- [[{slug}]]")
        lines.append("")
    return "\n".join(lines).rstrip()


def note_type_of(node: Mapping[str, Any]) -> str:
    """Return the vault note-type for a graph node."""
    kind = node.get("kind") or node.get("note_type")
    if kind == "transcript":
        if node.get("video_id") or node.get("channel"):
            return "video_transcript"
        return "transcript"
    if kind in _NOTE_KIND_PLURAL:
        return kind
    return "document"


def _meets_threshold(
    g: nx.Graph,
    node_id: str,
    note_type: str,
    thresholds: Mapping[str, int],
) -> bool:
    minimum = thresholds.get(note_type)
    if minimum is None:
        return True
    if note_type == "category":
        member_count = sum(
            1
            for _, _tgt, data in g.out_edges(node_id, data=True)
            if data.get("relation") == "parent_of"
        )
        return member_count >= minimum
    mentions = int(g.nodes[node_id].get("mentions", 1) or 1)
    return mentions >= minimum


_SLUG_PRIORITY: dict[str, int] = {
    "concept": 0,
    "document": 1,
    "transcript": 1,
    "video_transcript": 1,
    "audio_transcript": 1,
    "code": 1,
    "category": 2,
}


def build_slug_map(
    g: nx.Graph,
    included: set[str] | None = None,
) -> dict[str, str]:
    """Return ``{node_id: unique_slug}`` covering every included node."""
    scope = included if included is not None else set(g.nodes)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for node_id in sorted(
        scope,
        key=lambda nid: (
            _SLUG_PRIORITY.get(g.nodes[nid].get("kind", ""), 1) if nid in g.nodes else 1,
            nid,
        ),
    ):
        if node_id not in g.nodes:
            continue
        title = g.nodes[node_id].get("label", node_id)
        base = slugify(str(title))
        if base not in used:
            slug = base
        else:
            i = 2
            while f"{base}-{i}" in used:
                i += 1
            slug = f"{base}-{i}"
        used.add(slug)
        mapping[node_id] = slug
    return mapping


def included_nodes(
    g: nx.Graph,
    thresholds: Mapping[str, int] | None = None,
) -> set[str]:
    """Return the set of node ids that will get a vault note."""
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    kept: set[str] = set()
    for node_id in g.nodes:
        node = g.nodes[node_id]
        raw_kind = node.get("kind") or node.get("note_type")
        if raw_kind in _VAULT_SKIP_KINDS:
            continue
        note_type = note_type_of(node)
        if note_type in _VAULT_SKIP_KINDS:
            continue
        if _meets_threshold(g, node_id, note_type, thresholds):
            kept.add(node_id)

    # Second pass: drop concept nodes with zero edges to other included nodes.
    to_remove: set[str] = set()
    for node_id in sorted(kept):
        if note_type_of(g.nodes[node_id]) != "concept":
            continue
        has_included_edge = False
        for _u, v in g.out_edges(node_id):
            if v in kept and v != node_id:
                has_included_edge = True
                break
        if not has_included_edge and isinstance(g, nx.DiGraph):
            for u, _v in g.in_edges(node_id):
                if u in kept and u != node_id:
                    has_included_edge = True
                    break
        if not has_included_edge:
            to_remove.add(node_id)
    kept -= to_remove

    return kept


def extract_metadata(
    node: Mapping[str, Any],
    note_type: str,
    *,
    slug_map: Mapping[str, str] | None = None,
    node_id: str | None = None,
    g: nx.Graph | None = None,
) -> dict[str, Any]:
    """Build the metadata dict for a vault note from a graph node."""
    metadata: dict[str, Any] = {}
    for k, v in node.items():
        if k not in _METADATA_KEYS or k == "note_type":
            continue
        if k == "mentions" and note_type not in _MENTIONS_KINDS:
            continue
        if k == "source_path" and (not isinstance(v, str) or not v):
            continue
        metadata[k] = v
    source_file = node.get("source_file")
    if source_file:
        metadata["source_file"] = source_file
    if slug_map is not None and node_id is not None and g is not None:
        metadata["tags"] = _compute_tags(g, node_id, note_type, slug_map)
    return metadata


def render_body(
    note_type: str,
    node: Mapping[str, Any],
    slug_map: Mapping[str, str],
) -> str:
    """Render the body for a vault note based on its type."""
    if note_type in _CONTENT_KINDS or note_type == "code":
        return str(node.get("body") or "")
    if note_type == "category":
        return _render_category_body(node, slug_map)
    if note_type == "concept":
        return _render_concept_body(node, slug_map)
    return ""


__all__ = [
    "DEFAULT_THRESHOLDS",
    "NoteSpec",
    "build_slug_map",
    "extract_metadata",
    "format_frontmatter",
    "included_nodes",
    "note_type_of",
    "render_body",
]
