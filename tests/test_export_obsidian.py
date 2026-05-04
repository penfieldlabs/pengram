# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.export_obsidian."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from pengram.export_obsidian import (
    export_obsidian,
    render_note,
    render_relationships_section,
)
from pengram.export_penfield import NoteSpec


def _parse_frontmatter(md: str) -> dict:
    assert md.startswith("---\n")
    end = md.index("\n---", 4)
    return yaml.safe_load(md[4:end])


def test_inline_relationships_section_present() -> None:
    # NoteSpec is a low-level fixture — relationships values are already slugs.
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        relationships={"supports": ["[[alpha]]", "[[beta]]"]},
    )
    md = render_note(spec)
    assert "## Relationships" in md
    assert "[[alpha|alpha @supports]]" in md
    assert "[[beta|beta @supports]]" in md


def test_inline_matches_frontmatter() -> None:
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        relationships={
            "supports": ["[[alpha]]"],
            "references": ["[[beta]]"],
        },
    )
    md = render_note(spec)
    fm = _parse_frontmatter(md)
    for relation in ("supports", "references"):
        assert relation in fm
        assert f"@{relation}" in md


def test_render_note_body_starts_immediately_after_frontmatter() -> None:
    """Same invariant as the Penfield exporter: no blank separator line."""
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        body="Body text.",
    )
    md = render_note(spec)
    lines = md.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if line == "---"]
    assert len(fence_indices) >= 2
    assert lines[fence_indices[1] + 1] == "Body text."


def test_render_relationships_section_empty() -> None:
    assert render_relationships_section({}) == ""


def test_render_relationships_section_filters_invalid() -> None:
    out = render_relationships_section({"does_a_flip": ["[[x]]"]})
    assert out == ""


def test_export_obsidian_writes_inline_and_frontmatter(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=3)
    g.add_node("bob", label="Bob", kind="concept", mentions=3)
    g.add_edge("alice", "bob", relation="references")
    vault = export_obsidian(g, tmp_path)
    md = (vault / "concepts" / "alice.md").read_text()
    fm = _parse_frontmatter(md)
    # Wikilink target is the target's slug — which matches its filename.
    assert fm["references"] == ["[[bob]]"]
    assert "- → [[bob|bob @references]]" in md
    assert (vault / "concepts" / "bob.md").exists()


def test_export_obsidian_no_orphan_relationships_section(tmp_path: Path) -> None:
    # Node with no outgoing semantic edges should not get a Relationships section.
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=3)
    g.add_node("_doc", label="doc.md", kind="document")
    g.add_edge("_doc", "alice", relation="references")
    vault = export_obsidian(g, tmp_path)
    md = (vault / "concepts" / "alice.md").read_text()
    assert "## Relationships" not in md


def test_export_obsidian_filenames_are_slugs(tmp_path: Path) -> None:
    """Regression for Bug 8: Obsidian vault filenames are kebab-case."""
    g = nx.DiGraph()
    g.add_node("a", label="Arne Trautmann", kind="concept", mentions=3)
    g.add_node("b", label="Carlson, A.", kind="concept", mentions=3)
    g.add_edge("a", "b", relation="references")
    vault = export_obsidian(g, tmp_path)
    assert (vault / "concepts" / "arne-trautmann.md").exists()
    assert (vault / "concepts" / "carlson-a.md").exists()
    md = (vault / "concepts" / "arne-trautmann.md").read_text()
    # Inline Relationships section uses slug targets too, matching filenames.
    assert "- → [[carlson-a|carlson-a @references]]" in md


def test_obsidian_blank_line_between_body_and_relationships(tmp_path: Path) -> None:
    """CommonMark requires a blank line between a list and a following heading."""
    g = nx.DiGraph()
    g.add_node("a", label="Alpha", kind="concept", mentions=5, note="A note", appearances=["doc_x"])
    g.add_node("doc_x", label="x", kind="document", body="x", source_path="")
    g.add_node("b", label="Beta", kind="concept", mentions=5)
    g.add_edge("doc_x", "a", relation="references")
    g.add_edge("a", "b", relation="supports", confidence="EXTRACTED")

    vault = export_obsidian(g, tmp_path)
    note = (vault / "concepts" / "alpha.md").read_text()
    assert "\n\n## Relationships" in note
    import re

    assert not re.search(r"^- [^\n]+\n## ", note, re.MULTILINE)


def test_image_document_copies_attachment_and_embeds(tmp_path: Path) -> None:
    """Image-derived document gets a copied attachment + a resolving embed."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    img = src_dir / "diagram.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    g = nx.DiGraph()
    g.add_node(
        "doc:diagram",
        kind="document",
        label="diagram.png",
        summary="A test diagram",
        body="A test diagram",
        source_file="diagram.png",
        source_path="",
        _abs_source_path=str(img),
    )
    g.add_node("c", kind="concept", label="C", mentions=5)
    g.add_edge("doc:diagram", "c", relation="references", confidence="EXTRACTED")

    vault = export_obsidian(g, tmp_path, thresholds={"concept": 1})
    note = (vault / "documents" / "diagram-png.md").read_text()
    assert "![[diagram-png.png]]" in note
    assert "## Source" in note
    assert (vault / "documents" / "diagram-png.png").exists()
    assert img.exists()


def test_image_attachment_not_copied_in_penfield(tmp_path: Path) -> None:
    """Penfield vault gets the summary body but no image embed or copy."""
    from pengram.export_penfield import export_penfield

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    img = src_dir / "diagram.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    g = nx.DiGraph()
    g.add_node(
        "doc:diagram",
        kind="document",
        label="diagram.png",
        summary="A test diagram",
        body="A test diagram",
        source_file="diagram.png",
        source_path="",
        _abs_source_path=str(img),
    )
    g.add_node("c", kind="concept", label="C", mentions=5)
    g.add_edge("doc:diagram", "c", relation="references", confidence="EXTRACTED")

    vault = export_penfield(g, tmp_path, thresholds={"concept": 1})
    note = (vault / "documents" / "diagram-png.md").read_text()
    assert "A test diagram" in note
    assert "![[" not in note
    assert not (vault / "documents" / "diagram-png.png").exists()


def test_image_attachment_slug_collision(tmp_path: Path) -> None:
    """Two image docs with the same basename get unique slugs and attachments."""
    for subdir in ("a", "b"):
        d = tmp_path / "src" / subdir
        d.mkdir(parents=True)
        (d / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    g = nx.DiGraph()
    g.add_node(
        "doc:img1",
        kind="document",
        label="img.png",
        body="First",
        source_file="img.png",
        source_path="a",
        _abs_source_path=str(tmp_path / "src" / "a" / "img.png"),
    )
    g.add_node(
        "doc:img2",
        kind="document",
        label="img.png",
        body="Second",
        source_file="img.png",
        source_path="b",
        _abs_source_path=str(tmp_path / "src" / "b" / "img.png"),
    )
    g.add_node("c", kind="concept", label="C", mentions=5)
    g.add_edge("doc:img1", "c", relation="references", confidence="EXTRACTED")
    g.add_edge("doc:img2", "c", relation="references", confidence="EXTRACTED")

    vault = export_obsidian(g, tmp_path, thresholds={"concept": 1})
    pngs = sorted(vault.rglob("*.png"))
    assert len(pngs) == 2
    names = {p.name for p in pngs}
    assert len(names) == 2
