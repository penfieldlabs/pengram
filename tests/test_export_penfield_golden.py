# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Golden-file tests for the Penfield vault export.

These tests build a small but realistic graph, export it, and compare
every file against expected content. They catch silent drift in the
export format — the contract between PENgram and penfield-import.

To update the golden expectations after an intentional format change,
run the tests with ``--update-golden`` (not a real flag — just re-run
and update the expected strings in this file).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from pengram.export_penfield import export_penfield


def _parse_frontmatter(md: str) -> dict:
    assert md.startswith("---\n"), f"Expected frontmatter fence, got: {md[:40]!r}"
    end = md.index("\n---", 4)
    return yaml.safe_load(md[4:end])


def _body_after_frontmatter(md: str) -> str:
    end = md.index("\n---", 4)
    return md[end + 4 :].strip()


def _build_golden_graph() -> nx.DiGraph:
    """A small graph with one of each note type and realistic metadata."""
    g = nx.DiGraph()

    g.add_node(
        "doc_intro",
        label="intro.md",
        kind="document",
        body="An introduction to knowledge graphs.",
        source_file="intro.md",
        source_path="notes",
    )
    g.add_node(
        "c_kg",
        label="Knowledge Graphs",
        kind="concept",
        mentions=5,
        note="Structured representations of real-world entities and relationships.",
        appearances=["doc_intro"],
    )
    g.add_node(
        "c_neo4j",
        label="Neo4j",
        kind="concept",
        mentions=3,
    )
    g.add_node(
        "cat_graph",
        label="Graph Technologies",
        kind="category",
        members=["c_kg", "c_neo4j"],
        mentions=2,
    )
    g.add_node(
        "code_overview",
        label="demo (code)",
        kind="code",
        body="## Languages\n\n- **python** — 3 files",
    )

    g.add_edge("doc_intro", "c_kg", relation="references", confidence="EXTRACTED")
    g.add_edge("doc_intro", "c_neo4j", relation="references", confidence="INFERRED")
    g.add_edge("c_kg", "c_neo4j", relation="references", confidence="INFERRED")
    g.add_edge("cat_graph", "c_kg", relation="parent_of", confidence="EXTRACTED")
    g.add_edge("cat_graph", "c_neo4j", relation="parent_of", confidence="EXTRACTED")

    return g


def test_golden_vault_structure(tmp_path: Path) -> None:
    """Every expected file exists and no unexpected files are created."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)
    written = {p.relative_to(vault).as_posix() for p in vault.rglob("*.md")}
    expected = {
        "documents/notes/intro-md.md",
        "concepts/knowledge-graphs.md",
        "concepts/neo4j.md",
        "categories/graph-technologies.md",
        "code/demo-code.md",
    }
    assert written == expected


def test_golden_concept_frontmatter(tmp_path: Path) -> None:
    """Concept note frontmatter matches the contract exactly."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "concepts" / "knowledge-graphs.md").read_text()
    fm = _parse_frontmatter(md)

    assert fm["note_type"] == "concept"
    assert fm["mentions"] == 5
    assert "knowledge-graphs" in fm["tags"]
    assert "concept-hub" in fm["tags"]
    assert fm["references"] == ["[[neo4j]]"]
    assert "title" not in fm
    assert "type" not in fm
    assert "confidence" not in fm


def test_golden_concept_body(tmp_path: Path) -> None:
    """Concept note body includes the extraction note and appearance list."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "concepts" / "knowledge-graphs.md").read_text()
    body = _body_after_frontmatter(md)

    assert "Structured representations" in body
    assert "## Discussed in" in body
    assert "[[intro-md]]" in body


def test_golden_document_frontmatter(tmp_path: Path) -> None:
    """Document note has source_file, source_path, tags from referenced concepts."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "documents" / "notes" / "intro-md.md").read_text()
    fm = _parse_frontmatter(md)

    assert fm["note_type"] == "document"
    assert fm["source_file"] == "intro.md"
    assert fm["source_path"] == "notes"
    assert "knowledge-graphs" in fm["tags"]
    assert "neo4j" in fm["tags"]
    assert fm["references"] == ["[[knowledge-graphs]]", "[[neo4j]]"]
    assert "mentions" not in fm
    assert "title" not in fm


def test_golden_document_body(tmp_path: Path) -> None:
    """Document body is the verbatim source text."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "documents" / "notes" / "intro-md.md").read_text()
    body = _body_after_frontmatter(md)

    assert body == "An introduction to knowledge graphs."


def test_golden_category_frontmatter(tmp_path: Path) -> None:
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "categories" / "graph-technologies.md").read_text()
    fm = _parse_frontmatter(md)

    assert fm["note_type"] == "category"
    assert "mentions" not in fm
    assert "graph-technologies" in fm["tags"]
    assert "category-hub" in fm["tags"]
    assert set(fm["parent_of"]) == {"[[knowledge-graphs]]", "[[neo4j]]"}


def test_golden_category_body_has_members(tmp_path: Path) -> None:
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "categories" / "graph-technologies.md").read_text()
    body = _body_after_frontmatter(md)

    assert "## Members" in body
    assert "[[knowledge-graphs]]" in body
    assert "[[neo4j]]" in body


def test_golden_code_overview(tmp_path: Path) -> None:
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    md = (vault / "code" / "demo-code.md").read_text()
    fm = _parse_frontmatter(md)
    body = _body_after_frontmatter(md)

    assert fm["note_type"] == "code"
    assert "code-hub" in fm["tags"]
    assert "## Languages" in body
    assert "python" in body


def test_golden_no_relationships_section_in_any_note(tmp_path: Path) -> None:
    """Penfield export never contains an inline Relationships section."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    for md_file in vault.rglob("*.md"):
        content = md_file.read_text()
        assert "## Relationships" not in content, f"Found in {md_file.relative_to(vault)}"


def test_golden_no_dangling_wikilinks(tmp_path: Path) -> None:
    """Every wikilink target in any frontmatter corresponds to a written file."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    slugs_on_disk = {p.stem for p in vault.rglob("*.md")}

    for md_file in vault.rglob("*.md"):
        fm = _parse_frontmatter(md_file.read_text())
        for key, value in fm.items():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, str) and item.startswith("[[") and item.endswith("]]"):
                    target = item[2:-2]
                    assert target in slugs_on_disk, (
                        f"Dangling wikilink {item} in {md_file.relative_to(vault)}"
                    )


def test_golden_idempotent(tmp_path: Path) -> None:
    """Running export twice produces byte-identical files."""
    g = _build_golden_graph()
    vault = export_penfield(g, tmp_path)

    first_pass = {}
    for md_file in vault.rglob("*.md"):
        first_pass[md_file.relative_to(vault)] = md_file.read_text()

    export_penfield(g, tmp_path)

    for rel, content in first_pass.items():
        assert (vault / rel).read_text() == content, f"Drift in {rel}"
