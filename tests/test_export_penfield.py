# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.export_penfield."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from pengram.export_penfield import (
    NoteSpec,
    export_penfield,
    format_frontmatter,
    render_note,
)


def _anchor(g: nx.DiGraph, *concept_ids: str) -> None:
    """Add a document node with edges to each concept so they survive the
    zero-edge filter in included_nodes()."""
    g.add_node("_doc", label="doc.md", kind="document")
    for cid in concept_ids:
        g.add_edge("_doc", cid, relation="references")


def _parse_frontmatter(md: str) -> dict:
    assert md.startswith("---\n")
    end = md.index("\n---", 4)
    return yaml.safe_load(md[4:end])


def test_format_frontmatter_no_top_level_type() -> None:
    spec = NoteSpec(node_id="x", note_type="concept")
    fm = format_frontmatter(spec)
    parsed = yaml.safe_load(fm.strip("-\n"))
    assert "type" not in parsed  # reserved for relationship vocab
    assert parsed["note_type"] == "concept"


def test_format_frontmatter_has_no_title_field() -> None:
    """v0.2.0: ``title`` is gone. The filename IS the title."""
    spec = NoteSpec(node_id="x", note_type="concept")
    parsed = yaml.safe_load(format_frontmatter(spec).strip("-\n"))
    assert "title" not in parsed


def test_frontmatter_only_valid_relationship_keys() -> None:
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        relationships={
            "supports": ["[[A]]"],
            "not_a_real_type": ["[[B]]"],
        },
    )
    rendered = format_frontmatter(spec)
    parsed = _parse_frontmatter(rendered + "\n")
    assert "supports" in parsed
    assert "not_a_real_type" not in parsed


def test_frontmatter_relationship_values_are_wikilink_arrays() -> None:
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        relationships={"references": ["[[Alpha]]", "[[Beta]]"]},
    )
    parsed = _parse_frontmatter(format_frontmatter(spec) + "\n")
    assert parsed["references"] == ["[[Alpha]]", "[[Beta]]"]


def test_render_note_body_starts_immediately_after_frontmatter() -> None:
    """No blank line between the closing ``---`` and the body. Obsidian
    hides the frontmatter block in the rendered view, so a separator
    blank shows up as a blank first visible line. Users noticed."""
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        body="Body text starts here.",
    )
    md = render_note(spec)
    lines = md.split("\n")
    # Find the closing --- (second one).
    fence_indices = [i for i, line in enumerate(lines) if line == "---"]
    assert len(fence_indices) >= 2
    closing_idx = fence_indices[1]
    # The line immediately after the closing fence must be the body —
    # NOT an empty separator line.
    assert lines[closing_idx + 1] == "Body text starts here."


def test_render_note_no_inline_relationships_section() -> None:
    spec = NoteSpec(
        node_id="x",
        note_type="concept",
        body="Some content.",
        relationships={"references": ["[[Alpha]]"]},
    )
    md = render_note(spec)
    assert "## Relationships" not in md
    # v0.2.0: no synthetic '# X' heading — filename is the title.
    assert "# X" not in md
    # Body content still renders.
    assert "Some content." in md


def test_export_penfield_directory_layout(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=3)
    g.add_node("acme", label="Acme", kind="concept", mentions=3)
    g.add_node("graphs", label="Graph Theory", kind="concept", mentions=4)
    _anchor(g, "alice", "acme", "graphs")
    vault = export_penfield(g, tmp_path)
    assert (vault / "concepts" / "alice.md").exists()
    assert (vault / "concepts" / "acme.md").exists()
    assert (vault / "concepts" / "graph-theory.md").exists()


def test_export_penfield_no_title_in_frontmatter(tmp_path: Path) -> None:
    """v0.2.0: frontmatter has no ``title`` field. The filename (slug)
    is the title — Obsidian renders it, Penfield uses the filename."""
    g = nx.DiGraph()
    g.add_node("graphs", label="Graph Theory", kind="concept", mentions=4)
    _anchor(g, "graphs")
    vault = export_penfield(g, tmp_path)
    md = (vault / "concepts" / "graph-theory.md").read_text()
    parsed = _parse_frontmatter(md)
    assert "title" not in parsed
    # Synthetic heading is gone too.
    assert "# Graph Theory" not in md


def test_export_penfield_threshold_filter(tmp_path: Path) -> None:
    # Explicit threshold: person must have at least 2 mentions.
    g = nx.DiGraph()
    g.add_node("bob", label="Bob", kind="concept", mentions=1)
    g.add_node("carol", label="Carol", kind="concept", mentions=2)
    _anchor(g, "bob", "carol")
    vault = export_penfield(g, tmp_path, thresholds={"concept": 2})
    assert not (vault / "concepts" / "bob.md").exists()
    assert (vault / "concepts" / "carol.md").exists()


def test_export_penfield_explicit_no_threshold(tmp_path: Path) -> None:
    """With thresholds={} every node gets a note (v0.1.0's Bug-5 fix still reachable)."""
    g = nx.DiGraph()
    g.add_node("frank", label="Frank", kind="concept", mentions=1)
    g.add_node("karpathy", label="Karpathy", kind="concept", mentions=1)
    g.add_node("cache_behavior", label="cache behavior", kind="concept", mentions=1)
    _anchor(g, "frank", "karpathy", "cache_behavior")
    vault = export_penfield(g, tmp_path, thresholds={})
    assert (vault / "concepts" / "frank.md").exists()
    assert (vault / "concepts" / "karpathy.md").exists()
    assert (vault / "concepts" / "cache-behavior.md").exists()


def test_export_penfield_default_concept_threshold(tmp_path: Path) -> None:
    """v0.1.1 (4.7): default concept threshold is 3 mentions. 1- and 2-mention
    concepts stay on the graph but get no vault note; outgoing links to them
    are dropped so the vault has zero broken wikilinks."""
    g = nx.DiGraph()
    g.add_node("rare", label="Rare", kind="concept", mentions=1)
    g.add_node("sometimes", label="Sometimes", kind="concept", mentions=2)
    g.add_node("common", label="Common", kind="concept", mentions=3)
    _anchor(g, "common")
    g.add_edge("common", "rare", relation="references")
    g.add_edge("common", "sometimes", relation="references")
    vault = export_penfield(g, tmp_path)
    assert not (vault / "concepts" / "rare.md").exists()
    assert not (vault / "concepts" / "sometimes.md").exists()
    common_md = (vault / "concepts" / "common.md").read_text()
    assert "[[rare]]" not in common_md
    assert "[[sometimes]]" not in common_md


def test_export_penfield_no_broken_wikilinks_under_threshold(tmp_path: Path) -> None:
    """Regression for Bug 5: threshold-filtered targets must not appear as wikilinks."""
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=5)
    g.add_node("bob", label="Bob", kind="concept", mentions=1)  # below threshold
    _anchor(g, "alice")
    g.add_edge("alice", "bob", relation="references", confidence="EXTRACTED")
    vault = export_penfield(g, tmp_path, thresholds={"concept": 2})
    alice_md = (vault / "concepts" / "alice.md").read_text()
    # Bob's note was filtered; Alice must not link to it.
    assert "[[bob]]" not in alice_md
    assert "[[Bob]]" not in alice_md
    assert not (vault / "concepts" / "bob.md").exists()


def test_export_penfield_idempotent(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=3)
    _anchor(g, "alice")
    vault1 = export_penfield(g, tmp_path)
    content1 = (vault1 / "concepts" / "alice.md").read_text()
    vault2 = export_penfield(g, tmp_path)
    content2 = (vault2 / "concepts" / "alice.md").read_text()
    assert content1 == content2


def test_export_penfield_writes_relationships(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=3)
    g.add_node("bob", label="Bob", kind="concept", mentions=3)
    g.add_edge("alice", "bob", relation="supports", confidence="EXTRACTED")
    vault = export_penfield(g, tmp_path)
    md = (vault / "concepts" / "alice.md").read_text()
    parsed = _parse_frontmatter(md)
    # Wikilink target is the slug (matches Bob's filename), not the display label.
    assert parsed.get("supports") == ["[[bob]]"]


def test_export_penfield_drops_invalid_relation(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="concept", mentions=3)
    g.add_node("bob", label="Bob", kind="concept", mentions=3)
    g.add_edge("alice", "bob", relation="calls", confidence="EXTRACTED")  # structural, not semantic
    vault = export_penfield(g, tmp_path)
    md = (vault / "concepts" / "alice.md").read_text()
    parsed = _parse_frontmatter(md)
    assert "calls" not in parsed


def test_ast_file_nodes_skipped_from_vault(tmp_path: Path) -> None:
    """v0.1.1 (4.6): raw AST ``file`` nodes no longer produce vault notes.
    A synthesised ``code`` overview note (if emitted) is the only code-
    vault output. AST nodes stay in graph.json for the HTML visualiser."""
    g = nx.DiGraph()
    g.add_node("file1", label="main.py", kind="file")
    vault = export_penfield(g, tmp_path)
    assert not (vault / "code").exists()


def test_ast_class_and_function_nodes_also_skipped(tmp_path: Path) -> None:
    """Regression: ``class`` and ``function`` kinds are not in
    _NOTE_KIND_PLURAL, so ``_note_type_of`` falls back to ``document``.
    Early v0.1.1 checked the fallback type against VAULT_SKIP_KINDS,
    which let AST class/function nodes leak into ``vault/documents/``
    as bare ``bar.md`` / ``foo.md`` files. included_nodes now checks
    the raw kind first."""
    g = nx.DiGraph()
    g.add_node("cls1", label="Bar", kind="class")
    g.add_node("fn1", label="foo", kind="function")
    g.add_node("real", label="Graph Theory", kind="concept", mentions=3)
    _anchor(g, "real")
    vault = export_penfield(g, tmp_path)
    # No AST leakage into documents/.
    assert (
        not any((vault / "documents").rglob("bar.md")) if (vault / "documents").exists() else True
    )
    assert (
        not any((vault / "documents").rglob("foo.md")) if (vault / "documents").exists() else True
    )
    # The real concept still got its note.
    assert (vault / "concepts" / "graph-theory.md").exists()


def test_code_overview_node_writes_to_code_dir(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node(
        "code_overview",
        label="demo (code)",
        kind="code",
        body="## Languages\n\n- **python** — 1 file",
    )
    vault = export_penfield(g, tmp_path)
    overview = vault / "code" / "demo-code.md"
    assert overview.exists()
    assert "## Languages" in overview.read_text()


# ---------------------------------------------------------------------------
# Bug 8: filenames are slugs, wikilinks match filenames, collisions resolved.
# ---------------------------------------------------------------------------


def test_filenames_are_kebab_case(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("a", label="Arne Trautmann", kind="concept", mentions=3)
    g.add_node("b", label="Carlson, A.", kind="concept", mentions=3)
    g.add_node("c", label="Typed Relationships", kind="concept", mentions=3)
    _anchor(g, "a", "b", "c")
    vault = export_penfield(g, tmp_path)
    assert (vault / "concepts" / "arne-trautmann.md").exists()
    assert (vault / "concepts" / "carlson-a.md").exists()
    assert (vault / "concepts" / "typed-relationships.md").exists()


def test_wikilink_target_matches_filename(tmp_path: Path) -> None:
    """Regression for Bug 8: wikilink target must equal target note's filename
    (minus .md). No more [[Arne Trautmann]] → arne-trautmann.md mismatches."""
    g = nx.DiGraph()
    g.add_node("a", label="Arne Trautmann", kind="concept", mentions=3)
    g.add_node("b", label="Carlson, A.", kind="concept", mentions=3)
    g.add_edge("a", "b", relation="references")
    vault = export_penfield(g, tmp_path)
    md = (vault / "concepts" / "arne-trautmann.md").read_text()
    parsed = _parse_frontmatter(md)
    # The wikilink target is the slug, and the target file exists under that slug.
    assert parsed["references"] == ["[[carlson-a]]"]
    assert (vault / "concepts" / "carlson-a.md").exists()


def test_slug_collision_gets_numeric_suffix(tmp_path: Path) -> None:
    """Two labels that slugify identically must land on distinct files."""
    g = nx.DiGraph()
    # Both slugify to "hello-world".
    g.add_node("a", label="Hello World", kind="concept", mentions=3)
    g.add_node("b", label="hello world", kind="concept", mentions=3)
    g.add_node("c", label="HELLO, WORLD!", kind="concept", mentions=3)
    _anchor(g, "a", "b", "c")
    vault = export_penfield(g, tmp_path)
    concepts = {p.name for p in (vault / "concepts").iterdir()}
    assert concepts == {"hello-world.md", "hello-world-2.md", "hello-world-3.md"}


def test_tags_for_concept_hub_use_own_slug_and_hub_marker(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node("c_no", label="Nitric Oxide", kind="concept", mentions=5)
    _anchor(g, "c_no")
    vault = export_penfield(g, tmp_path)
    parsed = _parse_frontmatter((vault / "concepts" / "nitric-oxide.md").read_text())
    assert parsed["tags"] == ["nitric-oxide", "concept-hub"]


def test_tags_for_category_hub_use_own_slug_and_hub_marker(tmp_path: Path) -> None:
    g = nx.DiGraph()
    g.add_node(
        "cat_cv",
        label="Cardio / Nitric Oxide",
        kind="category",
        members=[],
        mentions=4,
    )
    vault = export_penfield(g, tmp_path)
    md = (vault / "categories" / "cardio-nitric-oxide.md").read_text()
    parsed = _parse_frontmatter(md)
    assert parsed["tags"] == ["cardio-nitric-oxide", "category-hub"]


def test_tags_for_document_list_referenced_concepts_and_parent_categories(
    tmp_path: Path,
) -> None:
    """A doc's tags include every concept it references AND every
    category that parents it. That's how the tag surface becomes the
    cross-cutting navigation layer."""
    g = nx.DiGraph()
    g.add_node(
        "doc_n",
        label="nathan_bryan.md",
        kind="document",
        body="NO health benefits " * 50,
        source_path="Experts/Nathan Bryan",
        source_file="nathan_bryan.md",
    )
    g.add_node("c_no", label="Nitric Oxide", kind="concept", mentions=5)
    g.add_node("c_cv", label="Cardiovascular", kind="concept", mentions=4)
    g.add_node(
        "cat_hub",
        label="Cardio Hub",
        kind="category",
        members=["doc_n", "c_no", "c_cv"],
        mentions=3,
    )
    g.add_edge("doc_n", "c_no", relation="references", confidence="EXTRACTED")
    g.add_edge("doc_n", "c_cv", relation="references", confidence="EXTRACTED")
    g.add_edge("cat_hub", "doc_n", relation="parent_of", confidence="EXTRACTED")
    vault = export_penfield(g, tmp_path)
    doc_md = next((vault / "documents").rglob("*.md"))
    parsed = _parse_frontmatter(doc_md.read_text())
    tags = parsed["tags"]
    assert "nitric-oxide" in tags
    assert "cardiovascular" in tags
    assert "cardio-hub" in tags
    # And definitely NOT the useless literal 'document' tag.
    assert "document" not in tags


def test_frontmatter_drops_confidence_and_filters_mentions(tmp_path: Path) -> None:
    """v0.2.1: confidence is gone from frontmatter; mentions only on concepts."""
    g = nx.DiGraph()
    g.add_node(
        "doc_1",
        label="a.md",
        kind="document",
        body="x",
        confidence="EXTRACTED",
        mentions=1,
        source_path="sub",
        source_file="a.md",
    )
    g.add_node(
        "c_x",
        label="X",
        kind="concept",
        mentions=5,
        confidence="EXTRACTED",
    )
    _anchor(g, "c_x")
    vault = export_penfield(g, tmp_path)
    doc_md = next((vault / "documents").rglob("*.md"))
    doc_fm = _parse_frontmatter(doc_md.read_text())
    assert "confidence" not in doc_fm
    assert "mentions" not in doc_fm  # dropped on document
    concept_md = (vault / "concepts" / "x.md").read_text()
    concept_fm = _parse_frontmatter(concept_md)
    assert "confidence" not in concept_fm
    assert concept_fm["mentions"] == 5  # kept on concept


def test_frontmatter_source_path_empty_is_omitted(tmp_path: Path) -> None:
    """A document at the corpus root has source_path='' which must NOT
    appear in the frontmatter at all."""
    g = nx.DiGraph()
    g.add_node(
        "doc_root",
        label="readme.md",
        kind="document",
        body="x",
        source_path="",
        source_file="readme.md",
    )
    vault = export_penfield(g, tmp_path)
    md = next((vault / "documents").rglob("*.md"))
    parsed = _parse_frontmatter(md.read_text())
    assert "source_path" not in parsed
    assert parsed["source_file"] == "readme.md"


def test_frontmatter_source_file_is_basename_only(tmp_path: Path) -> None:
    """source_file must never contain an absolute path — the build
    machine's layout is nobody else's business."""
    g = nx.DiGraph()
    g.add_node(
        "doc_x",
        label="book.md",
        kind="document",
        body="x",
        source_file="book.md",  # basename only — set by the pipeline
        source_path="Experts/Nathan",
    )
    vault = export_penfield(g, tmp_path)
    md = next((vault / "documents").rglob("*.md"))
    parsed = _parse_frontmatter(md.read_text())
    assert parsed["source_file"] == "book.md"
    assert "/" not in parsed["source_file"]
    assert parsed["source_path"] == "Experts/Nathan"


def test_min_confidence_extracted_drops_inferred_wikilinks(tmp_path: Path) -> None:
    """v0.2.0 (3.7): min_confidence='EXTRACTED' filters INFERRED edges
    out of vault frontmatter wikilinks. graph.json still contains them."""
    g = nx.DiGraph()
    g.add_node("a", label="Alice", kind="concept", mentions=5)
    g.add_node("b", label="Bob", kind="concept", mentions=5)
    g.add_node("c", label="Carol", kind="concept", mentions=5)
    g.add_edge("a", "b", relation="supports", confidence="EXTRACTED")
    g.add_edge("a", "c", relation="supports", confidence="INFERRED")
    vault = export_penfield(g, tmp_path, min_confidence="EXTRACTED")
    md = (vault / "concepts" / "alice.md").read_text()
    parsed = _parse_frontmatter(md)
    assert parsed.get("supports") == ["[[bob]]"]
    assert "[[carol]]" not in md


def test_min_confidence_inferred_keeps_both_tiers(tmp_path: Path) -> None:
    """min_confidence='INFERRED' keeps both EXTRACTED and INFERRED edges."""
    g = nx.DiGraph()
    g.add_node("a", label="Alice", kind="concept", mentions=5)
    g.add_node("b", label="Bob", kind="concept", mentions=5)
    g.add_node("c", label="Carol", kind="concept", mentions=5)
    g.add_edge("a", "b", relation="supports", confidence="EXTRACTED")
    g.add_edge("a", "c", relation="supports", confidence="INFERRED")
    vault = export_penfield(g, tmp_path, min_confidence="INFERRED")
    md = (vault / "concepts" / "alice.md").read_text()
    parsed = _parse_frontmatter(md)
    assert set(parsed.get("supports") or []) == {"[[bob]]", "[[carol]]"}


def test_min_confidence_default_keeps_everything(tmp_path: Path) -> None:
    """Default (None) doesn't filter — matches pre-v0.2.0 behaviour."""
    g = nx.DiGraph()
    g.add_node("a", label="Alice", kind="concept", mentions=5)
    g.add_node("b", label="Bob", kind="concept", mentions=5)
    g.add_edge("a", "b", relation="supports", confidence="INFERRED")
    vault = export_penfield(g, tmp_path)
    md = (vault / "concepts" / "alice.md").read_text()
    parsed = _parse_frontmatter(md)
    assert parsed.get("supports") == ["[[bob]]"]


def test_legacy_person_org_kinds_are_unknown_and_skip_vault(tmp_path: Path) -> None:
    """The entity-purge directive removed person/organization entirely.
    If a caller somehow hands the exporter a node with one of these kinds
    (stale hand-built graph, downstream consumer error), it must NOT
    produce ``vault/people/`` or ``vault/organizations/`` directories.

    With NOTE_KIND_PLURAL no longer listing those kinds, ``_note_type_of``
    falls back to ``'document'`` for unknown labels — and the threshold
    for documents is 0, so the node lands under ``vault/documents/`` with
    its label intact. The important invariant is the absence of the old
    dirs, not where the orphan lands.
    """
    g = nx.DiGraph()
    g.add_node("alice", label="Alice", kind="person", mentions=5)
    g.add_node("acme", label="Acme", kind="organization", mentions=5)
    g.add_node("graphs", label="Graph Theory", kind="concept", mentions=5)
    _anchor(g, "graphs")
    vault = export_penfield(g, tmp_path)
    assert not (vault / "people").exists()
    assert not (vault / "organizations").exists()
    assert (vault / "concepts" / "graph-theory.md").exists()


def test_no_trailing_dot_broken_wikilink(tmp_path: Path) -> None:
    """Regression for Bug 8 / the last 8% of Bug 5: `Carlson, A.` used to yield
    a trailing-dot-stripped filename whose wikilinks still contained the dot."""
    g = nx.DiGraph()
    g.add_node("src", label="Alice", kind="concept", mentions=3)
    g.add_node("dst", label="Carlson, A.", kind="concept", mentions=3)
    g.add_edge("src", "dst", relation="references")
    vault = export_penfield(g, tmp_path)
    md = (vault / "concepts" / "alice.md").read_text()
    parsed = _parse_frontmatter(md)
    assert parsed["references"] == ["[[carlson-a]]"]
    assert (vault / "concepts" / "carlson-a.md").exists()


def test_concept_with_zero_vault_edges_excluded(tmp_path: Path) -> None:
    """Concept meeting threshold but with no edges to other included nodes
    is excluded from vault output."""
    g = nx.DiGraph()
    g.add_node("lonely", label="Lonely", kind="concept", mentions=5)
    g.add_node("connected", label="Connected", kind="concept", mentions=5)
    _anchor(g, "connected")
    vault = export_penfield(g, tmp_path)
    assert not (vault / "concepts" / "lonely.md").exists()
    assert (vault / "concepts" / "connected.md").exists()


def test_penfield_and_obsidian_have_identical_node_counts(tmp_path: Path) -> None:
    """Both export formats use the same included_nodes() filter."""
    from pengram.export_obsidian import export_obsidian

    g = nx.DiGraph()
    g.add_node("a", label="Alpha", kind="concept", mentions=5)
    g.add_node("b", label="Beta", kind="concept", mentions=5)
    g.add_node("lone", label="Lone", kind="concept", mentions=5)
    g.add_edge("a", "b", relation="references")
    pen = export_penfield(g, tmp_path / "pen")
    obs = export_obsidian(g, tmp_path / "obs")
    pen_notes = set((pen / "concepts").iterdir())
    obs_notes = set((obs / "concepts").iterdir())
    assert {p.name for p in pen_notes} == {p.name for p in obs_notes}


def test_threshold_category_filters_small_categories(tmp_path: Path) -> None:
    """--threshold-category N drops categories with fewer than N members."""
    g = nx.DiGraph()
    g.add_node("cat_big", label="Big", kind="category", members=["a", "b", "c"])
    g.add_node("cat_tiny", label="Tiny", kind="category", members=["d"])
    g.add_node("a", label="A", kind="concept", mentions=5)
    g.add_node("b", label="B", kind="concept", mentions=5)
    g.add_node("c", label="C", kind="concept", mentions=5)
    g.add_node("d", label="D", kind="concept", mentions=5)
    g.add_edge("cat_big", "a", relation="parent_of")
    g.add_edge("cat_big", "b", relation="parent_of")
    g.add_edge("cat_big", "c", relation="parent_of")
    g.add_edge("cat_tiny", "d", relation="parent_of")
    g.add_edge("a", "b", relation="references")
    g.add_edge("b", "c", relation="references")
    g.add_edge("c", "d", relation="references")

    vault = export_penfield(g, tmp_path, thresholds={"concept": 1, "category": 2})
    cats = list((vault / "categories").rglob("*.md"))
    cat_names = {c.stem for c in cats}
    assert "big" in cat_names
    assert "tiny" not in cat_names


def test_child_of_backlinks_in_penfield_export(tmp_path: Path) -> None:
    """Member files must have child_of back-links to their categories."""
    g = nx.DiGraph()
    g.add_node("cat_tech", label="Technology", kind="category", members=["c_ai"])
    g.add_node("c_ai", label="AI", kind="concept", mentions=5)
    g.add_edge("cat_tech", "c_ai", relation="parent_of")
    _anchor(g, "c_ai")

    vault = export_penfield(g, tmp_path)
    concept_md = (vault / "concepts" / "ai.md").read_text()
    fm = _parse_frontmatter(concept_md)
    assert "child_of" in fm
    assert "[[technology]]" in fm["child_of"]


def test_tag_count_capped_at_ten(tmp_path: Path) -> None:
    """Tags must not exceed penfield-import's 10-tag limit."""
    g = nx.DiGraph()
    g.add_node("doc", kind="document", label="d", body="x", source_path="")
    for i in range(15):
        cid = f"c_{i}"
        g.add_node(cid, kind="concept", label=f"C{i}", mentions=5)
        g.add_edge("doc", cid, relation="references")
        g.add_edge(cid, "doc", relation="references")

    vault = export_penfield(g, tmp_path)
    note = (vault / "documents" / "d.md").read_text()
    fm = _parse_frontmatter(note)
    assert len(fm["tags"]) <= 10


def test_self_loop_excluded_from_frontmatter(tmp_path: Path) -> None:
    """Self-referencing edges must not appear in vault frontmatter."""
    g = nx.DiGraph()
    g.add_node("c", kind="concept", label="C", mentions=5)
    g.add_node("d", kind="document", label="d", body="x", source_path="")
    g.add_edge("c", "c", relation="references", confidence="EXTRACTED")
    g.add_edge("d", "c", relation="references", confidence="EXTRACTED")

    vault = export_penfield(g, tmp_path)
    text = (vault / "concepts" / "c.md").read_text()
    fm = _parse_frontmatter(text)
    assert "references" not in fm
