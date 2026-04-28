# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.enrich."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

from pengram.enrich import (
    EnrichmentResult,
    apply_enrichment,
    enrich_concepts,
    gather_contexts,
)

# ---------------------------------------------------------------------------
# gather_contexts
# ---------------------------------------------------------------------------


def _seed_graph_with_document(
    concept_label: str,
    body: str,
    mentions: int = 3,
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(
        "doc_1",
        label="notes.md",
        kind="document",
        body=body,
        source_file="/fake/notes.md",
    )
    g.add_node(
        "concept_x",
        label=concept_label,
        kind="concept",
        mentions=mentions,
    )
    g.add_edge("doc_1", "concept_x", relation="references", confidence="EXTRACTED")
    return g


def test_gather_contexts_full_doc_for_short_source() -> None:
    """Docs under the full-doc cutoff are passed whole, no window slicing."""
    body = "A tiny note about graphs."
    g = _seed_graph_with_document("graphs", body)
    contexts = gather_contexts(g, "concept_x")
    assert len(contexts) == 1
    assert contexts[0].text == body
    assert contexts[0].doc_label == "notes.md"


def test_gather_contexts_window_around_mentions_for_long_source() -> None:
    """Long docs yield ±window slices around each mention of the concept label."""
    filler = "x" * 5000
    body = f"{filler} graphs are cool {filler} graphs again {filler}"
    g = _seed_graph_with_document("graphs", body)
    contexts = gather_contexts(g, "concept_x", window=500)
    assert len(contexts) == 1  # one ConceptContext per source doc
    assert "graphs are cool" in contexts[0].text
    # Two mention windows get joined within a single ConceptContext.
    assert "graphs again" in contexts[0].text


def test_gather_contexts_skips_docs_without_mentions() -> None:
    """A long doc that edges into the concept but never names it yields nothing."""
    filler = "x" * 20000
    g = _seed_graph_with_document("graphs", filler)  # no 'graphs' in body
    assert gather_contexts(g, "concept_x", window=500) == []


def test_gather_contexts_unknown_concept_returns_empty() -> None:
    g = nx.DiGraph()
    assert gather_contexts(g, "missing") == []


# ---------------------------------------------------------------------------
# enrich_concepts — LLM is mocked
# ---------------------------------------------------------------------------


def test_enrich_concepts_parses_definition_and_quotes() -> None:
    body = "Graphs are networks of typed edges. Graphs are useful."
    g = _seed_graph_with_document("graphs", body)
    communities = {0: ["doc_1", "concept_x"]}

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "definition": "Graphs are networks of typed edges.",
                "quotes": [{"text": "Graphs are useful.", "source": "notes.md"}],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    results = enrich_concepts(g, communities, workers=1, llm=fake_llm)
    assert len(results) == 1
    assert results[0].concept_id == "concept_x"
    assert "networks of typed edges" in results[0].definition
    assert results[0].quotes[0]["text"] == "Graphs are useful."


def test_enrich_concepts_isolates_failures() -> None:
    """One bad LLM response doesn't drop other concepts' enrichments."""
    from pengram.llm import LLMError

    body = "graphs and trees " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    g.add_node("c_a", label="graphs", kind="concept", mentions=5)
    g.add_node("c_b", label="trees", kind="concept", mentions=5)
    g.add_edge("doc_1", "c_a", relation="references", confidence="EXTRACTED")
    g.add_edge("doc_1", "c_b", relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_a", "c_b"]}

    def fake_llm(prompt: str, **kw: Any) -> str:
        if "graphs" in prompt.split("OTHER CONCEPTS", 1)[0]:
            raise LLMError("bad response")
        return json.dumps(
            {
                "definition": "Trees are acyclic graphs.",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    results = enrich_concepts(g, communities, workers=1, llm=fake_llm)
    by_id = {r.concept_id: r for r in results}
    # Both concepts produced a result — c_a has an empty one from the
    # failure branch; c_b has the real enrichment.
    assert "c_b" in by_id
    assert by_id["c_b"].definition.startswith("Trees")


def test_enrich_concepts_emits_progress_messages(capsys: Any) -> None:
    """Regression for v0.2.0 Bug 3: enrichment must log progress so the
    user knows the 20-minute pass is alive."""
    # 20 concepts, workers=1 so output is deterministic.
    body = "x " * 500 + " ".join(f"c{i} " * 10 for i in range(20))
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    for i in range(20):
        nid = f"c_{i}"
        g.add_node(nid, label=f"c{i}", kind="concept", mentions=3)
        g.add_edge("doc_1", nid, relation="references", confidence="EXTRACTED")
    communities = {0: list(g.nodes)}

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "definition": "d",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    enrich_concepts(g, communities, workers=1, llm=fake_llm)
    out = capsys.readouterr().out
    # Progress line appears at least once before the final tally.
    progress_lines = [line for line in out.splitlines() if "Enrichment:" in line and "/20" in line]
    # With 20 concepts and progress_step = max(1, 20//10) = 2, we expect
    # roughly 10 progress lines including the final "20/20".
    assert len(progress_lines) >= 2
    # The last line reports the total.
    assert "20/20" in progress_lines[-1]


def test_enrich_concepts_skips_concepts_with_no_source() -> None:
    g = nx.DiGraph()
    g.add_node("c_x", label="ghost", kind="concept", mentions=3)  # no doc

    def fake_llm(prompt: str, **kw: Any) -> str:
        raise AssertionError("should not be called")

    results = enrich_concepts(g, {0: ["c_x"]}, workers=1, llm=fake_llm)
    assert results == []


# ---------------------------------------------------------------------------
# apply_enrichment — merging
# ---------------------------------------------------------------------------


def test_apply_enrichment_merges_duplicate_into_canonical() -> None:
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body="AMPK " * 50)
    g.add_node("concept_ampk", label="AMPK", kind="concept", mentions=5)
    g.add_node("concept_amp_kinase", label="AMP kinase", kind="concept", mentions=3)
    g.add_edge("doc_1", "concept_ampk", relation="references", confidence="EXTRACTED")
    g.add_edge("doc_1", "concept_amp_kinase", relation="references", confidence="EXTRACTED")

    results = [
        EnrichmentResult(
            concept_id="concept_amp_kinase",
            merge_into_label="AMPK",
        ),
        EnrichmentResult(
            concept_id="concept_ampk",
            definition="AMP-activated protein kinase.",
        ),
    ]
    merges, enriched = apply_enrichment(g, results)
    assert merges == 1
    assert "concept_amp_kinase" not in g.nodes
    assert g.nodes["concept_ampk"].get("definition") == "AMP-activated protein kinase."
    # Mentions summed.
    assert g.nodes["concept_ampk"]["mentions"] == 5 + 3
    # The merged node's references edge redirected to the canonical.
    assert g.has_edge("doc_1", "concept_ampk")


def test_apply_enrichment_chains_merges_via_union_find() -> None:
    """A → B → C flattens so every node in the chain resolves to C."""
    g = nx.DiGraph()
    for nid, label, mentions in [
        ("a", "Belief", 2),
        ("b", "Beliefs", 3),
        ("c", "Believing", 5),
    ]:
        g.add_node(nid, label=label, kind="concept", mentions=mentions)
    g.add_node("doc_1", label="notes.md", kind="document", body="...")
    for nid in ("a", "b", "c"):
        g.add_edge("doc_1", nid, relation="references", confidence="EXTRACTED")

    results = [
        EnrichmentResult(concept_id="a", merge_into_label="Beliefs"),
        EnrichmentResult(concept_id="b", merge_into_label="Believing"),
        EnrichmentResult(concept_id="c", definition="The mental act."),
    ]
    apply_enrichment(g, results)
    # Union-find chose the highest-mention concept ('c' / Believing) as root.
    assert "c" in g.nodes
    assert "a" not in g.nodes
    assert "b" not in g.nodes
    assert g.nodes["c"]["mentions"] == 2 + 3 + 5


def test_apply_enrichment_ignores_unknown_merge_target() -> None:
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body="x")
    g.add_node("a", label="graphs", kind="concept", mentions=3)
    g.add_edge("doc_1", "a", relation="references", confidence="EXTRACTED")

    results = [
        EnrichmentResult(concept_id="a", merge_into_label="ghost-concept"),
    ]
    merges, _ = apply_enrichment(g, results)
    assert merges == 0
    assert "a" in g.nodes


def test_apply_enrichment_without_results_is_noop() -> None:
    g = nx.DiGraph()
    g.add_node("a", label="x", kind="concept", mentions=3)
    merges, enriched = apply_enrichment(g, [])
    assert merges == 0
    assert enriched == 0
    assert "a" in g.nodes


# ---------------------------------------------------------------------------
# v0.2.1 incremental enrichment cache
# ---------------------------------------------------------------------------


def test_enrich_concepts_writes_cache_and_second_run_skips_llm(
    tmp_path,
) -> None:
    """First run enriches and caches; second run uses cache, no LLM call."""

    body = "concept-x body " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    g.add_node("c_x", label="concept-x", kind="concept", mentions=3)
    g.add_edge("doc_1", "c_x", relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_x"]}

    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "definition": "fresh",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    cache_dir = tmp_path / "enrichment_cache"

    # First run: one LLM call, one cache file.
    r1 = enrich_concepts(
        g,
        communities,
        workers=1,
        llm=fake_llm,
        cache_dir=cache_dir,
    )
    assert len(r1) == 1
    assert calls["n"] == 1
    assert (cache_dir / "c_x.json").exists()

    # Second run: zero fresh calls, result loaded from disk.
    r2 = enrich_concepts(
        g,
        communities,
        workers=1,
        llm=fake_llm,
        cache_dir=cache_dir,
    )
    assert len(r2) == 1
    assert calls["n"] == 1  # unchanged
    assert r2[0].definition == "fresh"


def test_enrich_concepts_only_enriches_new_concepts(tmp_path) -> None:
    """Add a concept after the first run; only that one hits the LLM."""

    cache_dir = tmp_path / "enrichment_cache"
    # Pre-seed one cached concept.
    cache_dir.mkdir()
    (cache_dir / "c_old.json").write_text(
        json.dumps(
            {
                "concept_id": "c_old",
                "definition": "from cache",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )
    )

    body = "old new " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    g.add_node("c_old", label="old", kind="concept", mentions=3)
    g.add_node("c_new", label="new", kind="concept", mentions=3)
    for cid in ("c_old", "c_new"):
        g.add_edge("doc_1", cid, relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_old", "c_new"]}

    calls: list[str] = []

    def fake_llm(prompt: str, **kw: Any) -> str:
        # Record which concept is being enriched (the prompt embeds the name).
        for line in prompt.splitlines():
            if line.startswith("  name:"):
                calls.append(line.split(":", 1)[1].strip())
                break
        return json.dumps(
            {
                "definition": "fresh",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    results = enrich_concepts(
        g,
        communities,
        workers=1,
        llm=fake_llm,
        cache_dir=cache_dir,
    )
    # Only c_new went to the LLM; c_old came from cache.
    assert calls == ["new"]
    by_id = {r.concept_id: r for r in results}
    assert by_id["c_old"].definition == "from cache"
    assert by_id["c_new"].definition == "fresh"


def test_enrich_concepts_crash_recovery_resumes_from_cache(tmp_path) -> None:
    """A crash during enrichment preserves work already written to cache."""
    from pengram.llm import LLMError

    cache_dir = tmp_path / "enrichment_cache"

    body = "a b c " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    for nid, label in (("c_a", "a"), ("c_b", "b"), ("c_c", "c")):
        g.add_node(nid, label=label, kind="concept", mentions=3)
        g.add_edge("doc_1", nid, relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_a", "c_b", "c_c"]}

    # First pass: c_b raises; c_a and c_c succeed and write cache.
    def crashing_llm(prompt: str, **kw: Any) -> str:
        for line in prompt.splitlines():
            if line.startswith("  name:"):
                name = line.split(":", 1)[1].strip()
                if name == "b":
                    raise LLMError("network glitch")
                return json.dumps(
                    {
                        "definition": f"def-{name}",
                        "quotes": [],
                        "cross_source_notes": "",
                        "merge_into": None,
                    }
                )
        return json.dumps(
            {"definition": "", "quotes": [], "cross_source_notes": "", "merge_into": None}
        )

    enrich_concepts(
        g,
        communities,
        workers=1,
        llm=crashing_llm,
        cache_dir=cache_dir,
    )
    # Two cache files landed (a + c). b failed and did NOT cache.
    assert (cache_dir / "c_a.json").exists()
    assert (cache_dir / "c_c.json").exists()
    assert not (cache_dir / "c_b.json").exists()

    # Second pass: now b's LLM works; a and c should not be re-called.
    calls: list[str] = []

    def healthy_llm(prompt: str, **kw: Any) -> str:
        for line in prompt.splitlines():
            if line.startswith("  name:"):
                calls.append(line.split(":", 1)[1].strip())
                break
        return json.dumps(
            {
                "definition": "fresh",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    enrich_concepts(
        g,
        communities,
        workers=1,
        llm=healthy_llm,
        cache_dir=cache_dir,
    )
    # Only b hit the LLM on the second pass.
    assert calls == ["b"]


def test_clear_cache_removes_all_entries(tmp_path) -> None:
    from pengram.enrich import clear_cache

    cache_dir = tmp_path / "ec"
    cache_dir.mkdir()
    for i in range(3):
        (cache_dir / f"c_{i}.json").write_text("{}")
    removed = clear_cache(cache_dir)
    assert removed == 3
    assert list(cache_dir.glob("*.json")) == []


def test_clear_cache_on_missing_dir_returns_zero(tmp_path) -> None:
    from pengram.enrich import clear_cache

    assert clear_cache(tmp_path / "never-exists") == 0


def test_parse_failure_does_not_poison_cache(tmp_path) -> None:
    """Malformed LLM response yields an empty result — must NOT be cached.

    Otherwise the concept gets pinned to a permanent no-op: subsequent
    runs would see a cache hit and never retry. The caller swaps in a
    working LLM on the second run; both concepts must be re-attempted
    and cached with real content.
    """
    body = "concept-x body " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    g.add_node("c_x", label="concept-x", kind="concept", mentions=3)
    g.add_edge("doc_1", "c_x", relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_x"]}

    cache_dir = tmp_path / "enrichment_cache"

    def garbage_llm(prompt: str, **kw: Any) -> str:
        return "not json at all"

    # First run: garbage response, empty result, cache stays clean.
    r1 = enrich_concepts(
        g,
        communities,
        workers=1,
        llm=garbage_llm,
        cache_dir=cache_dir,
    )
    assert len(r1) == 1
    assert r1[0].definition == ""
    assert not (cache_dir / "c_x.json").exists()

    # Second run with a healthy LLM: concept is retried because no cache
    # entry was ever written, and the real result lands on disk.
    def healthy_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "definition": "real",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    r2 = enrich_concepts(
        g,
        communities,
        workers=1,
        llm=healthy_llm,
        cache_dir=cache_dir,
    )
    assert r2[0].definition == "real"
    assert (cache_dir / "c_x.json").exists()


def test_cache_entries_are_byte_identical_across_runs(tmp_path) -> None:
    """Idempotency: two runs with the same inputs produce identical cache
    files. v0.2.1 dropped the ``enriched_at`` timestamp specifically so
    this invariant holds — otherwise re-running always dirtied the cache.
    """
    body = "concept-x body " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    g.add_node("c_x", label="concept-x", kind="concept", mentions=3)
    g.add_edge("doc_1", "c_x", relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_x"]}

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "definition": "same",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    cache_a = tmp_path / "a"
    cache_b = tmp_path / "b"
    enrich_concepts(
        g,
        communities,
        workers=1,
        llm=fake_llm,
        cache_dir=cache_a,
    )
    enrich_concepts(
        g,
        communities,
        workers=1,
        llm=fake_llm,
        cache_dir=cache_b,
    )
    assert (cache_a / "c_x.json").read_bytes() == (cache_b / "c_x.json").read_bytes()


def test_empty_enrichment_logs_warning(tmp_path: Any, capsys: Any) -> None:
    """Empty enrichment result must produce a visible warning."""
    body = "concept-x body " * 200
    g = nx.DiGraph()
    g.add_node("doc_1", label="notes.md", kind="document", body=body)
    g.add_node("c_x", label="concept-x", kind="concept", mentions=3)
    g.add_edge("doc_1", "c_x", relation="references", confidence="EXTRACTED")
    communities = {0: ["doc_1", "c_x"]}

    warnings: list[str] = []
    import pengram._ui

    orig_warn = pengram._ui._handler

    def capture_warn(msg: str) -> None:
        warnings.append(msg)

    pengram._ui._handler = capture_warn
    try:
        enrich_concepts(
            g,
            communities,
            workers=1,
            llm=lambda prompt, **kw: "not json",
            cache_dir=tmp_path / "cache",
        )
    finally:
        pengram._ui._handler = orig_warn

    assert any("enrichment returned empty for c_x" in w for w in warnings)


def test_apply_enrichment_promotes_note_to_definition() -> None:
    """Concepts with no definition after enrichment fall back to their note."""
    g = nx.DiGraph()
    g.add_node("a", label="Alpha", kind="concept", mentions=3, note="short desc")
    g.add_node("b", label="Beta", kind="concept", mentions=3, note="another desc")
    # Only 'a' gets a real enrichment; 'b' gets empty.
    results = [
        EnrichmentResult(concept_id="a", definition="full definition"),
        EnrichmentResult(concept_id="b"),
    ]
    apply_enrichment(g, results)
    assert g.nodes["a"]["definition"] == "full definition"
    assert g.nodes["b"]["definition"] == "another desc"
