# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.extract_llm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pengram.extract_llm import (
    Document,
    canonicalize_entities,
    extract_entities,
    extract_many,
)

# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


def test_extract_entities_parses_plain_json() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "concepts": [{"name": "graph theory", "mentions": 1}],
                "events": [{"name": "ICML 2024", "date": "2024-07"}],
                "summary": "A note about graph theory.",
            }
        )

    doc = Document(doc_id="d1", text="...")
    result = extract_entities(doc, llm=fake_llm)
    assert result["concepts"][0]["name"] == "graph theory"
    assert result["_doc_id"] == "d1"


def test_extract_entities_strips_markdown_fence() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return "```json\n" + json.dumps({"concepts": [], "summary": "ok"}) + "\n```"

    doc = Document(doc_id="d1", text="...")
    result = extract_entities(doc, llm=fake_llm)
    assert result["summary"] == "ok"
    assert result["concepts"] == []


def test_extract_entities_fills_missing_keys() -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps({"summary": "minimal"})

    doc = Document(doc_id="d1", text="...")
    result = extract_entities(doc, llm=fake_llm)
    assert result["concepts"] == []
    assert "events" not in result
    assert result["summary"] == "minimal"


def test_extract_entities_only_extracts_concepts() -> None:
    """Regression: v0.1.1 purged people/orgs/predictions/topics; v0.2.0
    added events to the banned-kinds list. Only ``concepts`` + ``summary``
    survive the whitelist."""

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "people": [{"name": "Alice", "mentions": 1}],
                "organizations": [{"name": "OpenAI", "mentions": 1}],
                "predictions": [{"statement": "AI takes over"}],
                "topics": ["ai"],
                "concepts": [{"name": "agent framework", "mentions": 2}],
                "summary": "ignored",
            }
        )

    doc = Document(doc_id="d1", text="...")
    result = extract_entities(doc, llm=fake_llm)
    for key in ("people", "organizations", "predictions", "topics", "events"):
        assert key not in result
    assert result["concepts"][0]["name"] == "agent framework"


def test_extract_entities_folds_legacy_events_into_concepts() -> None:
    """v0.2.0 kill-events directive, Option A: legacy LLM output with an
    ``events`` array merges those entries into ``concepts`` on the way in.
    This makes cached pre-v0.2.0 extractions survive the transition."""

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 2, "note": "c"}],
                "events": [
                    {"name": "CALERIE trial", "mentions": 3, "note": "study"},
                    {"name": "cold exposure", "mentions": 2},
                ],
                "summary": "s",
            }
        )

    doc = Document(doc_id="d1", text="...")
    result = extract_entities(doc, llm=fake_llm)
    assert "events" not in result
    names = {c.get("name") for c in result["concepts"] if isinstance(c, dict)}
    assert names == {"graphs", "CALERIE trial", "cold exposure"}


# ---------------------------------------------------------------------------
# extract_many
# ---------------------------------------------------------------------------


def test_extract_many_caches_to_disk(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps({"concepts": [], "events": [], "summary": "s"})

    docs = [Document(doc_id="a", text="..."), Document(doc_id="b", text="...")]
    # First run extracts both.
    extract_many(docs, output_dir=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 2
    # Second run uses cache, no new LLM calls.
    extract_many(docs, output_dir=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 2
    assert (tmp_path / "extractions" / "a.json").exists()
    assert (tmp_path / "extractions" / "b.json").exists()


def test_extract_many_parallel_workers(tmp_path: Path) -> None:
    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps({"concepts": [], "events": [], "summary": "s"})

    docs = [Document(doc_id=f"d{i}", text="...") for i in range(4)]
    results = extract_many(docs, output_dir=tmp_path, workers=3, llm=fake_llm)
    assert len(results) == 4


# ---------------------------------------------------------------------------
# canonicalize_entities
# ---------------------------------------------------------------------------


def test_canonicalize_merges_by_name() -> None:
    extractions = [
        {"concepts": [{"name": "Graphs", "mentions": 1}]},
        {"concepts": [{"name": "graphs", "mentions": 2}]},
    ]
    result = canonicalize_entities(extractions)
    assert len(result["concepts"]) == 1
    assert result["concepts"][0]["mentions"] == 3


def test_canonicalize_preserves_kinds() -> None:
    extractions = [
        {"concepts": [{"name": "C", "mentions": 1}]},
    ]
    result = canonicalize_entities(extractions)
    assert "concepts" in result
    # Legacy / dropped kinds must not appear.
    for legacy in ("people", "organizations", "predictions", "events"):
        assert legacy not in result


def test_canonicalize_folds_legacy_events() -> None:
    """Cached extractions from before the v0.2.0 kill-events directive
    still contain an ``events`` array. Canonicalize absorbs them into
    ``concepts`` via _fold_legacy_events."""
    extractions = [
        {
            "concepts": [{"name": "graphs", "mentions": 1}],
            "events": [{"name": "CALERIE trial", "mentions": 2}],
        },
    ]
    result = canonicalize_entities(extractions)
    names = {c["name"] for c in result["concepts"]}
    assert names == {"graphs", "CALERIE trial"}


def test_canonicalize_skips_unnamed() -> None:
    extractions = [
        {
            "concepts": [{"mentions": 1}, {"name": "", "mentions": 1}],
        }
    ]
    result = canonicalize_entities(extractions)
    assert result["concepts"] == []


def test_canonicalize_merges_plurals() -> None:
    """Plural variants should merge during canonicalization, not wait for enrichment."""
    extractions = [
        {"concepts": [{"name": "Belief", "mentions": 5}]},
        {"concepts": [{"name": "Beliefs", "mentions": 2}]},
    ]
    result = canonicalize_entities(extractions)
    assert len(result["concepts"]) == 1
    assert result["concepts"][0]["mentions"] == 7
    assert result["concepts"][0]["name"] == "Belief"


def test_canonicalize_merges_hyphens() -> None:
    """Hyphenated and non-hyphenated forms should merge."""
    extractions = [
        {"concepts": [{"name": "machine learning", "mentions": 3}]},
        {"concepts": [{"name": "machine-learning", "mentions": 1}]},
    ]
    result = canonicalize_entities(extractions)
    assert len(result["concepts"]) == 1
    assert result["concepts"][0]["mentions"] == 4
    assert result["concepts"][0]["name"] == "machine learning"


def test_canonicalize_keeps_highest_mention_form() -> None:
    """When merging fuzzy variants, the form with more mentions wins."""
    extractions = [
        {"concepts": [{"name": "DNA", "mentions": 2}]},
        {"concepts": [{"name": "dna", "mentions": 5}]},
    ]
    result = canonicalize_entities(extractions)
    assert len(result["concepts"]) == 1
    assert result["concepts"][0]["name"] == "dna"


def test_canonicalize_keeps_semantically_distinct_concepts() -> None:
    """Fuzzy dedup must NOT merge distinct concepts that share a substring."""
    extractions = [
        {"concepts": [{"name": "Inflation", "mentions": 5}]},
        {"concepts": [{"name": "Consumer Price Inflation", "mentions": 3}]},
        {"concepts": [{"name": "Apollo", "mentions": 2}]},
        {"concepts": [{"name": "Apollo Target", "mentions": 4}]},
    ]
    result = canonicalize_entities(extractions)
    names = {c["name"] for c in result["concepts"]}
    assert names == {"Inflation", "Consumer Price Inflation", "Apollo", "Apollo Target"}
    assert len(result["concepts"]) == 4


def test_extract_many_sorted_by_doc_id(tmp_path: Path) -> None:
    """Results must be returned in stable doc_id order regardless of thread timing."""

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps({"concepts": [], "events": [], "summary": "s"})

    docs = [Document(doc_id=f"doc_{i}", text="...") for i in range(6)]
    # Reverse input order; output must still sort by doc_id.
    results = extract_many(
        list(reversed(docs)),
        output_dir=tmp_path,
        workers=3,
        llm=fake_llm,
    )
    order = [r["_doc_id"] for r in results]
    assert order == sorted(order)


def test_extract_many_content_hash_cache_shared_across_output_dirs(tmp_path: Path) -> None:
    """Two runs into different output_dirs must share the content-hash cache."""
    src = tmp_path / "doc.md"
    src.write_text("hello world")
    doc = Document(doc_id="doc.md", text="hello world", source=str(src))
    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps({"concepts": [], "events": [], "summary": "s"})

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    extract_many([doc], output_dir=out1, cache_root=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 1
    # Second run into a different output dir must hit the content-hash cache.
    extract_many([doc], output_dir=out2, cache_root=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 1


def test_extract_entities_single_call_for_short_doc(tmp_path: Path) -> None:
    """Docs under chunk_chars fit in one call — no chunking overhead."""
    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 2, "note": "n"}],
                "events": [],
                "summary": "s",
            }
        )

    doc = Document(doc_id="d1", text="x" * 500, source="/fake/d1")
    from pengram.extract_llm import extract_entities

    result = extract_entities(doc, llm=fake_llm, chunk_chars=1000)
    assert calls["n"] == 1
    assert result["concepts"][0]["mentions"] == 2


def test_extract_entities_survives_one_bad_chunk(
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Regression for v0.2.0 Bug 2: one chunk returning bad JSON must not
    discard the successful chunks. Only raises when ALL chunks fail."""
    from pengram.extract_llm import extract_entities
    from pengram.llm import LLMError

    call_n = {"i": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        call_n["i"] += 1
        if call_n["i"] == 2:
            raise LLMError("Expecting ',' delimiter: line 31 column 74")
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 1, "note": "n"}],
                "events": [],
                "summary": f"chunk {call_n['i']}",
            }
        )

    doc = Document(doc_id="long.md", text="a" * 3000, source="/fake/long.md")
    result = extract_entities(
        doc,
        llm=fake_llm,
        chunk_chars=1000,
        chunk_overlap=200,
    )
    # Several chunks dispatched; chunk 2 raised, the rest merged. Mentions
    # equals (total_chunks - 1) since every surviving chunk returned
    # mentions=1 for the same concept.
    assert result["concepts"][0]["name"] == "graphs"
    total_calls = call_n["i"]
    assert result["concepts"][0]["mentions"] == total_calls - 1
    out = capsys.readouterr().out
    assert "WARN: chunk 2/" in out and "long.md" in out


def test_extract_entities_raises_when_all_chunks_fail(tmp_path: Path) -> None:
    """When every chunk fails the function raises — caller decides
    whether to skip or retry the whole document."""
    from pengram.extract_llm import extract_entities
    from pengram.llm import LLMError

    def fake_llm(prompt: str, **kw: Any) -> str:
        raise LLMError("boom")

    doc = Document(doc_id="bad.md", text="a" * 3000, source="/fake/bad.md")
    import pytest

    with pytest.raises(LLMError, match="all .* chunk"):
        extract_entities(doc, llm=fake_llm, chunk_chars=1000, chunk_overlap=0)


def test_extract_entities_chunks_oversized_and_merges(tmp_path: Path) -> None:
    """len(text) > chunk_chars ⇒ multiple calls merged into one result."""
    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 1, "note": "chunk"}],
                "events": [],
                "summary": f"chunk {calls['n']}",
            }
        )

    doc = Document(doc_id="d1", text="a" * 3000, source="/fake/d1")
    from pengram.extract_llm import extract_entities

    result = extract_entities(
        doc,
        llm=fake_llm,
        chunk_chars=1000,
        chunk_overlap=200,
    )
    # 3000 chars / step 800 → 4 chunks.
    assert calls["n"] > 1
    # Same concept across all chunks ⇒ mentions summed.
    assert result["concepts"][0]["mentions"] == calls["n"]
    # Summary is the first chunk's summary (cheap merge).
    assert result["summary"] == "chunk 1"


def test_chunk_text_single_chunk_for_small_input() -> None:
    from pengram.extract_llm import _chunk_text

    assert _chunk_text("short", 100, 10) == ["short"]


def test_chunk_text_overlapping_windows() -> None:
    from pengram.extract_llm import _chunk_text

    chunks = _chunk_text("abcdefghij", 4, 1)
    # step = 3, chunks start at 0, 3, 6, 9.
    assert chunks[0] == "abcd"
    assert chunks[1] == "defg"
    # Last chunk absorbs the remainder.
    assert chunks[-1].endswith("j")


def test_extract_many_continues_after_one_failure(
    tmp_path: Path,
    capsys: Any,
) -> None:
    """Regression: one malformed LLM response must not kill the batch."""
    from pengram.llm import LLMError

    call_order = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        call_order["n"] += 1
        # Second document's extraction returns unparseable JSON.
        if call_order["n"] == 2:
            raise LLMError("Expecting ',' delimiter: line 31 column 74")
        return json.dumps({"concepts": [], "events": [], "summary": "ok"})

    docs = [
        Document(doc_id="a", text="..."),
        Document(doc_id="b", text="..."),
        Document(doc_id="c", text="..."),
    ]
    # Sequential path (workers=1) — deterministic call_order.
    results = extract_many(docs, output_dir=tmp_path, workers=1, llm=fake_llm)
    # Two successes survive even though one call raised.
    assert {r["_doc_id"] for r in results} == {"a", "c"}
    out = capsys.readouterr().out
    assert "WARN" in out and "b" in out
    # The doc_id of the failed file is in the warning.
    assert "extraction failed for b" in out


def test_extract_many_parallel_failure_is_isolated(tmp_path: Path) -> None:
    """Same invariant as above, exercised through the ThreadPoolExecutor path."""
    import threading

    from pengram.llm import LLMError

    lock = threading.Lock()
    seen: list[str] = []

    def fake_llm(prompt: str, **kw: Any) -> str:
        # Identify the doc by its content, which the prompt embeds.
        doc_tag = prompt.split("CONTENT:\n", 1)[-1].strip()
        with lock:
            seen.append(doc_tag)
        if doc_tag == "bad":
            raise LLMError("bad delimiter")
        return json.dumps({"concepts": [], "events": [], "summary": "ok"})

    docs = [
        Document(doc_id="a", text="good-a"),
        Document(doc_id="b", text="bad"),
        Document(doc_id="c", text="good-c"),
        Document(doc_id="d", text="good-d"),
    ]
    results = extract_many(docs, output_dir=tmp_path, workers=3, llm=fake_llm)
    assert {r["_doc_id"] for r in results} == {"a", "c", "d"}


def test_extract_many_returns_cached_when_new_extraction_fails(
    tmp_path: Path,
) -> None:
    """Cached results survive even when a never-cached doc raises."""
    from pengram.llm import LLMError

    def fake_llm(prompt: str, **kw: Any) -> str:
        if "new" in prompt:
            raise LLMError("boom")
        return json.dumps({"concepts": [], "events": [], "summary": "ok"})

    docs = [Document(doc_id="a", text="cache-me")]
    # First run caches 'a'.
    extract_many(docs, output_dir=tmp_path, workers=1, llm=fake_llm)
    # Second run adds 'b' (which always fails) alongside 'a'.
    docs2 = [
        Document(doc_id="a", text="cache-me"),
        Document(doc_id="b", text="new content"),
    ]
    results = extract_many(docs2, output_dir=tmp_path, workers=1, llm=fake_llm)
    ids = {r["_doc_id"] for r in results}
    assert "a" in ids
    assert "b" not in ids


def test_extract_many_content_hash_cache_miss_on_edit(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("version one")
    doc1 = Document(doc_id="doc.md", text="version one", source=str(src))
    calls = {"n": 0}

    def fake_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        return json.dumps({"concepts": [], "events": [], "summary": "s"})

    out = tmp_path / "run"
    extract_many([doc1], output_dir=out, cache_root=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 1
    # Edit the file → content hash changes → cache miss → new LLM call.
    src.write_text("version two")
    doc2 = Document(doc_id="doc.md", text="version two", source=str(src))
    extract_many([doc2], output_dir=out, cache_root=tmp_path, workers=1, llm=fake_llm)
    assert calls["n"] == 2


def test_extraction_dump_excludes_internal_underscore_keys(tmp_path: Path) -> None:
    src = tmp_path / "x.md"
    src.write_text("x" * 100)
    doc = Document(doc_id="x.md", text="x" * 100, source=str(src))

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps({"concepts": [], "summary": ""})

    extract_many([doc], output_dir=tmp_path, workers=1, llm=fake_llm)
    dump = tmp_path / "extractions" / "x.md.json"
    assert dump.exists()
    text = dump.read_text()
    assert "_source" not in text
    assert "_doc_id" not in text
    assert str(src) not in text


def test_cache_reload_repopulates_source_and_doc_id(tmp_path: Path) -> None:
    src = tmp_path / "x.md"
    src.write_text("x" * 100)
    doc = Document(doc_id="x.md", text="x" * 100, source=str(src))

    def fake_llm(prompt: str, **kw: Any) -> str:
        return json.dumps({"concepts": [], "summary": ""})

    def fail_llm(prompt: str, **kw: Any) -> str:
        raise AssertionError("LLM should not be called on cache hit")

    extract_many([doc], output_dir=tmp_path / "out1", cache_root=tmp_path, workers=1, llm=fake_llm)
    results = extract_many(
        [doc], output_dir=tmp_path / "out2", cache_root=tmp_path, workers=1, llm=fail_llm
    )
    assert results[0]["_doc_id"] == "x.md"
    assert results[0]["_source"] == str(src)


def test_merge_rejects_short_and_numeric_concept_names() -> None:
    from pengram.extract_llm import _merge_chunk_extractions

    chunks = [
        {
            "concepts": [
                {"name": "AI", "mentions": 5},
                {"name": "42", "mentions": 3},
                {"name": "x", "mentions": 2},
                {"name": "machine learning", "mentions": 4},
            ],
            "summary": "test",
        }
    ]
    merged = _merge_chunk_extractions(chunks)
    names = {c["name"].lower() for c in merged["concepts"]}
    assert "machine learning" in names
    assert "ai" in names
    assert "42" not in names
    assert "x" not in names


# ---------------------------------------------------------------------------
# _coerce_mentions
# ---------------------------------------------------------------------------


def test_coerce_mentions_int_passthrough() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions(5) == 5
    assert _coerce_mentions(1) == 1


def test_coerce_mentions_clamps_zero_and_negative() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions(0) == 1
    assert _coerce_mentions(-3) == 1


def test_coerce_mentions_float_truncated() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions(3.7) == 3
    assert _coerce_mentions(1.0) == 1
    assert _coerce_mentions(0.5) == 1


def test_coerce_mentions_string_numeric() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions("4") == 4
    assert _coerce_mentions("2.9") == 2
    assert _coerce_mentions("0") == 1


def test_coerce_mentions_string_non_numeric() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions("many") == 1
    assert _coerce_mentions("") == 1
    assert _coerce_mentions("N/A") == 1


def test_coerce_mentions_none_and_bool() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions(None) == 1
    assert _coerce_mentions(True) == 1
    assert _coerce_mentions(False) == 1


def test_coerce_mentions_non_scalar() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions({"count": 3}) == 1
    assert _coerce_mentions([1, 2, 3]) == 1
    assert _coerce_mentions(()) == 1


def test_coerce_mentions_inf_and_nan() -> None:
    from pengram.extract_llm import _coerce_mentions

    assert _coerce_mentions(float("inf")) == 1
    assert _coerce_mentions(float("-inf")) == 1
    assert _coerce_mentions(float("nan")) == 1


def test_merge_chunk_extractions_coerces_adversarial_mentions() -> None:
    """String/None/dict mentions in per-chunk results must not crash merge."""
    from pengram.extract_llm import _merge_chunk_extractions

    chunks = [
        {
            "concepts": [
                {"name": "Alpha", "mentions": "3"},
                {"name": "Beta", "mentions": None},
            ],
            "summary": "c1",
        },
        {
            "concepts": [
                {"name": "Alpha", "mentions": {"count": 5}},
                {"name": "Beta", "mentions": -2},
            ],
            "summary": "c2",
        },
    ]
    merged = _merge_chunk_extractions(chunks)
    by_name = {c["name"]: c for c in merged["concepts"]}
    assert by_name["Alpha"]["mentions"] == 4  # 3 + 1 (dict→1)
    assert by_name["Beta"]["mentions"] == 2  # 1 (None→1) + 1 (neg→1)
    assert all(isinstance(c["mentions"], int) for c in merged["concepts"])


def test_canonicalize_coerces_adversarial_mentions() -> None:
    """LLM returning string/None/dict mentions must not crash or corrupt totals."""
    extractions = [
        {
            "concepts": [
                {"name": "Alpha", "mentions": "3"},
                {"name": "Beta", "mentions": None},
                {"name": "Gamma", "mentions": {"count": 5}},
            ]
        },
        {
            "concepts": [
                {"name": "Alpha", "mentions": 2.7},
                {"name": "Beta", "mentions": -1},
            ]
        },
    ]
    result = canonicalize_entities(extractions)
    by_name = {c["name"]: c for c in result["concepts"]}
    assert by_name["Alpha"]["mentions"] == 5  # 3 + 2
    assert by_name["Beta"]["mentions"] == 2  # 1 + 1
    assert by_name["Gamma"]["mentions"] == 1
    assert all(isinstance(c["mentions"], int) for c in result["concepts"])


def test_extract_many_emits_failure_summary_on_errors(tmp_path: Path, capsys: Any) -> None:
    """When some docs fail extraction, a summary line must appear."""
    from pengram.llm import LLMError

    calls = {"n": 0}

    def flaky_llm(prompt: str, **kw: Any) -> str:
        calls["n"] += 1
        if calls["n"] <= 1:
            raise LLMError("transient glitch")
        return json.dumps({"concepts": [{"name": "Alpha", "mentions": 1}], "summary": "ok"})

    docs = [
        Document(doc_id="fail", text="will fail " * 30, source="/fake/fail.md"),
        Document(doc_id="ok", text="will succeed " * 30, source="/fake/ok.md"),
    ]
    results = extract_many(docs, output_dir=tmp_path, workers=1, llm=flaky_llm)
    assert len(results) == 1
    out = capsys.readouterr().out
    assert "1/2 succeeded" in out
    assert "1 failed" in out
    assert "LLMError" in out
