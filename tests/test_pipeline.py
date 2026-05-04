# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from pengram.pipeline import RunConfig


def test_run_config_uses_provider_specific_defaults_on_override() -> None:
    with mock.patch("pengram.config.LLM_PROVIDER", "claude-cli"):
        rc = RunConfig.from_args(llm_provider="openai")
    assert rc.provider == "openai"
    assert rc.synth_model == "gpt-4o-mini"
    assert rc.extract_model == "gpt-4o-mini"
    assert rc.link_model == "gpt-4o-mini"


def test_run_config_keeps_config_defaults_without_override() -> None:
    with mock.patch("pengram.config.LLM_PROVIDER", "claude-cli"):
        with mock.patch(
            "pengram.config.LLM",
            {
                "extract_model": "haiku",
                "link_model": "haiku",
                "synth_model": "sonnet",
                "extract_timeout": 300,
                "link_timeout": 300,
                "synth_timeout": 300,
            },
        ):
            rc = RunConfig.from_args()
    assert rc.provider == "claude-cli"
    assert rc.synth_model == "sonnet"


def test_run_config_llm_model_override_applies_to_extract_and_link() -> None:
    rc = RunConfig.from_args(llm_model="custom-model")
    assert rc.extract_model == "custom-model"
    assert rc.link_model == "custom-model"


def test_ambiguous_edges_dropped() -> None:
    from pengram.pipeline import build_semantic_extraction
    from pengram.vocabulary import CONFIDENCE_AMBIGUOUS, CONFIDENCE_INFERRED

    fake_llm_results = [
        {
            "_doc_id": "doc1.txt",
            "_source": "doc1.txt",
            "concepts": [
                {"name": "Alpha Concept", "mentions": 3},
                {"name": "Beta Concept", "mentions": 3},
            ],
            "summary": "Test document.",
        },
    ]

    fake_decisions = [
        mock.MagicMock(
            source="concept_alpha_concept",
            target="concept_beta_concept",
            relation="references",
            confidence=CONFIDENCE_AMBIGUOUS,
            reason="unclear",
        ),
        mock.MagicMock(
            source="concept_alpha_concept",
            target="concept_beta_concept",
            relation="supports",
            confidence=CONFIDENCE_INFERRED,
            reason="clear",
        ),
    ]

    from pathlib import Path
    from unittest.mock import patch

    from pengram.link import LinkStats

    with patch("pengram.pipeline.link_all", return_value=(fake_decisions, LinkStats())):
        result = build_semantic_extraction(
            fake_llm_results,
            output_dir=Path("/tmp/test"),
            run_linker=True,
            provider="claude-cli",
        )

    link_edges = [e for e in result["edges"] if e.get("confidence") != "EXTRACTED"]
    for edge in link_edges:
        assert edge["confidence"] != CONFIDENCE_AMBIGUOUS
    assert any(e["relation"] == "supports" for e in link_edges)


def test_youtube_metadata_propagated_to_document_nodes() -> None:
    from pathlib import Path

    from pengram.extract_llm import Document
    from pengram.pipeline import build_semantic_extraction

    doc = Document(
        doc_id="transcripts/abc123.transcript",
        text="Some transcript text about music.",
        source="transcripts/abc123.transcript",
        metadata={
            "file_type": "transcript",
            "title": "Femi Kuti on Music",
            "video_id": "abc123",
            "channel": "test_channel",
            "upload_date": "20240315",
            "views": 12345,
            "likes": 500,
            "duration": 1200,
            "tab": "videos",
        },
    )
    llm_results = [
        {
            "_doc_id": "transcripts/abc123.transcript",
            "_source": "transcripts/abc123.transcript",
            "concepts": [{"name": "Music Theory", "mentions": 3}],
            "summary": "Discussion about music.",
        },
    ]

    result = build_semantic_extraction(
        llm_results,
        output_dir=Path("/tmp/test"),
        run_linker=False,
        documents=[doc],
    )

    doc_nodes = [n for n in result["nodes"] if n["id"].startswith("doc_")]
    assert len(doc_nodes) == 1
    node = doc_nodes[0]
    assert node["label"] == "Femi Kuti on Music"
    assert node["video_id"] == "abc123"
    assert node["channel"] == "test_channel"
    assert node["upload_date"] == "20240315"
    assert node["views"] == 12345
    assert node["view_count"] == 12345
    assert node["likes"] == 500
    assert node["like_count"] == 500
    assert node["duration"] == 1200


def test_non_youtube_document_has_no_yt_fields() -> None:
    from pathlib import Path

    from pengram.extract_llm import Document
    from pengram.pipeline import build_semantic_extraction

    doc = Document(
        doc_id="readme.md",
        text="A normal document.",
        source="readme.md",
        metadata={"file_type": "document"},
    )
    llm_results = [
        {
            "_doc_id": "readme.md",
            "_source": "readme.md",
            "concepts": [{"name": "Documentation", "mentions": 2}],
            "summary": "A readme.",
        },
    ]
    result = build_semantic_extraction(
        llm_results,
        output_dir=Path("/tmp/test"),
        run_linker=False,
        documents=[doc],
    )
    doc_nodes = [n for n in result["nodes"] if n["id"].startswith("doc_")]
    assert len(doc_nodes) == 1
    assert "video_id" not in doc_nodes[0]
    assert "view_count" not in doc_nodes[0]


def test_read_document_text_strips_frontmatter(tmp_path: Path) -> None:
    from pengram.cli import _read_document_text
    from pengram.detect import FileType

    transcript = tmp_path / "abc123.transcript"
    transcript.write_text('---\nvideo_id: abc123\ntitle: "Test"\n---\n\nactual content here\n')
    text = _read_document_text(transcript, FileType.TRANSCRIPT)
    assert "video_id" not in text
    assert "actual content here" in text


def test_read_document_text_no_frontmatter_unchanged(tmp_path: Path) -> None:
    from pengram.cli import _read_document_text
    from pengram.detect import FileType

    doc = tmp_path / "readme.md"
    doc.write_text("# Hello World\n\nSome content.\n")
    text = _read_document_text(doc, FileType.DOCUMENT)
    assert text == "# Hello World\n\nSome content.\n"


def test_parse_transcript_meta(tmp_path: Path) -> None:
    from pengram.cli import _parse_transcript_meta

    transcript = tmp_path / "vid001.transcript"
    transcript.write_text(
        '---\nvideo_id: vid001\ntitle: "Great"\nviews: 100\nlikes: NA\n---\n\ncontent\n'
    )
    meta = _parse_transcript_meta(transcript)
    assert meta["video_id"] == "vid001"
    assert meta["title"] == "Great"
    assert meta["views"] == 100
    assert "likes" not in meta  # NA values stripped


def test_fuzzy_variants_resolve_to_single_node() -> None:
    """Hyphen, plural, and case variants from raw results must resolve
    to the single canonical node created by canonicalize_entities,
    not spawn orphan nodes.
    """
    from pengram.pipeline import build_semantic_extraction

    fake_llm_results = [
        {
            "_doc_id": "doc1.txt",
            "_source": "doc1.txt",
            "concepts": [
                {"name": "Apollo system", "mentions": 5},
                {"name": "Apollo-system", "mentions": 3},
            ],
            "summary": "Doc one.",
        },
        {
            "_doc_id": "doc2.txt",
            "_source": "doc2.txt",
            "concepts": [
                {"name": "apollo systems", "mentions": 2},
            ],
            "summary": "Doc two.",
        },
    ]

    result = build_semantic_extraction(
        fake_llm_results,
        output_dir=Path("/tmp/test"),
        run_linker=False,
    )
    concept_nodes = [n for n in result["nodes"] if n.get("kind") == "concept"]
    assert len(concept_nodes) == 1
    assert concept_nodes[0]["mentions"] == 10


def test_fuzzy_variants_produce_correct_edges() -> None:
    """Each raw variant should produce a doc→entity edge to the canonical
    node, not a missing node.
    """
    from pengram.pipeline import build_semantic_extraction

    fake_llm_results = [
        {
            "_doc_id": "doc1.txt",
            "_source": "doc1.txt",
            "concepts": [
                {"name": "Machine Learning", "mentions": 4},
                {"name": "machine-learning", "mentions": 2},
            ],
            "summary": "Doc one.",
        },
    ]

    result = build_semantic_extraction(
        fake_llm_results,
        output_dir=Path("/tmp/test"),
        run_linker=False,
    )
    concept_nodes = [n for n in result["nodes"] if n.get("kind") == "concept"]
    assert len(concept_nodes) == 1
    ref_edges = [
        e
        for e in result["edges"]
        if e["relation"] == "references" and e["target"] == concept_nodes[0]["id"]
    ]
    assert len(ref_edges) == 2


def test_adversarial_mentions_in_pipeline() -> None:
    """String/None/dict mentions from LLM must not crash or corrupt the
    pipeline node and edge assembly."""
    from pengram.pipeline import build_semantic_extraction

    fake_llm_results = [
        {
            "_doc_id": "doc1.txt",
            "_source": "doc1.txt",
            "concepts": [
                {"name": "Alpha Concept", "mentions": "5"},
                {"name": "Beta Concept", "mentions": None},
                {"name": "Gamma Concept", "mentions": {"count": 3}},
            ],
            "summary": "Test document.",
        },
    ]

    result = build_semantic_extraction(
        fake_llm_results,
        output_dir=Path("/tmp/test"),
        run_linker=False,
    )
    concept_nodes = {n["label"]: n for n in result["nodes"] if n.get("kind") == "concept"}
    assert concept_nodes["Alpha Concept"]["mentions"] == 5
    assert concept_nodes["Beta Concept"]["mentions"] == 1
    assert concept_nodes["Gamma Concept"]["mentions"] == 1
    assert all(isinstance(n["mentions"], int) for n in concept_nodes.values())


def test_build_semantic_extraction_idempotent_on_same_input() -> None:
    """Two calls with identical LLM results must produce identical output."""
    from pengram.pipeline import build_semantic_extraction

    fake_llm_results = [
        {
            "_doc_id": "doc1.txt",
            "_source": "doc1.txt",
            "concepts": [
                {"name": "Alpha Concept", "mentions": 5},
                {"name": "Beta Concept", "mentions": 3},
            ],
            "summary": "First document.",
        },
        {
            "_doc_id": "doc2.txt",
            "_source": "doc2.txt",
            "concepts": [
                {"name": "Alpha Concept", "mentions": 2},
                {"name": "Gamma Concept", "mentions": 1},
            ],
            "summary": "Second document.",
        },
    ]

    r1 = build_semantic_extraction(
        fake_llm_results,
        output_dir=Path("/tmp/test-idem"),
        run_linker=False,
    )
    r2 = build_semantic_extraction(
        fake_llm_results,
        output_dir=Path("/tmp/test-idem"),
        run_linker=False,
    )
    assert r1["nodes"] == r2["nodes"]
    assert r1["edges"] == r2["edges"]
