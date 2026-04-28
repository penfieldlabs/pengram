# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.vocabulary."""

from __future__ import annotations

from pengram import vocabulary as v

# ---------------------------------------------------------------------------
# Semantic types
# ---------------------------------------------------------------------------

EXPECTED_SEMANTIC_TYPES = {
    # Knowledge evolution
    "supersedes",
    "updates",
    "evolution_of",
    # Evidence
    "supports",
    "contradicts",
    "disputes",
    # Hierarchy
    "parent_of",
    "child_of",
    "sibling_of",
    "composed_of",
    "part_of",
    # Causation
    "causes",
    "influenced_by",
    "prerequisite_for",
    # Implementation
    "implements",
    "documents",
    "tests",
    "example_of",
    # Conversation
    "responds_to",
    "references",
    "inspired_by",
    # Sequence
    "follows",
    "precedes",
    # Dependencies
    "depends_on",
}


def test_exactly_24_semantic_types() -> None:
    assert len(v.SEMANTIC_TYPES) == 24


def test_all_expected_semantic_types_present() -> None:
    assert set(v.SEMANTIC_TYPES) == EXPECTED_SEMANTIC_TYPES


def test_semantic_categories_cover_all_types() -> None:
    covered: set[str] = set()
    for types in v.SEMANTIC_CATEGORIES.values():
        covered |= set(types)
    assert covered == set(v.SEMANTIC_TYPES)


def test_semantic_categories_are_disjoint() -> None:
    seen: set[str] = set()
    for category, types in v.SEMANTIC_CATEGORIES.items():
        overlap = seen & set(types)
        assert not overlap, f"category {category} overlaps with previous: {overlap}"
        seen |= set(types)


# ---------------------------------------------------------------------------
# Structural types
# ---------------------------------------------------------------------------

EXPECTED_STRUCTURAL_TYPES = {
    "calls",
    "imports",
    "uses",
    "extends",
    "implements_interface",
    "instantiates",
    "overrides",
    "decorates",
}


def test_exactly_8_structural_types() -> None:
    assert len(v.STRUCTURAL_TYPES) == 8


def test_all_expected_structural_types_present() -> None:
    assert set(v.STRUCTURAL_TYPES) == EXPECTED_STRUCTURAL_TYPES


# ---------------------------------------------------------------------------
# Union and disjointness
# ---------------------------------------------------------------------------


def test_semantic_and_structural_disjoint() -> None:
    assert not (v.SEMANTIC_TYPES & v.STRUCTURAL_TYPES)


def test_all_types_is_union() -> None:
    assert v.ALL_TYPES == v.SEMANTIC_TYPES | v.STRUCTURAL_TYPES
    assert len(v.ALL_TYPES) == 32


# ---------------------------------------------------------------------------
# Confidence labels
# ---------------------------------------------------------------------------


def test_confidence_constants() -> None:
    assert v.CONFIDENCE_EXTRACTED == "EXTRACTED"
    assert v.CONFIDENCE_INFERRED == "INFERRED"
    assert v.CONFIDENCE_AMBIGUOUS == "AMBIGUOUS"


def test_confidence_labels_set() -> None:
    assert v.CONFIDENCE_LABELS == {"EXTRACTED", "INFERRED", "AMBIGUOUS"}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_all_valid_types() -> None:
    for kind, relation in v.DEFAULTS.items():
        assert v.is_valid(relation), f"default for {kind} is invalid: {relation}"


def test_defaults_cover_expected_kinds() -> None:
    # Only the kinds the pipeline actually produces. video/paper/image
    # were pruned in v0.2.1 — PDFs and ePubs land as ``document`` and
    # audio/video files land as ``transcript``.
    expected_kinds = {
        "category",
        "concept",
        "document",
        "code",
        "transcript",
    }
    assert expected_kinds.issubset(v.DEFAULTS.keys())


def test_defaults_exclude_unused_kinds() -> None:
    # person / organization / prediction were removed by v0.1.1's
    # entity-purge directive; event was killed in v0.2.0 (the LLM
    # couldn't reliably separate it from concept); video / paper /
    # image were pruned in v0.2.1 (no pipeline produces them — PDFs
    # are plain documents, A/V becomes transcripts). Pin all of them
    # as absent so they don't creep back in.
    for kind in (
        "person",
        "organization",
        "prediction",
        "event",
        "video",
        "paper",
        "image",
    ):
        assert kind not in v.DEFAULTS


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def test_is_valid_semantic_happy_path() -> None:
    assert v.is_valid_semantic("supports")
    assert v.is_valid_semantic("depends_on")


def test_is_valid_semantic_rejects_structural() -> None:
    assert not v.is_valid_semantic("calls")


def test_is_valid_semantic_rejects_unknown() -> None:
    assert not v.is_valid_semantic("totally_made_up")
    assert not v.is_valid_semantic("")


def test_is_valid_structural_happy_path() -> None:
    assert v.is_valid_structural("calls")
    assert v.is_valid_structural("decorates")


def test_is_valid_structural_rejects_semantic() -> None:
    assert not v.is_valid_structural("supports")


def test_is_valid_structural_rejects_unknown() -> None:
    assert not v.is_valid_structural("nope")


def test_is_valid_accepts_both() -> None:
    assert v.is_valid("supports")
    assert v.is_valid("calls")


def test_is_valid_rejects_unknown() -> None:
    assert not v.is_valid("not_a_real_type")


def test_is_valid_confidence_happy_path() -> None:
    assert v.is_valid_confidence("EXTRACTED")
    assert v.is_valid_confidence("INFERRED")
    assert v.is_valid_confidence("AMBIGUOUS")


def test_is_valid_confidence_rejects_lowercase() -> None:
    assert not v.is_valid_confidence("extracted")


def test_is_valid_confidence_rejects_unknown() -> None:
    assert not v.is_valid_confidence("MAYBE")
    assert not v.is_valid_confidence("")
