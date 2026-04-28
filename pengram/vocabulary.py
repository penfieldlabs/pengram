# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Relationship type vocabulary for PENgram.

Single source of truth for all relationship types produced by the pipeline.

- ``SEMANTIC_TYPES`` — the 24 Penfield relationship types used by LLM-based
  linking. Defined in the obsidian-wikilink-types skill.
- ``STRUCTURAL_TYPES`` — deterministic code/AST relationships produced by
  tree-sitter extraction.
- ``ALL_TYPES`` — the union of both.
- ``CONFIDENCE_*`` — the three confidence labels attached to every edge.
- ``DEFAULTS`` — fallback relationship type per entity kind, used when the
  LLM cannot confidently assign a type.
"""

from __future__ import annotations

from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Semantic relationship types (24) — the Penfield vocabulary
# ---------------------------------------------------------------------------

KNOWLEDGE_EVOLUTION: frozenset[str] = frozenset(
    {
        "supersedes",
        "updates",
        "evolution_of",
    }
)

EVIDENCE: frozenset[str] = frozenset(
    {
        "supports",
        "contradicts",
        "disputes",
    }
)

HIERARCHY: frozenset[str] = frozenset(
    {
        "parent_of",
        "child_of",
        "sibling_of",
        "composed_of",
        "part_of",
    }
)

CAUSATION: frozenset[str] = frozenset(
    {
        "causes",
        "influenced_by",
        "prerequisite_for",
    }
)

IMPLEMENTATION: frozenset[str] = frozenset(
    {
        "implements",
        "documents",
        "tests",
        "example_of",
    }
)

CONVERSATION: frozenset[str] = frozenset(
    {
        "responds_to",
        "references",
        "inspired_by",
    }
)

SEQUENCE: frozenset[str] = frozenset(
    {
        "follows",
        "precedes",
    }
)

DEPENDENCIES: frozenset[str] = frozenset(
    {
        "depends_on",
    }
)

SEMANTIC_TYPES: frozenset[str] = (
    KNOWLEDGE_EVOLUTION
    | EVIDENCE
    | HIERARCHY
    | CAUSATION
    | IMPLEMENTATION
    | CONVERSATION
    | SEQUENCE
    | DEPENDENCIES
)

SEMANTIC_CATEGORIES: Mapping[str, frozenset[str]] = {
    "knowledge_evolution": KNOWLEDGE_EVOLUTION,
    "evidence": EVIDENCE,
    "hierarchy": HIERARCHY,
    "causation": CAUSATION,
    "implementation": IMPLEMENTATION,
    "conversation": CONVERSATION,
    "sequence": SEQUENCE,
    "dependencies": DEPENDENCIES,
}

# ---------------------------------------------------------------------------
# Structural relationship types — code/AST extraction
# ---------------------------------------------------------------------------

STRUCTURAL_TYPES: frozenset[str] = frozenset(
    {
        "calls",
        "imports",
        "uses",
        "extends",
        "implements_interface",
        "instantiates",
        "overrides",
        "decorates",
    }
)

ALL_TYPES: frozenset[str] = SEMANTIC_TYPES | STRUCTURAL_TYPES

# ---------------------------------------------------------------------------
# Confidence labels
# ---------------------------------------------------------------------------

CONFIDENCE_EXTRACTED = "EXTRACTED"
CONFIDENCE_INFERRED = "INFERRED"
CONFIDENCE_AMBIGUOUS = "AMBIGUOUS"

CONFIDENCE_LABELS: frozenset[str] = frozenset(
    {
        CONFIDENCE_EXTRACTED,
        CONFIDENCE_INFERRED,
        CONFIDENCE_AMBIGUOUS,
    }
)

# ---------------------------------------------------------------------------
# Default relationship types per entity kind
# ---------------------------------------------------------------------------

DEFAULTS: Mapping[str, str] = {
    "category": "part_of",
    "concept": "references",
    "document": "references",
    "code": "references",
    "transcript": "references",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_semantic(relation: str) -> bool:
    """Return True if ``relation`` is one of the 24 semantic types."""
    return relation in SEMANTIC_TYPES


def is_valid_structural(relation: str) -> bool:
    """Return True if ``relation`` is a known structural (AST) type."""
    return relation in STRUCTURAL_TYPES


def is_valid(relation: str) -> bool:
    """Return True if ``relation`` is any known relationship type."""
    return relation in ALL_TYPES


def is_valid_confidence(label: str) -> bool:
    """Return True if ``label`` is a valid confidence label."""
    return label in CONFIDENCE_LABELS


__all__ = [
    "SEMANTIC_TYPES",
    "STRUCTURAL_TYPES",
    "ALL_TYPES",
    "SEMANTIC_CATEGORIES",
    "KNOWLEDGE_EVOLUTION",
    "EVIDENCE",
    "HIERARCHY",
    "CAUSATION",
    "IMPLEMENTATION",
    "CONVERSATION",
    "SEQUENCE",
    "DEPENDENCIES",
    "CONFIDENCE_EXTRACTED",
    "CONFIDENCE_INFERRED",
    "CONFIDENCE_AMBIGUOUS",
    "CONFIDENCE_LABELS",
    "DEFAULTS",
    "is_valid_semantic",
    "is_valid_structural",
    "is_valid",
    "is_valid_confidence",
]
