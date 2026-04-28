# ADR-010: Entity deduplication boundaries

**Status:** Accepted
**Date:** 2026-04-27
**Context:** `enrich.py`, `extract_llm.py`, enrichment pipeline

## Decision

PENgram performs entity deduplication at two levels, each with a defined
scope. Semantic equivalence beyond surface-form is delegated to the LLM
and is **not** guaranteed.

### Level 1 — Deterministic dedup (`_deterministic_dedup`)

Runs after LLM enrichment. Merges concept pairs whose labels reduce to the
same key after normalisation:

1. Lowercase
2. Strip hyphens and whitespace
3. Drop trailing `s` (unless the stem ends in `ss`)

This catches plurals (`enzyme` / `enzymes`), hyphenation variants
(`wild-type` / `wild type`), and trivial casing differences. It does
**not** catch semantic equivalents (`lactobacillus` vs
`lactobacillus-bacteria`) because those normalise to different keys.

### Level 2 — LLM enrichment merge (`merge_into`)

The enrichment prompt presents cluster rosters to the LLM and instructs it
to set `merge_into` when two names refer to the same entity. This is where
semantic dedup happens — the LLM has definitions, context, and co-occurrence
to judge equivalence.

## Known limitation

The LLM (particularly smaller models like gpt-4o-mini) can miss semantic
duplicates that a human would catch. Observed example: `lactobacillus` and
`lactobacillus-bacteria` survived as separate concepts because the model
judged them as distinct despite near-identical definitions.

## Mitigation options (not implemented in v0.1.x)

If users report persistent dedup misses:

- **Prompt tuning:** Add explicit examples of redundant suffixes to the
  enrichment prompt (X vs X bacteria, X vs X cells, X vs X disease).
- **Stronger model:** Use gpt-4o instead of gpt-4o-mini for the enrichment
  pass — better semantic judgment, ~10x cost.
- **Embedding-similarity pass:** Add an opt-in `--enrich-dedup-threshold 0.95`
  flag that catches near-synonyms via cosine similarity on embeddings. Real
  engineering, not a prompt tweak.

## Rationale

Deterministic dedup is fast, predictable, and catches the majority of
duplicates. Pushing semantic equivalence into the deterministic layer would
require maintaining a synonym dictionary — brittle, domain-specific, and
redundant with what the LLM already does. The current split keeps each layer
doing what it's best at.
