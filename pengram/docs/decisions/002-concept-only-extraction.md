# ADR-002: Concept-only extraction (events killed)

**Status:** Accepted  
**Date:** 2026-04-25  
**Context:** `extract_llm.py`, entity taxonomy

## Decision

The extraction prompt produces a single entity kind: **concepts**. The earlier
`events`, `people`, `organizations`, `predictions`, and `topics` kinds were
removed.

## Rationale

- **People and organizations** appear naturally in concept notes and document
  summaries. Extracting them as first-class nodes produced high-cardinality,
  low-signal noise (e.g. "John" appearing once in a transcript).
- **Events** overlapped with concepts — "the 2024 election" is both. Merging
  them into concepts eliminated a fuzzy boundary the LLM struggled with.
- **Predictions and topics** were downstream concerns: predictions are concepts
  with a temporal qualifier, and topics are what clustering produces
  deterministically from co-occurrence.

Legacy cached extractions that still contain an `events` key are folded into
`concepts` on read via `_fold_legacy_events`, so the cache is not invalidated.

## Alternatives considered

- **Keep all kinds, filter downstream:** increases prompt complexity, LLM cost,
  and dedup surface for no additional graph quality.
- **Make kinds configurable:** premature — no user has asked for it, and the
  prompt would need per-kind tuning.
