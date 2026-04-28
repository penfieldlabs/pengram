# ADR-001: Last-write-wins for duplicate edges

**Status:** Accepted  
**Date:** 2026-04-25  
**Context:** `build.py`, edge deduplication

## Decision

When two extractions produce edges for the same `(source, target)` pair with
different relation types, the later write wins. A warning is emitted so the
data loss is visible.

## Rationale

The spec calls for single-edge semantics on a DiGraph. A MultiDiGraph would
preserve both relations but complicates every downstream consumer (export,
cluster, analyze) that assumes at most one edge per pair.

The "later wins" rule is predictable: semantic extractions merge on top of
AST extractions, so LLM-typed relations override structural defaults. This
matches the pipeline's build order (AST first, LLM second) and the principle
that richer information supersedes mechanical.

## Alternatives considered

- **MultiDiGraph:** preserves both, but doubles the surface area for every
  consumer and the spec doesn't require it.
- **First-write-wins:** would lock in the structural relation and discard the
  semantic one — worse in the common case.
- **Fail on conflict:** too strict for a pipeline that merges heterogeneous
  extraction passes.
