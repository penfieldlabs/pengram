# ADR-003: Content-hash cache is authoritative

**Status:** Accepted  
**Date:** 2026-04-25  
**Context:** `extract_llm.py`, `cache.py`, two-layer caching

## Decision

The SHA256 content-hash cache (`.pengram-cache/<hash>.json`) is authoritative.
On a cache miss, the pipeline re-extracts even if a per-output dump
(`output_dir/extractions/<doc_id>.json`) exists. The per-output dump is a
secondary, human-readable output only.

## Rationale

Idempotency requires that the same input always produces the same output. If
the user edits a source file, the content hash changes and the extraction
re-runs — correct. If we fell back to the per-output dump on a content-hash
miss, we'd serve stale results from a previous run, breaking "edit the file,
rerun, see new results."

The per-output dump still serves one purpose: crash recovery when no
content-hash cache is available (e.g. the cache root is on a different
filesystem, or the user is processing stdin). In that case, and only that case,
the dump is consulted.

## Alternatives considered

- **Single cache layer (content-hash only):** loses crash recovery for the
  no-cache-root case.
- **Per-output dump as primary:** breaks idempotency after file edits.
- **Timestamp-based invalidation:** fragile across filesystems and CI
  environments where mtime is unreliable.
