# ADR-009: No explicit `type:` in Penfield vault frontmatter

**Status:** Accepted
**Date:** 2026-04-27
**Context:** `export_common.py`, `export_penfield.py`, [penfield-import](https://github.com/penfieldlabs/penfield-import) compatibility

## Decision

PENgram does **not** emit a `type:` field in vault note frontmatter. Penfield's
API auto-classifies memories by content when `memory_type` is omitted from the
POST payload.

`note_type:` remains — it drives PENgram's internal routing (subdirectory
selection, body rendering, threshold logic) and is ignored by [penfield-import](https://github.com/penfieldlabs/penfield-import).

## Background

[penfield-import](https://github.com/penfieldlabs/penfield-import) reads `type:` from frontmatter and maps it to the `memory_type`
API field. When `type:` is absent, `memory_type` is omitted from the request
and Penfield's server auto-classifies the memory based on content analysis.

Per Penfield's SKILL documentation:

> "Memory type is auto-detected from content — you don't pass it as a
> parameter. Write descriptively so the system classifies correctly."

## Rationale

- **Auto-classification has more signal.** Penfield's classifier sees the full
  note content — definitions, quotes, source text, relationships. A static
  mapping table in PENgram only sees `note_type` (concept, document, category,
  etc.) and would map almost everything to `reference`, losing the nuance
  Penfield can detect (e.g., distinguishing `fact` from `insight` within
  concepts).

- **No collision risk.** The original v0.1.0 codebase reserved `type:` to
  avoid a hypothetical conflict with the relationship vocabulary. In practice,
  `type` is not one of the 24 semantic relationship types and never was. The
  reservation had zero practical value — but since we're also not emitting it,
  the point is moot.

- **Simpler contract.** PENgram's job is to produce well-structured vault notes
  with accurate content, typed relationships, and clean YAML. Penfield's job is
  to classify those memories. Each system does what it's best at.

## Monitoring

If users report systematic misclassification after import (e.g., concept notes
landing as `conversation` instead of `fact`), we can add an opt-in
`--penfield-types` flag that emits an explicit `type:` field using a static
mapping:

| PENgram `note_type` | Penfield `type` |
|---|---|
| concept | fact |
| document, transcript, video_transcript, audio_transcript | reference |
| category, code | reference |

This mapping is documented here for future reference but is **not implemented
in v0.1.x**.

## Alternatives considered

- **Always emit `type:`:** Overrides Penfield's auto-classifier with a coarse
  mapping. Risks forcing incorrect types on edge cases the classifier would
  have gotten right.
- **Emit `type:` only for concepts:** Inconsistent — some notes would have it,
  others wouldn't. Confusing for users inspecting the vault.
