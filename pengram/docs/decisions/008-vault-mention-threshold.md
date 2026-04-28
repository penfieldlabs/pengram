# ADR-008: Vault thresholds

**Status:** Accepted
**Date:** 2026-04-26 (updated 2026-04-27)
**Context:** `export_common.py`, `export_penfield.py`, `export_obsidian.py`

## Decision

Two thresholds control which nodes get vault notes:

1. **Concept threshold** (`--threshold-concept N`, default 3): concepts with
   fewer than N mentions are excluded. Minimum 1.
2. **Category threshold** (`--threshold-category N`, default: no filter):
   categories with fewer than N members are excluded. Member count is the
   number of outgoing `parent_of` edges. Minimum 1.

Both thresholds also exclude edges — if either endpoint is below threshold,
the relationship does not appear in any vault note's YAML frontmatter.

The full unfiltered graph is always available in `graph.json`. The thresholds
use different units because they measure different things: concepts are
weighted by how often they recur across sources (mentions), categories by
how many concepts they group (member count).

## Rationale

- A corpus of 257 documents produced 1,831 concepts. 1,074 of those (59%)
  had fewer than 3 mentions — typically one-off terms the LLM extracted
  from a single document. Including them all would flood the vault with
  low-signal notes that have no cross-document connections.
- The threshold preserves 757 concepts and 6,136 edges — the subgraph of
  ideas that actually recur across sources. This is what makes a knowledge
  graph useful: repeated co-occurrence, not exhaustive inventory.
- 3,888 edges (39%) connect to below-threshold concepts. These edges are
  not lost — they remain in `graph.json` and `graph.html` for anyone who
  needs the full picture.
- Both export formats (`export_penfield` and `export_obsidian`) share the
  same `included_nodes()` filter from `export_common`, so the behavior is
  identical and consistent.

## Alternatives considered

- **No threshold (include everything):** produces a vault with 1,000+
  single-mention concept notes that add noise without insight. The lowest
  allowed value is `--threshold-concept 1` — a threshold of 0 is rejected
  because a concept with zero mentions does not exist in the corpus.
- **Higher threshold (5+):** too aggressive for small corpora where 3
  mentions is already significant. The current default balances signal
  for both small (50-doc) and large (300-doc) corpora.
- **Category threshold default at 1 vs no filter:** the default is no
  filter (all categories emitted) rather than an explicit minimum,
  because singleton categories are legitimate when a community has one
  dominant concept. Users who find them noisy can pass
  `--threshold-category 2`.
- **Per-export thresholds:** unnecessary complexity — if you want the
  Penfield vault curated and the Obsidian vault complete, export twice
  with different flags.
