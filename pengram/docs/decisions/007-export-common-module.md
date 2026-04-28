# ADR-007: Extract shared export logic into export_common

## Status
Accepted

## Context
`export_obsidian.py` imported 13 items from `export_penfield.py`,
9 of which were underscore-prefixed private functions. This coupling
meant any refactoring of penfield internals required coordinated
changes in obsidian. `export_catalog.py` independently redefined
`_CONTENT_KINDS` — a DRY violation that would silently diverge.

The metadata-extraction and body-rendering code blocks were
duplicated verbatim across both vault exporters (~30 lines each).

## Decision
Create `pengram/export_common.py` containing:

- Shared constants (`_CONTENT_KINDS`, `_METADATA_KEYS`, etc.)
- `NoteSpec` dataclass
- Frontmatter rendering (`format_frontmatter`, `_yaml_scalar`, etc.)
- Graph-to-vault helpers (`build_slug_map`, `included_nodes`,
  `_group_edges_by_relation`, `_compute_tags`, `note_type_of`)
- Body rendering (`render_body`, `_render_concept_body`,
  `_render_category_body`)
- Metadata extraction (`extract_metadata`) — previously duplicated

Both `export_penfield.py` and `export_obsidian.py` import from
`export_common.py`. `export_catalog.py` imports `_CONTENT_KINDS`
from the same source. No more cross-exporter private imports.

## Consequences
- Adding a third export format only requires importing from
  `export_common`, not reaching into penfield internals.
- The previously-private functions are now the module's public API
  (some still underscore-prefixed for internal-only helpers).
- `export_penfield.py` re-exports key names (`NoteSpec`,
  `format_frontmatter`, `build_slug_map`, etc.) so existing
  importers don't break.
