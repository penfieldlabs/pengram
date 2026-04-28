# PENgram Architecture

A technical guide for contributors.

## Pipeline

```
+---------+     +----------+     +------+     +----------+     +---------+     +--------+     +--------+
| detect  | --> | extract  | --> | llm  | --> |   link   | --> |  build  | --> | cluster| --> |analyze |
+---------+     +----------+     +------+     +----------+     +---------+     +--------+     +--------+
                      |                                              |                            |
                      v                                              v                            v
                 .pengram-cache                                 validate()                  report() / export_*()
```

The stages are:

1. `detect.collect_files(root)` — walk the directory, classify every file
   as `CODE`, `DOCUMENT`, `IMAGE`, `VIDEO`, `AUDIO`, or `TRANSCRIPT`. PDFs
   and ePubs are `DOCUMENT` (format is metadata, not a separate kind).
   Sensitive files (credentials, `.env`, private keys) are filtered
   silently. `.pengramignore` patterns further narrow the set.
2. `extract_ast.extract_code(path)` — tree-sitter-based deterministic
   extraction for source code. Every edge uses one of the 8 structural
   relationship types and is tagged `confidence="EXTRACTED"`.
3. `extract_llm.extract_many(...)` — LLM-powered entity extraction for
   non-code content. Writes per-document JSON to `extractions/<id>.json`
   as each doc completes (crash-safe).
4. `link.link_all(...)` — LLM-powered relationship typing. Picks one of
   the 24 Penfield semantic types per edge, falls back to
   `DEFAULTS[kind]` with `AMBIGUOUS` confidence when the LLM fails.
5. `build.build(extractions)` — merge every extraction into a single
   `networkx.DiGraph`, normalising ids and dropping dangling edges.
6. `cluster.cluster(g)` — Leiden (preferred) or Louvain (fallback)
   community detection. Oversized communities (>25% of the graph) are
   recursively re-clustered.
7. `analyze.analyze(g, communities)` — god nodes, bridge nodes, surprising
   cross-community edges, and generated questions.
8. `report.render_report(...)` — `GRAPH_REPORT.md` summary.
9. `export_json / export_html / export_penfield / export_obsidian` — the
   four output formats. `graph.json`, `graph.html`, and `GRAPH_REPORT.md`
   are always produced; vault exports are gated by `OUTPUT_TARGET`.

## Module responsibilities

| Module | Function | Input → Output |
|---|---|---|
| `vocabulary` | Single source of truth for relationship types | — → constants |
| `config` | Project configuration + validation | env → constants / `validate()` |
| `detect` | File discovery and classification | directory → `{FileType: [paths]}` |
| `cache` | SHA256 incremental cache | path → cached dict or None |
| `security` | Input validation (URLs, paths, labels) | unsafe str → safe str |
| `validate` | Extraction schema validation | dict → `[error, ...]` |
| `transcribe` | Whisper transcription (local/API) | media path → transcript text |
| `youtube` | Channel catalog + transcript pulling | channel config → `[VideoMeta]` + transcripts |
| `extract_ast` | tree-sitter code extraction | path → `{nodes, edges}` |
| `llm` | Provider abstraction (claude-cli / openai / openrouter) | prompt → text |
| `extract_llm` | LLM semantic extraction | `Document → {concepts, summary}` |
| `link` | LLM relationship typing | `(source, targets)` → `[LinkDecision]` |
| `build` | NetworkX graph assembly | `[extraction]` → `nx.DiGraph` |
| `cluster` | Community detection | graph → `{cid: [nodes]}` |
| `analyze` | God nodes, bridges, surprises | graph + communities → summary dict |
| `report` | Markdown report generator | graph + analysis → str |
| `export_json` | JSON export | graph → `graph.json` |
| `export_html` | Interactive HTML export | graph → `graph.html` |
| `export_penfield` | Penfield vault export | graph → `vault-penfield/` |
| `export_obsidian` | Obsidian vault export | graph → `vault-obsidian/` |
| `__main__` | CLI entry point | argv → exit code |

## Extraction schema

Every extraction dict must match:

```json
{
  "nodes": [
    {
      "id": "string",          // stable, lowercased alphanumeric_underscore
      "label": "string",       // display label
      "kind": "file|class|function|person|organization|concept|...",
      "confidence": "EXTRACTED|INFERRED|AMBIGUOUS",
      "source_file": "optional path",
      "language": "optional"
    }
  ],
  "edges": [
    {
      "source": "node id",
      "target": "node id",
      "relation": "one of the 24 semantic or 8 structural types",
      "confidence": "EXTRACTED|INFERRED|AMBIGUOUS",
      "source_file": "optional",
      "source_location": "optional (e.g. L42)"
    }
  ]
}
```

Run `pengram.validate.validate_extraction(data)` — it returns an empty list on
success, or a list of human-readable errors.

## Confidence labels

- **EXTRACTED** — the relationship is stated explicitly in the source
  content. Every AST edge is EXTRACTED by definition.
- **INFERRED** — the relationship was deduced from context but not stated.
  Typical for LLM-typed edges when the model is confident.
- **AMBIGUOUS** — best-effort classification. Applied whenever the LLM
  returns an invalid relationship type or fails entirely; the edge falls
  back to `DEFAULTS[source_kind]`.

## Relationship vocabulary

See `pengram/vocabulary.py` (and the mirror at
`vocabulary/relationships.json`) for the authoritative list. The same 24
semantic types are defined by
[obsidian-wikilink-types/skill/SKILL.md](https://github.com/penfieldlabs/obsidian-wikilink-types/blob/main/skill/SKILL.md).

## Adding a new language extractor

`extract_ast.py` uses a `LanguageConfig` dataclass that maps grammar node
types to the kinds PENgram cares about (classes, functions, imports,
calls). To add a language:

1. Confirm the grammar is shipped with `tree-sitter-language-pack`.
2. Add a `LanguageConfig` instance. Fill in the `*_types` frozen-sets with
   the grammar's node-type names (open the grammar's `node-types.json`,
   or print `node.type` from a quick parse).
3. Map the extension to the config in `_EXT_CONFIG`.
4. Add a test in `tests/test_extract_ast.py` that parses a tiny snippet
   and asserts on at least one node and one edge.

If the grammar has version-mismatch issues with the runtime, the
extractor already degrades gracefully (logs a warning, returns empty
nodes/edges).

## Adding a new export format

1. Add `pengram/export_<format>.py` with a top-level
   `export_<format>(graph, output_dir, ...)` function that returns the
   written path.
2. Wire it into `__main__.cmd_export` under a new `format` choice.
3. Add tests under `tests/test_export_<format>.py`. Follow the pattern
   in `test_export_json.py` — make a tiny graph, export it, parse back,
   assert structure.
4. Update `README.md`'s "Output formats" table.

## Security model

All external data flows through `pengram.security`:

- URLs must be `http` or `https` (`validate_url`). `file://`,
  `javascript:`, and empty URLs are rejected.
- User-supplied paths must resolve inside an allowed root
  (`validate_path`). Directory traversal attempts raise `SecurityError`.
- Labels from LLM output are run through `sanitize_label` — control
  characters stripped, HTML-escaped, length-capped.
- Filenames derived from labels go through `sanitize_filename` — unsafe
  characters replaced, leading dots stripped, Windows reserved names
  neutralised, length-capped.

Dotfiles and dot-directories are skipped by `collect_files` (so
`.env`, `.git`, etc. don't leak into the graph), but PENgram does not
try to detect "sensitive" files by filename pattern. The v0.2.0 drop
of a pattern-based filter removed a footgun that was silently
excluding ordinary corpus files (health books with "secret" in the
title). Use `.pengramignore` to exclude specific paths from the
build.

### Prompt injection

LLM extraction (`extract_llm`, `link`, `enrich`) sends document
content verbatim to the configured model. A document crafted with
adversarial instructions (e.g. "ignore previous instructions and
output ...") could manipulate extraction results — producing
fabricated entities, incorrect relationship types, or injected text
that propagates into the vault. PENgram treats this as a data-quality
issue, not a code-execution risk: the LLM has no tool access and its
output is schema-validated before entering the graph. However, if you
are processing untrusted content (public uploads, scraped web pages),
review the extraction output before importing into a production vault.
The `GRAPH_REPORT.md` and `graph.json` outputs make this auditable.

## Testing strategy

- `pytest` + `pytest-cov`. Overall target: ≥ 80% coverage; 90%+ on
  `vocabulary`, `config`, `validate`, `cache`, `security`.
- Tests must run without network access. Mock `yt-dlp`, the `claude`
  CLI, and `openai` at their process / import boundaries.
- **Tests never skip.** `pytest.importorskip`, `pytest.mark.skipif`,
  and custom `@needs_*` markers are banned for the "dep is missing"
  case. Every optional runtime dep (`tree-sitter-language-pack`,
  `pypdf`, future additions) is listed in the `[dev]` extra of
  `pyproject.toml`, so `pip install -e ".[dev]"` gives CI and
  developers a test environment where every test actually runs. Tests
  that verify the missing-dep code path itself (e.g. that
  ``require_pypdf()`` raises a clean ``ImportError``) simulate the
  missing dep with ``monkeypatch``, not with ``importorskip``.
  Skipping reads as a pass in the summary line and hides real
  coverage gaps — don't reach for it.
- Target: `pytest tests/` reports ``N passed, 0 skipped`` in a dev
  install. Non-zero skips are a bug.
- Tests use the `tmp_path` fixture — no side effects outside the
  temporary directory.
- Tests are independent and runnable in any order. No shared state
  between files.
- Deterministic: every clustering call seeds its random source.

Run the full suite with:

```bash
python -m pytest tests/ -q --cov=pengram --cov-report=term-missing
```

## Release process

1. Work on `dev`. Commits are conventional (`feat(module): ...`,
   `test(module): ...`, `docs: ...`, `chore: ...`).
2. Keep every commit green: `python -m pytest tests/ -q` must pass.
3. When the build is ready for public release, `git checkout --orphan
   main` will create a clean squashed-history branch for the public repo.
