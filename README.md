# PENgram

**Parse. Extract. Normalize.**

PENgram takes raw content — code, documents, YouTube channels, PDFs, audio,
images — extracts entities and typed relationships, and outputs a structured
knowledge graph ready for [Penfield](https://penfield.app), Obsidian, or any
graph-aware tool.

PENgram is a personal knowledge-management tool. It is not intended for bulk
scraping or republication of copyrighted material. When using the YouTube
pipeline, you are responsible for complying with YouTube's Terms of Service.

## How it works

PENgram uses a three-pass architecture:

1. **Deterministic** — tree-sitter extracts classes, functions, imports, and
   call graphs from code. No model inference, no tokens burnt.
2. **Local** — faster-whisper transcribes audio and video on CPU or GPU. No
   API calls required.
3. **LLM** — a small model (Claude Haiku / GPT-4o-mini by default) extracts
   entities and topics per document; a larger model synthesizes taxonomy;
   and a typing pass assigns one of the 24 Penfield semantic relationship
   types to every inferred edge.

All extractions are content-hashed (SHA256) and cached on disk, so re-runs
only reprocess what changed. Per-document and per-entity LLM calls are
written to disk as they complete, so a crash resumes cleanly.

**Idempotent vs reproducible.** Re-running PENgram on unchanged input
produces identical output (idempotent). But the pipeline is not
byte-reproducible across machines or model versions — LLM responses are
inherently non-deterministic, so a fresh extraction of the same content
may produce slightly different entities and edges.

## Quick start

```bash
pip install 'pengram[all]'      # or `[code]` / `[video]` / `[youtube]`
pengram run ./my-project        # write output to ./pengram-out/
open pengram-out/graph.html     # interactive visualization
cat pengram-out/GRAPH_REPORT.md # god nodes, surprising connections, questions
```

## Input types

| Input type | Extensions | Extractor |
|---|---|---|
| Code (25 languages) | `.py` `.js` `.ts` `.go` `.rs` `.java` `.c` `.cpp` `.rb` `.cs` `.kt` `.scala` `.php` `.swift` `.lua` `.zig` `.ps1` `.ex` `.m` `.jl` `.dart` `.v` `.vue` `.svelte` ... | tree-sitter (deterministic) |
| Documents | `.md` `.txt` `.rst` `.html` `.pdf` `.epub` | LLM (pypdf / ebooklib when needed) |
| YouTube | channel URLs | yt-dlp (captions) |
| Audio/Video | `.mp3` `.wav` `.mp4` `.mov` `.webm` ... | Whisper transcription (local or OpenAI API) → LLM extraction |
| Transcripts | `.transcript` `.vtt` `.srt` | pass-through (YouTube pipeline writes `.transcript`) |
| Images | `.png` `.jpg` `.gif` `.webp` `.bmp` `.tif` ... | vision LLM (entity extraction from visual content) |

**Tip — image-heavy corpora:** Each image yields `mentions: 1` per concept
(one image = one occurrence). The default vault threshold is 3 mentions, so
image-only projects may produce a graph but no vault notes. Pass
`--threshold-concept 1` to keep every extracted concept.

## Output formats

| File | Always produced | Purpose |
|---|---|---|
| `graph.json` | yes | Queryable graph — nodes, edges, communities |
| `graph.html` | yes | Interactive visualization — opens in any browser |
| `GRAPH_REPORT.md` | yes | God nodes, surprising connections, questions |
| `vault-penfield/` | when `OUTPUT_TARGET=penfield` or `both` | Penfield-compliant vault |
| `vault-obsidian/` | when `OUTPUT_TARGET=obsidian` or `both` | Obsidian vault with wikilink-types |

## Pipeline health

`GRAPH_REPORT.md` includes a Pipeline Health table showing success, empty,
and failure counts for each processing phase:

```
## Pipeline Health

| Phase        | Total | Succeeded | Empty | Failed |
| ---          | ---   | ---       | ---   | ---    |
| Extraction   | 42    | 38        | 3     | 1      |
| Linking      | 127   | 124       | 0     | 3      |
| Enrichment   | 89    | 87        | 2     | 0      |
| Transcripts  | 500   | 483       | 0     | 17     |
```

"Empty" means the phase ran but produced no output (e.g. a document with
no extractable entities). "Failed" means the phase threw an error (LLM
timeout, network failure). Failed documents are retried on the next run.

## PENgram + Penfield

PENgram builds the graph. [Penfield](https://penfield.app) consumes it.

PENgram's `OUTPUT_TARGET=penfield` export writes a vault of Markdown files
with YAML frontmatter — one file per entity, relationships encoded as
frontmatter keys with wikilink-array values. This is the format
[penfield-import](https://github.com/penfieldlabs/penfield-import) expects.
The typical workflow:

1. **PENgram** processes a corpus (code, documents, YouTube channels) and
   writes `pengram-out/vault-penfield/`.
2. **penfield-import** reads the vault, resolves wikilinks, and loads nodes
   and edges into the Penfield graph store.
3. **Penfield** provides search, traversal, and visualization over the
   resulting knowledge graph.

You don't need Penfield to use PENgram — `graph.json`, `graph.html`, and
`GRAPH_REPORT.md` are always produced and work standalone. The Obsidian
export (`OUTPUT_TARGET=obsidian`) is also independent of Penfield. But when
the two are paired, the typed-relationship vocabulary flows end-to-end:
PENgram assigns one of the 24 semantic types to every edge, and Penfield
preserves those types through import.

## Relationship vocabulary

PENgram uses the Penfield 24-type semantic vocabulary plus 8 structural
types for code. The canonical definitions live in
[`vocabulary/relationships.json`](vocabulary/relationships.json) and
[obsidian-wikilink-types](https://github.com/penfieldlabs/obsidian-wikilink-types).

- **Knowledge evolution:** `supersedes`, `updates`, `evolution_of`
- **Evidence:** `supports`, `contradicts`, `disputes`
- **Hierarchy:** `parent_of`, `child_of`, `sibling_of`, `composed_of`, `part_of`
- **Causation:** `causes`, `influenced_by`, `prerequisite_for`
- **Implementation:** `implements`, `documents`, `tests`, `example_of`
- **Conversation:** `responds_to`, `references`, `inspired_by`
- **Sequence:** `follows`, `precedes`
- **Dependencies:** `depends_on`
- **Structural (code):** `calls`, `imports`, `uses`, `extends`, `implements_interface`, `instantiates`, `overrides`, `decorates`

Every edge carries a confidence label: `EXTRACTED` (stated), `INFERRED`
(deduced), or `AMBIGUOUS` (best-effort).

## YouTube channels

Configure channels in `pengram/config.py`:

```python
from pengram.config import YouTubeChannel

YOUTUBE_CHANNELS = {
    "mychannel": YouTubeChannel(
        url="https://youtube.com/@your-channel",
        label="My Channel",
    ),
    "another": YouTubeChannel(
        url="https://youtube.com/@another-channel",
        label="Another Channel",
        tabs=["videos", "streams"],  # also pull past livestreams
    ),
}
```

Then:

```bash
pengram youtube mychannel          # default: up to 50 new videos
pengram youtube mychannel --max-videos 10
```

The YouTube pipeline accesses only publicly available metadata and captions.
No cookies or authentication tokens are used. A default 2-second delay is
inserted between requests to stay well within rate limits.

**No captions?** Some channels have no auto-generated or manual captions
(common for IR/corporate channels). `pengram youtube` will report this
as `no_subtitles`. To extract content from uncaptioned videos, download
the audio manually and feed it through the regular ingest pipeline:

```bash
yt-dlp -x --audio-format m4a -o "input/%(title)s.%(ext)s" <channel-url>
pengram run input/
```

Whisper (local via `pengram[video]` or via OpenAI API) will transcribe.

### Transcript workflow

The YouTube pipeline writes one `.transcript` file per video into
`pengram-out/transcripts/`. These files are the user-facing contract:

- **Edit a `.transcript` file** to correct OCR errors, remove filler, or
  add annotations. On the next run, PENgram re-extracts entities from
  your edited text (the content hash changed) but does not re-download.
- **Delete a `.transcript` file** to force a fresh download and VTT
  cleaning pass.
- **Leave it alone** and PENgram skips both download and extraction
  (content unchanged, cache hit).

Each `.transcript` includes YAML frontmatter (video ID, title, views,
upload date) followed by the cleaned transcript text.

### YouTube video states

The state file (`pengram-out/transcripts/_state.json`) tracks each
video's transcript status:

| Status | Meaning | Retried? |
|---|---|---|
| `ok` | Transcript downloaded and cleaned | no |
| `no_subtitles` | Video has no captions (auto or manual) | no |
| `empty_after_cleaning` | Captions existed but contained no usable text | no |
| `rate_limited` | YouTube returned 429 or sign-in gate | yes (next run) |
| `timeout` | yt-dlp timed out | yes (next run) |
| `error` | Other failure | yes (next run) |

Stale `rate_limited` and `error` entries are automatically reclassified
on load: if the `.transcript` file exists on disk, the status becomes
`ok`; otherwise `no_subtitles`.

**Default: `videos` only.** To pull more, add tabs explicitly:

| Tab | Default? | What it is |
|---|---|---|
| `videos` | yes | Regular uploads |
| `streams` | opt-in | Past livestreams. YouTube's UI labels this tab "Live"; yt-dlp's URL segment is `/streams`. Same content, both names work. |
| `shorts` | opt-in | Short-form vertical clips. Usually too brief for useful extraction. |

## Whisper configuration

Set via environment variable or edit `pengram/config.py`:

```bash
# Local (default): CPU or GPU, no API key
export PENGRAM_WHISPER_MODE=local
export PENGRAM_WHISPER_MODEL=base.en

# OpenAI Whisper API
export PENGRAM_WHISPER_MODE=openai
export OPENAI_API_KEY=sk-...

# OpenRouter (OpenAI-compatible)
export PENGRAM_WHISPER_MODE=openrouter
export OPENROUTER_API_KEY=sk-or-...
```

## LLM configuration

Every provider defaults to its cheapest sensible model. Set env vars or
pass `--llm-model` to override.

```bash
# claude-cli (default): uses `claude -p`, no API key.
export PENGRAM_LLM_PROVIDER=claude-cli
# defaults: extract/link=haiku, synth=sonnet

# OpenAI — cheapest sensible default for every role.
export PENGRAM_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
# defaults: extract/link/synth = gpt-4o-mini

# OpenRouter — same, via the gpt-4o-mini slug on their platform.
export PENGRAM_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
# defaults: extract/link/synth = openai/gpt-4o-mini

# Ollama — local, auto-detects the running model.
# Pulls /api/ps first (what's loaded), falls back to /api/tags
# (what's available). Chunk size derived from the model's num_ctx.
export PENGRAM_LLM_PROVIDER=ollama
export PENGRAM_OLLAMA_BASE_URL=http://localhost:11434  # default
# Or pick explicitly:
pengram run ./project --llm-provider ollama --llm-model qwen2.5:7b
```

Separate knobs for each pipeline phase:

```bash
export PENGRAM_EXTRACT_MODEL=...   # per-doc entity extraction
export PENGRAM_LINK_MODEL=...      # relationship typing
export PENGRAM_SYNTH_MODEL=...     # enrichment definitions + quotes
export PENGRAM_ENRICH_MODEL=...    # overrides synth for enrichment only
```

## Acknowledgments

PENgram's architecture was influenced by
[Graphify](https://github.com/safishamsi/graphify) by Safi Shamsi (MIT
License), particularly its approach to deterministic-first extraction,
SHA256 incremental caching, and Leiden community detection. We adopted
these patterns while building a pipeline optimized for typed relationship
extraction and [Penfield Import.](https://github.com/penfieldlabs/penfield-import)

The relationship vocabulary is defined by
[obsidian-wikilink-types](https://github.com/penfieldlabs/obsidian-wikilink-types).

## License

MIT — see [LICENSE](LICENSE).
