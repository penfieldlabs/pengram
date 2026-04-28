# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""LLM-powered semantic extraction.

Three-phase pipeline — the same prompts work for any content type
(documents, transcripts, …):

1. **Extract** per-document concepts (cheap model).
2. **Canonicalize** across documents to deduplicate concepts by
   case-insensitive name.
3. **Synthesize** a hierarchical taxonomy from the canonical set (heavy
   model, optional).

Each per-document extraction is persisted as a JSON file under
``output_dir/extractions/<doc_id>.json`` so a crash resumes cleanly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cache as _cache
from . import config as _config
from ._ui import step as _ui_step
from ._ui import warn as _ui_warn
from .llm import LLMError, call_llm, parse_json_response
from .security import sanitize_filename

EXTRACT_PROMPT = """\
You are an entity extractor. Read the content below and produce STRICT JSON
with this schema:

{
  "concepts": [{"name": "...", "mentions": 1, "note": "..."}],
  "summary": "one paragraph"
}

Rules:
- Extract concepts: ideas, technologies, theories, methods, mechanisms,
  substances, practices, studies, trials, significant occurrences —
  anything that represents a discrete, nameable piece of knowledge in
  the text. Do NOT extract people, organizations, predictions, or
  topics — names appear naturally in the summary and concept notes;
  grouping by topic is handled downstream by clustering.
- Use canonical names ("OpenAI", not "open ai").
- mentions is an integer count of how many times the entity appears.
- Prefer precision over recall: do not invent entities.
- Respond with JSON only, no commentary, no markdown fences.

CONTENT:
{content}
"""


@dataclass
class Document:
    """A content blob to extract entities from."""

    doc_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _extraction_path(output_dir: Path, doc_id: str) -> Path:
    return output_dir / "extractions" / f"{sanitize_filename(doc_id)}.json"


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping fixed-char chunks.

    Last chunk absorbs any remainder. Short inputs that already fit in
    one chunk return a single-element list, which keeps the single-call
    path identical to the pre-chunking behaviour.
    """
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    step = max(1, chunk_chars - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _fold_legacy_events(result: Any) -> dict[str, Any]:
    """Absorb legacy ``events`` arrays into ``concepts``.

    The v0.2.0 'kill events' directive collapsed the two-type taxonomy
    into one. Cached results from before the change still contain an
    ``events`` key; rather than invalidate the whole cache (wasteful —
    the entities were already extracted), we fold those entries into
    ``concepts`` as-is on read. Events and concepts share the same
    ``{name, mentions, note}`` shape so the fold is lossless for the
    fields we use.
    """
    if not isinstance(result, dict):
        return result
    legacy = result.get("events")
    if not legacy:
        # Still drop the key if it's present-but-empty so downstream
        # doesn't see a stale field.
        if "events" in result:
            result.pop("events", None)
        return result
    concepts = list(result.get("concepts") or [])
    for entity in legacy:
        if isinstance(entity, dict):
            concepts.append(entity)
    result["concepts"] = concepts
    result.pop("events", None)
    return result


def _merge_chunk_extractions(per_chunk: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-chunk ``{concepts, summary}`` dicts.

    Union concepts by case-insensitive name, sum their mentions, keep the
    longest ``note`` observed. Summary falls back to the first chunk's
    summary (cheap; a future enrichment pass can synthesise across
    chunks if ever desired).
    """
    merged: dict[str, dict[str, Any]] = {}
    for result in per_chunk:
        for entity in result.get("concepts") or []:
            if not isinstance(entity, dict):
                continue
            key = str(entity.get("name") or "").strip().lower()
            if not key or len(key) < 2 or key.isdigit():
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(entity)
            else:
                existing["mentions"] = (existing.get("mentions", 1) or 1) + (
                    entity.get("mentions", 1) or 1
                )
                new_note = str(entity.get("note") or "")
                old_note = str(existing.get("note") or "")
                if len(new_note) > len(old_note):
                    existing["note"] = new_note
    return {
        "concepts": list(merged.values()),
        "summary": (per_chunk[0].get("summary") if per_chunk else "") or "",
    }


def extract_entities(
    document: Document,
    *,
    model: str | None = None,
    timeout: int | None = None,
    llm: Callable[..., str] | None = None,
    chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Run the per-document extraction prompt. Returns the parsed JSON dict.

    When ``len(document.text)`` exceeds ``chunk_chars`` (default:
    :func:`pengram.config.default_chunk_chars`) the text is split into
    overlapping chunks and each chunk is extracted independently, then
    merged. The merged result still describes the whole document — the
    caller sees a single extraction dict whether chunking kicked in or
    not.
    """
    caller = llm if llm is not None else call_llm
    cc = chunk_chars if chunk_chars is not None else _config.default_chunk_chars()
    ov = chunk_overlap if chunk_overlap is not None else _config.default_chunk_overlap()
    chunks = _chunk_text(document.text, cc, ov)

    per_chunk: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        prompt = EXTRACT_PROMPT.replace("{content}", chunk)
        try:
            response = caller(
                prompt,
                model=model or _config.LLM["extract_model"],
                timeout=timeout or _config.LLM["extract_timeout"],
                provider=provider,
            )
            data = parse_json_response(response)
            if not isinstance(data, dict):
                raise LLMError("extraction response was not a JSON object")
            # Fold any stray legacy 'events' (e.g. an older prompt
            # variant still deployed somewhere) into 'concepts'.
            data = _fold_legacy_events(data)
            per_chunk.append(data)
        except LLMError as exc:
            # One bad chunk must not kill a long document's extraction.
            # Log which chunk failed and keep going; the others merge fine.
            _ui_warn(f"chunk {i + 1}/{len(chunks)} failed for {document.doc_id}: {exc}")

    if chunks and not per_chunk:
        raise LLMError(f"all {len(chunks)} chunk(s) failed for {document.doc_id}")

    # Whitelist keys at the boundary. Even single-chunk paths go through
    # the merge helper so legacy kinds from a stale prompt are dropped.
    merged = _merge_chunk_extractions(per_chunk)
    merged["_doc_id"] = document.doc_id
    merged["_source"] = document.source
    return merged


def extract_many(
    documents: Iterable[Document],
    *,
    output_dir: Path,
    cache_root: Path | None = None,
    model: str | None = None,
    timeout: int | None = None,
    workers: int | None = None,
    llm: Callable[..., str] | None = None,
    chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Extract entities from many documents with caching and parallelism.

    Caching has two layers:

    - Content-hash cache (``cache_root/.pengram-cache/<sha256>.json``) —
      keyed on the document file's content, so the same input produces
      the same cached result regardless of ``output_dir``. This is what
      makes the pipeline idempotent: two runs into different output
      directories share the cache.
    - Human-readable per-document dump at
      ``output_dir/extractions/<doc_id>.json`` for inspection. This is a
      secondary output only.

    Results are always returned sorted by ``doc_id`` so downstream
    processing (dedup, edge creation) is deterministic regardless of
    thread completion order.
    """
    output_dir = Path(output_dir)
    (output_dir / "extractions").mkdir(parents=True, exist_ok=True)
    workers = workers or int(_config.LLM.get("parallel_workers", 4))
    caller = llm if llm is not None else call_llm

    def _dump_extraction(doc_id: str, result: dict[str, Any]) -> None:
        try:
            _extraction_path(output_dir, doc_id).write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _run(doc: Document) -> dict[str, Any]:
        source_path = Path(doc.source) if doc.source else None
        has_content_cache = (
            cache_root is not None and source_path is not None and source_path.exists()
        )

        # When we have a content-hash cache, it is authoritative: on miss we
        # re-extract even if a stale dump exists in this output_dir. This
        # keeps the pipeline idempotent and makes "edit the file, rerun"
        # actually re-run the LLM.
        if has_content_cache:
            cached = _cache.load_cached(cache_root, source_path)  # type: ignore[arg-type]
            if isinstance(cached, dict) and cached.get("_doc_id"):
                cached = _fold_legacy_events(cached)
                _dump_extraction(doc.doc_id, cached)
                return cached
        else:
            # No content-hash cache available — fall back to the per-output
            # dump so a crashed run can still resume.
            dump = _extraction_path(output_dir, doc.doc_id)
            if dump.exists():
                try:
                    return _fold_legacy_events(json.loads(dump.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    pass

        result = extract_entities(
            doc,
            model=model,
            timeout=timeout,
            llm=caller,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            provider=provider,
        )
        _dump_extraction(doc.doc_id, result)
        if has_content_cache:
            _cache.save_cached(cache_root, source_path, result)  # type: ignore[arg-type]
        return result

    results: list[dict[str, Any]] = []
    docs = list(documents)
    total = len(docs)
    progress_step = max(1, total // 10) if total else 1
    completed = 0

    def _warn(doc: Document, exc: BaseException) -> None:
        _ui_warn(f"extraction failed for {doc.doc_id}: {exc.__class__.__name__}: {exc}")

    def _tick() -> None:
        nonlocal completed
        completed += 1
        if completed % progress_step == 0 or completed == total:
            _ui_step(completed, total, "Extraction")

    if workers <= 1 or len(docs) <= 1:
        for doc in docs:
            try:
                results.append(_run(doc))
            except Exception as exc:
                _warn(doc, exc)
            _tick()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, d): d for d in docs}
            for fut in as_completed(futures):
                doc = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    _warn(doc, exc)
                _tick()

    # Deterministic ordering, independent of thread-completion order.
    results.sort(key=lambda r: r.get("_doc_id") or "")
    return results


def canonicalize_entities(
    extractions: list[dict[str, Any]],
    *,
    model: str | None = None,
    timeout: int | None = None,
    llm: Callable[..., str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return a dict of canonicalised entity lists keyed by kind.

    The only extracted kind is ``concepts``. Merging is deterministic:
    exact-match (case-insensitive) name collisions combine mentions.
    ``llm``, ``model`` and ``timeout`` are accepted for API symmetry
    with the other extraction-phase helpers but are unused — the v0.2
    enrichment pass does the fuzzier surface-form dedup.

    Legacy ``events`` arrays (from before the kill-events directive)
    are folded into concepts on the way in so cached extractions still
    flow through cleanly.
    """
    del model, timeout, llm  # reserved; see docstring.
    merged: dict[str, dict[str, Any]] = {}
    for extraction in extractions:
        extraction = _fold_legacy_events(extraction)
        for entity in extraction.get("concepts", []):
            if not isinstance(entity, dict):
                continue
            key = (entity.get("name") or "").strip().lower()
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(entity)
            else:
                existing["mentions"] = (existing.get("mentions", 1) or 1) + (
                    entity.get("mentions", 1) or 1
                )
    return {"concepts": list(merged.values())}


__all__ = [
    "Document",
    "extract_entities",
    "extract_many",
    "canonicalize_entities",
    "EXTRACT_PROMPT",
]
