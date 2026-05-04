# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Semantic extraction pipeline — LLM results to build-ready dicts.

Converts raw per-document LLM outputs into the ``{nodes, edges}``
format that :func:`pengram.build.build` expects, including entity
canonicalization, document-node creation, co-occurrence linking,
and AMBIGUOUS-default edge pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import config as _config
from . import detect as _detect
from . import extract_llm
from ._ui import say, warn
from .build import normalize_id
from .extract_llm import Document, _coerce_mentions, _fuzzy_key
from .link import Entity, link_all
from .llm import LLMError
from .vocabulary import (
    CONFIDENCE_AMBIGUOUS,
    CONFIDENCE_EXTRACTED,
)


@dataclass(frozen=True)
class RunConfig:
    """Resolved per-run LLM settings — built from CLI args + config defaults.

    Threading this through the pipeline replaces the old pattern of
    mutating ``config.LLM_PROVIDER`` / ``config.LLM`` at the start of
    ``cmd_run``.
    """

    provider: str
    extract_model: str
    link_model: str
    synth_model: str
    image_model: str
    extract_timeout: int
    link_timeout: int
    synth_timeout: int
    chunk_chars: int | None = None

    @classmethod
    def from_args(
        cls,
        *,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        chunk_chars: int | None = None,
    ) -> RunConfig:
        """Build from CLI overrides, falling back to ``config`` defaults."""
        provider = llm_provider or _config.LLM_PROVIDER
        if llm_provider and llm_provider != _config.LLM_PROVIDER:
            prov = _config._DEFAULT_MODELS.get(provider, {})
            extract = prov.get("extract_model", _config.LLM["extract_model"])
            link = prov.get("link_model", _config.LLM["link_model"])
            synth = prov.get("synth_model", _config.LLM["synth_model"])
            image = prov.get("image_model", _config.LLM.get("image_model", "sonnet"))
        else:
            extract = _config.LLM["extract_model"]
            link = _config.LLM["link_model"]
            synth = _config.LLM["synth_model"]
            image = _config.LLM.get("image_model", extract)
        return cls(
            provider=provider,
            extract_model=llm_model or extract,
            link_model=llm_model or link,
            synth_model=synth,
            image_model=image,
            extract_timeout=int(_config.LLM.get("extract_timeout", 300)),
            link_timeout=int(_config.LLM.get("link_timeout", 300)),
            synth_timeout=int(_config.LLM.get("synth_timeout", 300)),
            chunk_chars=chunk_chars,
        )


_LINKER_MAX_TARGETS_PER_SOURCE = 20

_MIN_CO_OCCURRENCE_MENTIONS = 2

_FILE_TYPE_TO_NOTE_KIND: dict[str, str] = {
    _detect.FileType.DOCUMENT.value: "document",
    _detect.FileType.TRANSCRIPT.value: "transcript",
    _detect.FileType.VIDEO.value: "transcript",
    _detect.FileType.AUDIO.value: "transcript",
}

_ENTITY_KINDS: tuple[tuple[str, str], ...] = (("concepts", "concept"),)

_YT_META_PROPAGATE: tuple[str, ...] = (
    "video_id",
    "channel",
    "upload_date",
    "duration",
    "views",
    "likes",
    "comments",
    "tab",
    "url",
)


def _entity_id(node_kind: str, name: str) -> str:
    """Return a stable id for a canonical entity."""
    return f"{node_kind}_{name.lower().replace(' ', '_')}"


def _entity_name(entity: dict[str, Any]) -> str:
    return str(entity.get("name") or entity.get("statement") or "").strip()


def build_semantic_extraction(
    llm_results: list[dict[str, Any]],
    *,
    output_dir: Path,
    run_linker: bool,
    documents: list[Document] | None = None,
    link_model: str | None = None,
    link_timeout: int | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Convert LLM extractions into a build-ready ``{nodes, edges}`` dict.

    Pipeline:
      1. Canonicalize entities across documents (deterministic dedup).
      2. Create one canonical node per entity and one node per document.
         Document nodes carry the verbatim source text in ``body``
         (passthrough, not a summary) and a ``note_type`` derived from
         the file's detected type (``document`` or ``transcript``).
         Entity nodes carry the LLM's per-entity ``note`` and an
         ``appearances`` list of doc nodes that reference them.
      3. Emit ``doc --references--> entity`` edges (factual co-occurrence).
      4. When ``run_linker`` is true, call the LLM linker on co-occurring
         entity pairs. Pairs are pruned to those whose endpoints both
         have >=2 mentions in the same doc. After linking, drop edges
         whose relation is the vocabulary default for the source kind
         AND whose confidence is ``AMBIGUOUS``.

    The pipeline is deterministic given the same LLM results: entity
    IDs are derived from canonical names, edges are emitted in
    document order, and results are sorted by ``doc_id``. Re-running
    with cached LLM results produces identical output.
    """
    docs_by_id: dict[str, Document] = {d.doc_id: d for d in documents} if documents else {}

    canonical = extract_llm.canonicalize_entities(llm_results)

    nodes: dict[str, dict[str, Any]] = {}

    def _entity_note(entity: dict[str, Any]) -> str:
        return str(entity.get("note") or "").strip()

    fuzzy_to_canonical: dict[str, str] = {}
    notes_by_id: dict[str, str] = {}
    for kind_key, node_kind in _ENTITY_KINDS:
        for entity in canonical.get(kind_key, []) or []:
            if not isinstance(entity, dict):
                continue
            name = _entity_name(entity)
            if not name:
                continue
            node_id = _entity_id(node_kind, name)
            fk = f"{node_kind}:{_fuzzy_key(name)}"
            fuzzy_to_canonical[fk] = node_id
            nodes[node_id] = {
                "id": node_id,
                "label": name,
                "kind": node_kind,
                "mentions": _coerce_mentions(entity.get("mentions")),
                "confidence": CONFIDENCE_EXTRACTED,
            }

    edges: list[dict[str, Any]] = []
    doc_entities: dict[str, dict[str, int]] = {}
    appearances: dict[str, list[str]] = {}

    for result in llm_results:
        doc_id = result.get("_doc_id") or "document"
        source = str(result.get("_source") or doc_id)
        doc = docs_by_id.get(doc_id)
        label = (doc.metadata.get("title") if doc else None) or Path(source).name or doc_id
        doc_node_id = f"doc_{doc_id}"
        file_type = (doc.metadata.get("file_type") if doc else None) or "document"
        note_kind = _FILE_TYPE_TO_NOTE_KIND.get(file_type, "document")
        node_attrs: dict[str, Any] = {
            "id": doc_node_id,
            "label": label,
            "kind": note_kind,
            "source_file": Path(source).name,
            "source_path": ""
            if str(rel_dir := PurePosixPath(doc_id).parent) in (".", "")
            else str(rel_dir),
            "summary": str(result.get("summary", "")),
            "body": doc.text
            if doc
            else (
                str(result.get("summary", ""))
                if doc_id.startswith("image:") and result.get("summary")
                else ""
            ),
            "confidence": CONFIDENCE_EXTRACTED,
        }
        if doc_id.startswith("image:"):
            node_attrs["_abs_source_path"] = source
        if doc:
            for key in _YT_META_PROPAGATE:
                val = doc.metadata.get(key)
                if val is not None:
                    node_attrs[key] = val
            # Frontmatter uses "views"/"likes"; export_catalog expects
            # "view_count"/"like_count" — alias both for compat.
            if "views" in doc.metadata:
                node_attrs["view_count"] = doc.metadata["views"]
            if "likes" in doc.metadata:
                node_attrs["like_count"] = doc.metadata["likes"]
        nodes[doc_node_id] = node_attrs
        per_doc_mentions: dict[str, int] = {}
        for kind_key, node_kind in _ENTITY_KINDS:
            for entity in result.get(kind_key, []) or []:
                if not isinstance(entity, dict):
                    continue
                name = _entity_name(entity)
                if not name:
                    continue
                fk = f"{node_kind}:{_fuzzy_key(name)}"
                node_id = fuzzy_to_canonical.get(fk)
                if node_id is None or node_id not in nodes:
                    continue
                note = _entity_note(entity)
                if note and len(note) > len(notes_by_id.get(node_id, "")):
                    notes_by_id[node_id] = note

                doc_mentions = _coerce_mentions(entity.get("mentions"))
                prev = per_doc_mentions.get(node_id, 0)
                per_doc_mentions[node_id] = max(prev, doc_mentions)
                appearances.setdefault(node_id, []).append(doc_node_id)
                edges.append(
                    {
                        "source": doc_node_id,
                        "target": node_id,
                        "relation": "references",
                        "confidence": CONFIDENCE_EXTRACTED,
                    }
                )
        doc_entities[doc_node_id] = per_doc_mentions

    for node_id, note in notes_by_id.items():
        if node_id in nodes:
            nodes[node_id]["note"] = note
    for node_id, doc_ids in appearances.items():
        if node_id in nodes:
            seen: set[str] = set()
            ordered: list[str] = []
            for d in doc_ids:
                norm = normalize_id(d)
                if norm not in seen:
                    seen.add(norm)
                    ordered.append(norm)
            nodes[node_id]["appearances"] = ordered

    link_stats = None
    if run_linker and any(doc_entities.values()):
        link_pairs = build_link_pairs(nodes, doc_entities)
        if link_pairs:
            try:
                decisions, link_stats = link_all(
                    link_pairs,
                    output_dir=output_dir,
                    model=link_model,
                    timeout=link_timeout,
                    provider=provider,
                )
            except (ImportError, LLMError) as exc:
                warn(f"Linking skipped: {exc}")
                decisions = []
            dropped = 0
            for decision in decisions:
                if decision.confidence == CONFIDENCE_AMBIGUOUS:
                    dropped += 1
                    continue
                edges.append(
                    {
                        "source": decision.source,
                        "target": decision.target,
                        "relation": decision.relation,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                    }
                )
            if dropped:
                say(f"  Dropped {dropped} AMBIGUOUS default edges from the linker.")

    result: dict[str, Any] = {"nodes": list(nodes.values()), "edges": edges}
    if link_stats is not None:
        result["_link_stats"] = link_stats
    return result


def build_link_pairs(
    nodes: dict[str, dict[str, Any]],
    doc_entities: dict[str, dict[str, int]],
) -> list[tuple[Entity, list[Entity]]]:
    """Produce one ``(source, targets)`` pair per entity from co-occurrence.

    For each entity, collect the set of other entities it co-occurs with in
    the same document where BOTH entities have at least
    ``_MIN_CO_OCCURRENCE_MENTIONS`` mentions in that document.

    Duplicate pairs are suppressed across documents and target lists are
    capped to ``_LINKER_MAX_TARGETS_PER_SOURCE``.
    """
    co_occurrence: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for doc_id in sorted(doc_entities):
        entity_mentions = doc_entities[doc_id]
        strong = [
            eid
            for eid, mentions in entity_mentions.items()
            if mentions >= _MIN_CO_OCCURRENCE_MENTIONS
        ]
        for src_id in strong:
            for tgt_id in strong:
                if src_id == tgt_id:
                    continue
                key = (src_id, tgt_id)
                if key in seen:
                    continue
                seen.add(key)
                co_occurrence.setdefault(src_id, []).append(tgt_id)

    def _context_for(entity_id: str) -> str:
        bits: list[str] = []
        for doc_id, members in doc_entities.items():
            if entity_id in members:
                summary = nodes.get(doc_id, {}).get("summary", "")
                if summary:
                    bits.append(summary)
        return " ".join(bits)[:2000]

    pairs: list[tuple[Entity, list[Entity]]] = []
    for src_id in sorted(co_occurrence):
        src_node = nodes[src_id]
        source = Entity(
            id=src_id,
            name=str(src_node.get("label", src_id)),
            kind=str(src_node.get("kind", "concept")),
            context=_context_for(src_id),
        )
        targets: list[Entity] = []
        for tgt_id in co_occurrence[src_id][:_LINKER_MAX_TARGETS_PER_SOURCE]:
            tgt_node = nodes[tgt_id]
            targets.append(
                Entity(
                    id=tgt_id,
                    name=str(tgt_node.get("label", tgt_id)),
                    kind=str(tgt_node.get("kind", "concept")),
                )
            )
        if targets:
            pairs.append((source, targets))
    return pairs


__all__ = [
    "RunConfig",
    "build_semantic_extraction",
    "build_link_pairs",
]
