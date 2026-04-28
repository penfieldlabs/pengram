# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Enrichment pass: rich concept notes + surface-form dedup.

Pipeline position:

    extract → build → cluster → **enrich** → category-inject → analyze → export

For every concept node above threshold (default ``concept: 3``):

1. :func:`gather_contexts` harvests each mention's surrounding text
   (±2000 chars) from the source documents the concept appears in. If a
   source document is under 9500 chars, the whole body is passed as one
   context instead of slicing a window.
2. :func:`enrich_concepts` sends one LLM call per concept. The prompt
   returns a JSON block with a short definition, attributed quotes,
   cross-source notes, and (optionally) a ``merge_into`` hint pointing
   at another concept in the same cluster that is the same thing under
   a different name.
3. :func:`apply_enrichment` applies merges first (union-find resolves
   chains / cycles), then attaches the remaining concepts' definitions
   and quotes to their graph nodes.

Cost ceiling: one LLM call per concept above threshold. With Haiku,
~$0.30–0.50 per mid-sized corpus (~250 docs). ``--no-enrich`` skips the whole
pass.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from . import config as _config
from ._ui import say as _ui_say
from ._ui import step as _ui_step
from ._ui import warn as _ui_warn
from .llm import LLMError, call_llm, parse_json_response
from .security import sanitize_filename
from .vocabulary import CONFIDENCE_INFERRED

# How much text around each mention to pull out as a context window.
_CONTEXT_WINDOW = 2000
# Full-document cutoff: under this size, pass the whole doc as the context
# rather than searching for mentions (shorter docs are usually focused
# enough that the full body IS the context).
_FULL_DOC_CUTOFF = 9500
# Per-concept cap on contexts handed to the LLM (prompt size bound).
_MAX_CONTEXTS_PER_CONCEPT = 8
# Per-concept cap on context characters (secondary prompt size bound).
_MAX_CONTEXT_CHARS = 20_000


@dataclass
class ConceptContext:
    """One slice of source text where a concept is discussed."""

    doc_id: str  # the document node id in the graph
    doc_label: str  # the document's display label (for attribution)
    text: str  # the context window (or the whole short doc)


@dataclass
class EnrichmentResult:
    """What the LLM returned for one concept."""

    concept_id: str
    definition: str = ""
    quotes: list[dict[str, str]] = field(default_factory=list)
    cross_source_notes: str = ""
    merge_into_label: str | None = None  # canonical label to merge into, or None

    def is_empty(self) -> bool:
        """True when nothing usable came back from the LLM.

        Covers both parse failure (JSON rejected upstream) and the case
        where the model returned a structurally-valid but vacant response.
        Empty results are NOT cached — next run retries instead of
        pinning the concept to a permanent no-op.
        """
        return (
            not self.definition
            and not self.quotes
            and not self.cross_source_notes
            and not self.merge_into_label
        )

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialise to the shape written under ``enrichment_cache/<id>.json``."""
        return {
            "concept_id": self.concept_id,
            "definition": self.definition,
            "quotes": self.quotes,
            "cross_source_notes": self.cross_source_notes,
            "merge_into": self.merge_into_label,
        }

    @classmethod
    def from_cache_dict(cls, data: Mapping[str, Any]) -> EnrichmentResult:
        quotes_raw = data.get("quotes") or []
        quotes: list[dict[str, str]] = [
            {
                "text": str(q.get("text") or ""),
                "source": str(q.get("source") or ""),
            }
            for q in quotes_raw
            if isinstance(q, Mapping)
        ]
        merge = data.get("merge_into")
        return cls(
            concept_id=str(data.get("concept_id") or ""),
            definition=str(data.get("definition") or ""),
            quotes=quotes,
            cross_source_notes=str(data.get("cross_source_notes") or ""),
            merge_into_label=str(merge).strip()
            if isinstance(merge, str) and merge.strip()
            else None,
        )


ENRICH_PROMPT = """\
You are enriching a knowledge-graph concept node with definition, quotes,
and dedup guidance.

CONCEPT
  name: {concept_name}
  mentions across the corpus: {mention_count}

OTHER CONCEPTS IN THE SAME CLUSTER (candidate duplicates)
{cluster_roster}

SOURCE CONTEXTS (from documents that discuss this concept)
{contexts}

Produce STRICT JSON with this schema:

{
  "definition": "2-3 sentences explaining this concept in the corpus's
                 own voice",
  "quotes": [
    {"text": "verbatim sentence from source", "source": "doc label"}
  ],
  "cross_source_notes": "notes on how the concept appears across different
                          documents / content types (empty string if N/A)",
  "merge_into": null OR "one of the OTHER CONCEPT names above if it is
                          the same thing under a different surface form"
}

Rules:
- Prefer precision over recall. Empty arrays and empty strings are fine.
- Quote text verbatim from the contexts. Do not paraphrase quotes.
- merge_into is OFF by default. Only set it when you're confident the two
  names refer to the same entity ("AMPK" / "AMP kinase", "Belief" /
  "Beliefs"). If uncertain, leave it null.
- Respond with JSON only. No commentary, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _doc_bodies(g: nx.Graph) -> dict[str, tuple[str, str]]:
    """Return ``{doc_node_id: (label, body)}`` for every content node."""
    out: dict[str, tuple[str, str]] = {}
    for node_id in g.nodes:
        node = g.nodes[node_id]
        body = node.get("body")
        if not isinstance(body, str) or not body.strip():
            continue
        label = str(node.get("label", node_id))
        out[node_id] = (label, body)
    return out


def _find_mention_windows(
    body: str,
    name: str,
    *,
    window: int = _CONTEXT_WINDOW,
) -> list[str]:
    """Return ±``window``-char slices around each case-insensitive match."""
    if not body or not name:
        return []
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    slices: list[str] = []
    for match in pattern.finditer(body):
        start = max(0, match.start() - window)
        end = min(len(body), match.end() + window)
        slices.append(body[start:end].strip())
    return slices


def gather_contexts(
    g: nx.Graph,
    concept_id: str,
    *,
    window: int = _CONTEXT_WINDOW,
    full_doc_cutoff: int = _FULL_DOC_CUTOFF,
    max_contexts: int = _MAX_CONTEXTS_PER_CONCEPT,
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> list[ConceptContext]:
    """Harvest source contexts for a concept from the graph.

    For each document that references the concept, pull either the whole
    body (if the doc is short) or ±``window``-char slices around every
    mention of the concept's label. Caps on context count and total
    characters bound prompt size.
    """
    if concept_id not in g.nodes:
        return []
    concept = g.nodes[concept_id]
    name = str(concept.get("label", concept_id)).strip()
    if not name:
        return []

    contexts: list[ConceptContext] = []
    total = 0
    iterator = (
        g.in_edges(concept_id, data=True)
        if isinstance(g, nx.DiGraph)
        else ((n, concept_id, data) for n, data in g[concept_id].items())
    )
    doc_bodies = _doc_bodies(g)

    # Sort by source id for deterministic output.
    incoming = sorted(iterator, key=lambda e: str(e[0]))
    for src, _tgt, data in incoming:
        if data.get("relation") != "references":
            continue
        if src not in doc_bodies:
            continue
        doc_label, body = doc_bodies[src]
        if len(body) <= full_doc_cutoff:
            text = body
        else:
            windows = _find_mention_windows(body, name, window=window)
            if not windows:
                continue
            text = "\n---\n".join(windows)
        if not text.strip():
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        contexts.append(
            ConceptContext(
                doc_id=src,
                doc_label=doc_label,
                text=text,
            )
        )
        total += len(text)
        if len(contexts) >= max_contexts:
            break
    return contexts


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _cluster_roster_for(
    concept_id: str,
    communities: Mapping[int, list[str]],
    g: nx.Graph,
) -> list[str]:
    """Return the labels of OTHER concepts in the same cluster.

    Used by the dedup prompt so the LLM can spot surface-form duplicates
    ("AMPK" vs "AMP kinase"). Non-concept cluster members are ignored;
    the concept we're enriching is excluded.
    """
    for members in communities.values():
        if concept_id in members:
            return sorted(
                str(g.nodes[m].get("label", m))
                for m in members
                if m != concept_id and g.nodes.get(m, {}).get("kind") == "concept"
            )
    return []


def _build_prompt(
    concept_id: str,
    g: nx.Graph,
    contexts: list[ConceptContext],
    cluster_roster: list[str],
) -> str:
    concept = g.nodes[concept_id]
    name = str(concept.get("label", concept_id))
    roster_lines = "\n".join(f"- {r}" for r in cluster_roster) or "(none)"
    if contexts:
        context_block = "\n\n".join(f"=== source: {c.doc_label} ===\n{c.text}" for c in contexts)
    else:
        context_block = "(no source contexts found)"
    return (
        ENRICH_PROMPT.replace("{concept_name}", name)
        .replace("{mention_count}", str(concept.get("mentions", 1)))
        .replace("{cluster_roster}", roster_lines)
        .replace("{contexts}", context_block)
    )


def _parse_enrichment(concept_id: str, raw: str) -> EnrichmentResult:
    """Map an LLM JSON blob back into an :class:`EnrichmentResult`."""
    try:
        data = parse_json_response(raw)
    except LLMError:
        return EnrichmentResult(concept_id=concept_id)
    if not isinstance(data, dict):
        return EnrichmentResult(concept_id=concept_id)
    quotes_raw = data.get("quotes") or []
    quotes: list[dict[str, str]] = []
    if isinstance(quotes_raw, list):
        for item in quotes_raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            quotes.append(
                {
                    "text": text,
                    "source": str(item.get("source") or "").strip(),
                }
            )
    merge = data.get("merge_into")
    merge_label = str(merge).strip() if isinstance(merge, str) and merge.strip() else None
    return EnrichmentResult(
        concept_id=concept_id,
        definition=str(data.get("definition") or "").strip(),
        quotes=quotes,
        cross_source_notes=str(data.get("cross_source_notes") or "").strip(),
        merge_into_label=merge_label,
    )


def _cache_path_for(cache_dir: Path, concept_id: str) -> Path:
    return cache_dir / f"{sanitize_filename(concept_id)}.json"


def _load_cache(cache_dir: Path | None) -> dict[str, EnrichmentResult]:
    """Return ``{concept_id: EnrichmentResult}`` from disk.

    Missing directory, unreadable files, and malformed JSON are all
    treated as "no cache for that concept" — they never raise.
    """
    if cache_dir is None or not cache_dir.exists():
        return {}
    out: dict[str, EnrichmentResult] = {}
    for path in cache_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _ui_warn(f"enrichment cache unreadable ({path.name}): {exc}")
            continue
        if not isinstance(data, dict):
            _ui_warn(
                f"enrichment cache malformed ({path.name}): expected dict, got {type(data).__name__}"
            )
            continue
        result = EnrichmentResult.from_cache_dict(data)
        if result.concept_id:
            out[result.concept_id] = result
    return out


def _save_cache_entry(cache_dir: Path, result: EnrichmentResult) -> None:
    """Write one enrichment result to disk. Errors are swallowed —
    losing a single cache entry doesn't justify aborting the pass."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path_for(cache_dir, result.concept_id)
        path.write_text(
            json.dumps(result.to_cache_dict(), indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        _ui_warn(f"enrichment cache write failed for {result.concept_id}: {exc}")
        return


def clear_cache(cache_dir: Path) -> int:
    """Delete every enrichment cache file. Returns the count removed."""
    if not cache_dir.exists():
        return 0
    removed = 0
    for path in cache_dir.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _normalize_for_dedup(label: str) -> str:
    """Collapse plurals and hyphenation variants to a canonical key."""
    s = label.strip().lower()
    s = re.sub(r"[-\s]+", "", s)
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def _deterministic_dedup(
    g: nx.Graph,
    concept_ids: list[str],
) -> list[EnrichmentResult]:
    """Catch obvious surface-form duplicates the LLM might miss.

    Groups concepts by a normalised key (lowercased, hyphens removed,
    trailing plural 's' stripped). Within each collision group the
    highest-mention concept is canonical; the rest get merge directives.
    """
    groups: dict[str, list[tuple[str, str, int]]] = {}
    for cid in concept_ids:
        if cid not in g.nodes:
            continue
        node = g.nodes[cid]
        label = str(node.get("label", cid))
        norm = _normalize_for_dedup(label)
        mentions = int(node.get("mentions", 1) or 1)
        groups.setdefault(norm, []).append((cid, label, mentions))

    results: list[EnrichmentResult] = []
    for _norm, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda x: (-x[2], len(x[1]), x[1]))
        canonical_label = members[0][1]
        for cid, _label, _mentions in members[1:]:
            results.append(EnrichmentResult(concept_id=cid, merge_into_label=canonical_label))
    return results


def enrich_concepts(
    g: nx.Graph,
    communities: Mapping[int, list[str]],
    *,
    concept_ids: Iterable[str] | None = None,
    model: str | None = None,
    timeout: int | None = None,
    workers: int | None = None,
    llm: Callable[..., str] | None = None,
    cache_dir: Path | None = None,
    provider: str | None = None,
) -> list[EnrichmentResult]:
    """Run enrichment for every concept that has source contexts.

    When ``cache_dir`` is set, previously-enriched concepts are loaded
    from disk instead of calling the LLM. Each fresh result writes to
    disk immediately after the LLM returns — so a crash at concept
    900/1838 only costs the current concept's work; the next run
    resumes from the 900 cached entries.

    Concepts with no source contexts are skipped entirely (nothing to
    enrich). LLM errors are isolated per-concept — a bad response
    yields an empty ``EnrichmentResult``; the rest of the batch still
    completes.
    """
    caller = llm if llm is not None else call_llm
    workers = workers or int(_config.LLM.get("parallel_workers", 4))
    model = model or _config.LLM.get("synth_model") or _config.LLM.get("extract_model")
    timeout = timeout or int(_config.LLM.get("synth_timeout", 600))

    if concept_ids is None:
        concept_ids = [n for n in g.nodes if g.nodes[n].get("kind") == "concept"]
    ids = sorted(concept_ids)
    if not ids:
        return []

    # Load whatever's been enriched previously. Concepts that appear in
    # the current graph AND in the cache are lifted straight out; only
    # concepts missing from the cache are sent to the LLM.
    cached_results: dict[str, EnrichmentResult] = _load_cache(cache_dir)
    cache_hits: dict[str, EnrichmentResult] = {
        cid: cached_results[cid] for cid in ids if cid in cached_results
    }
    fresh_ids: list[str] = [cid for cid in ids if cid not in cache_hits]

    if cache_hits:
        _ui_say(
            f"  Enrichment: {len(cache_hits)} concept(s) loaded from cache; "
            f"{len(fresh_ids)} to enrich"
        )

    def _run(concept_id: str) -> EnrichmentResult | None:
        contexts = gather_contexts(g, concept_id)
        if not contexts:
            return None
        roster = _cluster_roster_for(concept_id, communities, g)
        prompt = _build_prompt(concept_id, g, contexts, roster)
        try:
            raw = caller(prompt, model=model, timeout=timeout, provider=provider)
        except Exception as exc:
            _ui_warn(f"enrichment failed for {concept_id}: {exc.__class__.__name__}: {exc}")
            return EnrichmentResult(concept_id=concept_id)
        result = _parse_enrichment(concept_id, raw)
        # Persist immediately — a crash at concept N preserves 0..N-1.
        # Empty results (parse failure, vacant response) are deliberately
        # NOT cached: caching would pin the concept to a permanent no-op;
        # skipping the write lets the next run retry with the same or a
        # different LLM.
        if result.is_empty():
            _ui_warn(f"enrichment returned empty for {concept_id}")
        elif cache_dir is not None:
            _save_cache_entry(cache_dir, result)
        return result

    # Progress granularity: ~10 lines total regardless of corpus size.
    total_fresh = len(fresh_ids)
    progress_step = max(1, total_fresh // 10) if total_fresh else 1

    def _announce(done: int) -> None:
        _ui_step(done, total_fresh, "Enrichment")

    fresh_results: dict[str, EnrichmentResult] = {}
    completed = 0
    if total_fresh == 0:
        pass
    elif workers <= 1 or len(fresh_ids) <= 1:
        for concept_id in fresh_ids:
            r = _run(concept_id)
            completed += 1
            if r is not None:
                fresh_results[concept_id] = r
            if completed % progress_step == 0 and completed < total_fresh:
                _announce(completed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, cid): cid for cid in fresh_ids}
            for fut in as_completed(futures):
                completed += 1
                try:
                    r = fut.result()
                except Exception as exc:
                    cid = futures[fut]
                    _ui_warn(f"enrichment failed for {cid}: {exc.__class__.__name__}: {exc}")
                    continue
                if r is not None:
                    fresh_results[r.concept_id] = r
                if completed % progress_step == 0 and completed < total_fresh:
                    _announce(completed)
    if total_fresh:
        _announce(total_fresh)

    combined: dict[str, EnrichmentResult] = {**cache_hits, **fresh_results}
    results = [combined[cid] for cid in ids if cid in combined]

    dedup_merges = _deterministic_dedup(g, ids)
    if dedup_merges:
        _ui_say(f"  Deterministic dedup: {len(dedup_merges)} surface-form duplicate(s) detected")
    for dm in dedup_merges:
        existing = combined.get(dm.concept_id)
        if existing and not existing.merge_into_label:
            existing.merge_into_label = dm.merge_into_label
        elif not existing:
            results.append(dm)

    return results


# ---------------------------------------------------------------------------
# Merge application (union-find)
# ---------------------------------------------------------------------------


def _label_to_concept_id(g: nx.Graph) -> dict[str, str]:
    """Case-insensitive ``label → node_id`` lookup for concept nodes."""
    mapping: dict[str, str] = {}
    for node_id in g.nodes:
        node = g.nodes[node_id]
        if node.get("kind") != "concept":
            continue
        label = str(node.get("label") or "").strip().lower()
        if label and label not in mapping:
            mapping[label] = node_id
    return mapping


def _resolve_merges(
    results: list[EnrichmentResult],
    label_index: Mapping[str, str],
    g: nx.Graph,
) -> dict[str, str]:
    """Return ``{merged_id: canonical_id}`` with chains flattened via union-find.

    A merge decision ``A.merge_into = B-label`` becomes the edge A → B.
    Union-find picks the root of each equivalence class. The root is the
    concept with the highest mention count (ties broken by lower node id
    lexicographically — a stable, deterministic choice).
    """
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return

        # Root choice: higher mentions wins, lexicographic id breaks ties.
        def _score(nid: str) -> tuple[int, str]:
            mentions = int(g.nodes[nid].get("mentions", 1) or 1)
            return (-mentions, nid)

        winner, loser = (ra, rb) if _score(ra) < _score(rb) else (rb, ra)
        parent[loser] = winner

    # Seed every concept as its own root.
    for r in results:
        parent.setdefault(r.concept_id, r.concept_id)

    for r in results:
        target = r.merge_into_label
        if not target:
            continue
        target_id = label_index.get(target.strip().lower())
        if not target_id or target_id == r.concept_id:
            continue
        if target_id not in g.nodes or g.nodes[target_id].get("kind") != "concept":
            continue
        parent.setdefault(target_id, target_id)
        _union(r.concept_id, target_id)

    merge_map: dict[str, str] = {}
    for concept_id in list(parent):
        root = _find(concept_id)
        if root != concept_id:
            merge_map[concept_id] = root
    return merge_map


def _redirect_edges(g: nx.Graph, merged_id: str, canonical_id: str) -> None:
    """Rewire every edge touching ``merged_id`` to ``canonical_id``.

    In-edges: re-add from the original source to the canonical, keeping
    the highest-confidence edge per (src, relation) pair. Out-edges:
    same treatment. Then the ``merged_id`` node is removed.
    """
    if isinstance(g, nx.DiGraph):
        in_edges = list(g.in_edges(merged_id, data=True))
        out_edges = list(g.out_edges(merged_id, data=True))
    else:
        in_edges = [(n, merged_id, data) for n, data in g[merged_id].items()]
        out_edges = []

    for src, _tgt, data in in_edges:
        if src == canonical_id:
            continue
        _add_or_merge_edge(g, src, canonical_id, data)
    for _src, tgt, data in out_edges:
        if tgt == canonical_id:
            continue
        _add_or_merge_edge(g, canonical_id, tgt, data)


def _add_or_merge_edge(g: nx.Graph, src: str, tgt: str, data: dict[str, Any]) -> None:
    if g.has_edge(src, tgt):
        existing = g[src][tgt]
        # Keep the existing attrs; no upgrade unless incoming has higher
        # confidence than AMBIGUOUS and existing is below it.
        existing_conf = existing.get("confidence", "AMBIGUOUS")
        new_conf = data.get("confidence", "AMBIGUOUS")
        rank = {"AMBIGUOUS": 0, "INFERRED": 1, "EXTRACTED": 2}
        if rank.get(new_conf, 0) > rank.get(existing_conf, 0):
            existing.update(data)
    else:
        g.add_edge(src, tgt, **data)


def _mentions_union(src: dict[str, Any], dst: dict[str, Any]) -> None:
    dst_mentions = int(dst.get("mentions", 1) or 1)
    src_mentions = int(src.get("mentions", 1) or 1)
    dst["mentions"] = dst_mentions + src_mentions
    # Keep the longer note.
    src_note = str(src.get("note") or "")
    dst_note = str(dst.get("note") or "")
    if len(src_note) > len(dst_note):
        dst["note"] = src_note
    # Union appearance lists.
    src_app = list(src.get("appearances") or [])
    dst_app = list(dst.get("appearances") or [])
    seen: set[str] = set()
    merged: list[str] = []
    for a in dst_app + src_app:
        if a not in seen:
            seen.add(a)
            merged.append(a)
    if merged:
        dst["appearances"] = merged


def apply_enrichment(
    g: nx.Graph,
    results: list[EnrichmentResult],
) -> tuple[int, int]:
    """Mutate the graph with enrichment data.

    Returns ``(merges_applied, concepts_enriched)``. Merges are applied
    first (union-find roots) so the remaining concepts' enrichments land
    on the surviving canonical nodes.
    """
    if not results:
        return 0, 0

    label_index = _label_to_concept_id(g)
    merges = _resolve_merges(results, label_index, g)

    # Apply merges.
    for merged_id, canonical_id in merges.items():
        if merged_id not in g.nodes or canonical_id not in g.nodes:
            continue
        _mentions_union(dict(g.nodes[merged_id]), g.nodes[canonical_id])
        _redirect_edges(g, merged_id, canonical_id)
        g.remove_node(merged_id)

    # Attach per-concept enrichment to surviving nodes. ``definition``,
    # ``quotes`` and ``cross_source_notes`` are kept separate from the
    # original per-document ``note`` — the concept-body renderer handles
    # layout and dedup between them.
    enriched_count = 0
    for r in results:
        target = merges.get(r.concept_id, r.concept_id)
        if target not in g.nodes:
            continue
        node = g.nodes[target]
        if r.definition:
            node["definition"] = r.definition
            enriched_count += 1
        if r.quotes:
            node["quotes"] = r.quotes
        if r.cross_source_notes:
            node["cross_source_notes"] = r.cross_source_notes
        node.setdefault("confidence", CONFIDENCE_INFERRED)

    # Fallback: promote 'note' to 'definition' for unenriched concepts.
    for node_id in g.nodes:
        node = g.nodes[node_id]
        if node.get("kind") != "concept":
            continue
        if not node.get("definition") and node.get("note"):
            node["definition"] = node["note"]

    return len(merges), enriched_count


__all__ = [
    "ConceptContext",
    "EnrichmentResult",
    "ENRICH_PROMPT",
    "gather_contexts",
    "enrich_concepts",
    "apply_enrichment",
    "clear_cache",
]
