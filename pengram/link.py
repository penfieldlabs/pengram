# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""LLM-powered typed relationship linking.

For a given source entity and a list of candidate target entities, :func:`link_entities`
asks the LLM to pick the most-specific relationship from the 24 Penfield semantic
types. Invalid LLM output falls back to :data:`pengram.vocabulary.DEFAULTS` for
the source entity kind.

Linking decisions are cached per-source on disk under
``output_dir/links/<source>.json`` so re-runs are idempotent and crash-safe.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as _config
from . import vocabulary as v
from ._ui import say as _ui_say
from ._ui import step as _ui_step
from ._ui import warn as _ui_warn
from .llm import LLMError, call_llm, parse_json_response
from .security import sanitize_filename

LINK_PROMPT = """\
You are assigning typed relationships between entities in a knowledge graph.
Use ONLY the 24 Penfield relationship types below. Pick the MOST SPECIFIC type
that applies. If none fit precisely, use a default rather than force a match.

Relationship types (grouped):

- Knowledge evolution: supersedes, updates, evolution_of
- Evidence: supports, contradicts, disputes
- Hierarchy: parent_of, child_of, sibling_of, composed_of, part_of
- Causation: causes, influenced_by, prerequisite_for
- Implementation: implements, documents, tests, example_of
- Conversation: responds_to, references, inspired_by
- Sequence: follows, precedes
- Dependencies: depends_on

Confidence labels:
- EXTRACTED — the relationship is stated explicitly in the source content.
- INFERRED  — you deduced the relationship from context but it is not stated.
- AMBIGUOUS — the relationship is plausible but uncertain.

SOURCE ENTITY
kind: {source_kind}
name: {source_name}
context: {source_context}

TARGET ENTITIES (one per line, as index | name | kind)
{targets}

Respond with STRICT JSON:

{
  "links": [
    {
      "target_index": 0,
      "relation": "...",
      "confidence": "EXTRACTED|INFERRED|AMBIGUOUS",
      "reason": "..."
    }
  ]
}

Skip targets you cannot confidently link.
"""


@dataclass
class Entity:
    """A minimal description of an entity sufficient for linking."""

    id: str
    name: str
    kind: str = "concept"
    context: str = ""


@dataclass
class LinkDecision:
    source: str
    target: str
    relation: str
    confidence: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable dict representation."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _default_relation(kind: str) -> str:
    return v.DEFAULTS.get(kind, "references")


def _format_targets(targets: Iterable[Entity]) -> str:
    lines: list[str] = []
    for i, t in enumerate(targets):
        lines.append(f"{i} | {t.name} | {t.kind}")
    return "\n".join(lines)


def link_entities(
    source: Entity,
    targets: list[Entity],
    *,
    model: str | None = None,
    timeout: int | None = None,
    llm: Callable[..., str] | None = None,
    provider: str | None = None,
) -> list[LinkDecision]:
    """Return one :class:`LinkDecision` per target.

    Invalid or missing LLM output for a target falls back to the default
    relation for the source kind, labelled ``AMBIGUOUS``.
    """
    if not targets:
        return []
    caller = llm if llm is not None else call_llm
    prompt = (
        LINK_PROMPT.replace("{source_kind}", source.kind)
        .replace("{source_name}", source.name)
        .replace("{source_context}", source.context[:2000] or "(no additional context)")
        .replace("{targets}", _format_targets(targets))
    )
    llm_failed = False
    try:
        response = caller(
            prompt,
            model=model or _config.LLM["link_model"],
            timeout=timeout or _config.LLM["link_timeout"],
            provider=provider,
        )
        data = parse_json_response(response)
    except LLMError:
        data = None
        llm_failed = True

    chosen: dict[int, dict[str, Any]] = {}
    if isinstance(data, dict):
        for entry in data.get("links", []) or []:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("target_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(targets):
                chosen[idx] = entry

    decisions: list[LinkDecision] = []
    fallback_relation = _default_relation(source.kind)
    fallback_reason = (
        "LLM call failed; default applied"
        if llm_failed
        else "LLM did not produce a link; default applied"
    )
    for i, target in enumerate(targets):
        entry = chosen.get(i)
        if entry is None:
            decisions.append(
                LinkDecision(
                    source=source.id,
                    target=target.id,
                    relation=fallback_relation,
                    confidence=v.CONFIDENCE_AMBIGUOUS,
                    reason=fallback_reason,
                )
            )
            continue
        relation = entry.get("relation", "")
        if not v.is_valid_semantic(relation):
            relation = fallback_relation
            confidence = v.CONFIDENCE_AMBIGUOUS
        else:
            confidence = entry.get("confidence", v.CONFIDENCE_INFERRED)
            if not v.is_valid_confidence(confidence):
                confidence = v.CONFIDENCE_INFERRED
        decisions.append(
            LinkDecision(
                source=source.id,
                target=target.id,
                relation=relation,
                confidence=confidence,
                reason=str(entry.get("reason", ""))[:400],
            )
        )
    return decisions


def _cache_path(output_dir: Path, source_id: str) -> Path:
    return output_dir / "links" / f"{sanitize_filename(source_id)}.json"


@dataclass
class LinkStats:
    """Phase-level counts from :func:`link_all`."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    error_classes: Counter[str] = field(default_factory=Counter)


def link_all(
    pairs: Iterable[tuple[Entity, list[Entity]]],
    *,
    output_dir: Path,
    model: str | None = None,
    timeout: int | None = None,
    workers: int | None = None,
    llm: Callable[..., str] | None = None,
    provider: str | None = None,
) -> tuple[list[LinkDecision], LinkStats]:
    """Link every ``(source, targets)`` pair with caching and parallelism."""
    output_dir = Path(output_dir)
    (output_dir / "links").mkdir(parents=True, exist_ok=True)
    workers = workers or int(_config.LLM.get("parallel_workers", 4))
    caller = llm if llm is not None else call_llm

    def _run(source: Entity, targets: list[Entity]) -> list[LinkDecision]:
        cache = _cache_path(output_dir, source.id)
        if cache.exists():
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
                return [LinkDecision(**item) for item in payload]
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                _ui_warn(f"link cache unreadable for {source.id}: {exc}")
        decisions = link_entities(
            source,
            targets,
            model=model,
            timeout=timeout,
            llm=caller,
            provider=provider,
        )
        try:
            cache.write_text(
                json.dumps([d.to_dict() for d in decisions], indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            _ui_warn(f"link cache write failed for {source.id}: {exc}")
        return decisions

    all_decisions: list[LinkDecision] = []
    pairs = list(pairs)
    total = len(pairs)
    progress_step = max(1, total // 10) if total else 1
    completed = 0
    error_classes: Counter[str] = Counter()

    if workers <= 1 or total <= 1:
        for source, targets in pairs:
            try:
                all_decisions.extend(_run(source, targets))
            except Exception as exc:
                error_classes[type(exc).__name__] += 1
                _ui_warn(f"linking failed for {source.id}: {exc.__class__.__name__}: {exc}")
            completed += 1
            if completed % progress_step == 0 or completed == total:
                _ui_step(completed, total, "Linking")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, s, t): s for s, t in pairs}
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    all_decisions.extend(fut.result())
                except Exception as exc:
                    error_classes[type(exc).__name__] += 1
                    _ui_warn(f"linking failed for {src.id}: {exc.__class__.__name__}: {exc}")
                completed += 1
                if completed % progress_step == 0 or completed == total:
                    _ui_step(completed, total, "Linking")

    exception_failed = sum(error_classes.values())
    llm_failed_sources = {d.source for d in all_decisions if d.reason.startswith("LLM call failed")}
    failed = exception_failed + len(llm_failed_sources)
    succeeded = total - failed
    if failed:
        parts = []
        if llm_failed_sources:
            parts.append(f"{len(llm_failed_sources)} LLM failures (fallback applied)")
        if exception_failed:
            breakdown = ", ".join(f"{n} {cls}" for cls, n in error_classes.most_common())
            parts.append(f"{exception_failed} exceptions ({breakdown})")
        _ui_say(f"  Linking: {succeeded}/{total} succeeded, {failed} failed — {', '.join(parts)}")
        if total and failed / total > 0.1:
            _ui_warn(
                f"Linking failure rate {failed}/{total} ({100 * failed // total}%) exceeds 10%"
            )

    stats = LinkStats(
        total=total,
        succeeded=succeeded,
        failed=failed,
        error_classes=error_classes,
    )
    return all_decisions, stats


__all__ = [
    "Entity",
    "LinkDecision",
    "LinkStats",
    "link_entities",
    "link_all",
    "LINK_PROMPT",
]
