# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Graph mutation passes — synthetic nodes injected after clustering."""

from __future__ import annotations

from pathlib import Path

from ..security import slugify
from ..vocabulary import CONFIDENCE_EXTRACTED

_CATEGORY_MIN_MEMBERS = 2
_CATEGORY_LABEL_TOP_K = 3
_CATEGORY_MEMBER_KINDS: frozenset[str] = frozenset(
    {
        "concept",
        "document",
        "transcript",
    }
)
_CATEGORY_CONTENT_KINDS: frozenset[str] = frozenset(
    {
        "document",
        "transcript",
    }
)


def inject_code_overview(g, code_files: list[Path], root: Path) -> None:
    """Add one synthetic ``code`` overview note summarising the repo.

    Individual AST nodes (``file``/``class``/``function``) stay in the
    graph for the HTML visualiser but do not land as vault notes — they
    were noise in v0.1.0's vault (37 tiny notes for a small repo). The
    overview collapses that into a single note with a language
    breakdown and the file count.
    """
    if not code_files:
        return
    by_language: dict[str, int] = {}
    for node_id, data in list(g.nodes(data=True)):
        if data.get("kind") == "file":
            lang = str(data.get("language") or "unknown")
            by_language[lang] = by_language.get(lang, 0) + 1
    if not by_language:
        return
    lines = ["## Languages", ""]
    for lang, count in sorted(by_language.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{lang}** — {count} file{'s' if count != 1 else ''}")
    lines.append("")
    lines.append(f"## Files ({sum(by_language.values())})")
    lines.append("")
    for code_path in sorted(code_files):
        try:
            rel = code_path.relative_to(root)
        except ValueError:
            rel = code_path
        lines.append(f"- `{rel}`")

    g.add_node(
        "code_overview",
        label=f"{root.name or 'code'} (code)",
        kind="code",
        body="\n".join(lines),
        confidence=CONFIDENCE_EXTRACTED,
        mentions=sum(by_language.values()),
    )


def inject_categories(g, communities: dict[int, list[str]]) -> None:
    """Add synthetic ``category`` nodes that summarise each cluster.

    For each cluster with >=2 members:

    1. Collect the vault-eligible cluster members (content + concepts +
       events that happened to cluster together).
    2. Close the concept gap: walk every content member's outgoing
       ``references`` edges and add each referenced concept to the
       category's member set — even if that concept ended up in a
       different cluster. Without this step clusters that split
       concepts away from the documents discussing them leave the
       category without its own concept hub.
    3. Label the category from the top-``_CATEGORY_LABEL_TOP_K``
       highest-degree concepts in the combined set.
    4. Emit ``parent_of`` edges from the category to every final member.

    Mutates the graph in place.

    Duplicate-label dedup: any two communities whose top-K concept set
    is the same (regardless of the degree-ordering within each
    community) fold into a single category node. The comparison key is
    the alphabetically-sorted concept set so permuted labels still
    match ("Burnout / Cancel Culture" and "Cancel Culture / Burnout"
    dedupe). The display label keeps the winner's degree order so
    readers still see the most-connected concept first.

    Winner selection: when multiple communities share the same
    canonical key, the community with the most members wins (tie-break
    on smallest ``cid`` for stability). Folding into the bigger
    cluster produces a ``cat_id`` that mirrors the more representative
    community.
    """
    ordered = sorted(
        communities.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )

    key_to_catid: dict[tuple[str, ...], str] = {}

    for cid, members in ordered:
        if len(members) < _CATEGORY_MIN_MEMBERS:
            continue

        vault_members: list[str] = [
            m for m in members if g.nodes[m].get("kind") in _CATEGORY_MEMBER_KINDS
        ]
        if not vault_members:
            continue

        content_members = [
            m for m in vault_members if g.nodes[m].get("kind") in _CATEGORY_CONTENT_KINDS
        ]
        seen: set[str] = set(vault_members)
        extra_concepts: list[str] = []
        for content_id in content_members:
            for _src, tgt, data in g.out_edges(content_id, data=True):
                if data.get("relation") != "references":
                    continue
                if tgt in seen:
                    continue
                if g.nodes[tgt].get("kind") == "concept":
                    seen.add(tgt)
                    extra_concepts.append(tgt)
        vault_members.extend(extra_concepts)

        concepts_by_degree = sorted(
            (m for m in vault_members if g.nodes[m].get("kind") == "concept"),
            key=lambda n: (-g.degree(n), n),
        )
        if not concepts_by_degree:
            continue
        label_parts = [
            str(g.nodes[n].get("label", n)) for n in concepts_by_degree[:_CATEGORY_LABEL_TOP_K]
        ]
        label_parts = [p for p in label_parts if p]
        if not label_parts:
            continue
        label = " / ".join(label_parts)
        canonical_key = tuple(sorted(label_parts))

        existing_id = key_to_catid.get(canonical_key)
        if existing_id is not None:
            existing_members = g.nodes[existing_id].get("members") or []
            existing_set = set(existing_members)
            new_members = [m for m in vault_members if m not in existing_set]
            merged = list(existing_members) + new_members
            g.nodes[existing_id]["members"] = merged
            g.nodes[existing_id]["mentions"] = len(merged)
            for member in new_members:
                g.add_edge(
                    existing_id,
                    member,
                    relation="parent_of",
                    confidence=CONFIDENCE_EXTRACTED,
                )
            continue

        cat_id = f"category_{slugify('-'.join(canonical_key))}"
        g.add_node(
            cat_id,
            label=label,
            kind="category",
            mentions=len(vault_members),
            members=vault_members,
            confidence=CONFIDENCE_EXTRACTED,
        )
        for member in vault_members:
            g.add_edge(
                cat_id,
                member,
                relation="parent_of",
                confidence=CONFIDENCE_EXTRACTED,
            )
        key_to_catid[canonical_key] = cat_id


__all__ = [
    "inject_code_overview",
    "inject_categories",
]
