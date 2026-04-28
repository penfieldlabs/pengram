# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""CLI parser and command dispatchers for PENgram.

Each ``cmd_*`` function handles one subcommand and returns an exit code.
The parser is built by :func:`build_parser`; the entry-point is
:func:`main` (which lives in ``__main__.py`` and delegates here).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__, cache, extract_ast, extract_llm
from . import config as _config
from . import detect as _detect
from . import enrich as _enrich
from . import report as _report
from ._ui import error as _ui_error
from ._ui import say, warn
from .analyze import analyze
from .build import build
from .cluster import cluster
from .export_catalog import export_catalog
from .export_html import export_html
from .export_json import export_json
from .export_obsidian import export_obsidian
from .export_penfield import export_penfield
from .extract_llm import Document
from .llm import LLMError
from .orchestrate.inject import inject_categories, inject_code_overview
from .pipeline import _ENTITY_KINDS, RunConfig, build_semantic_extraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(message: str, code: int = 1) -> int:
    _ui_error(message)
    return code


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_transcript_meta(path: Path) -> dict[str, Any]:
    """Parse YAML frontmatter from a transcript file for metadata propagation."""
    from .youtube import parse_transcript_frontmatter

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    fm = parse_transcript_frontmatter(text)
    na_keys = [k for k, v in fm.items() if v == "NA"]
    for k in na_keys:
        del fm[k]
    return fm


def _read_document_text(path: Path, file_type: _detect.FileType) -> str:
    """Return plain text for a document or transcript file.

    Any leading YAML frontmatter is stripped so the LLM only sees
    content.  PDF and ePub are delegated to the format-specific
    extractors in :mod:`pengram.detect`.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _detect.extract_pdf_text(path)
    if suffix == ".epub":
        return _detect.extract_epub_text(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        warn(f"could not read {path}: {exc}")
        return ""
    if text.startswith("---\n"):
        from .youtube import strip_frontmatter

        text = strip_frontmatter(text)
    return text


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full PENgram pipeline on a directory."""
    root = Path(args.path).resolve()
    if not root.exists():
        return _error(f"{root} does not exist")
    output_dir = _ensure_output_dir(Path(args.output or _config.OUTPUT_DIR))
    use_llm = not args.no_llm

    rc = RunConfig.from_args(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        chunk_chars=args.chunk_chars,
    )
    if rc.provider == "ollama" and args.llm_model is None:
        try:
            from .llm import _resolve_ollama_model

            detected, ctx_chunk_chars = _resolve_ollama_model()
            rc = RunConfig(
                provider=rc.provider,
                extract_model=detected if rc.extract_model == "_auto" else rc.extract_model,
                link_model=detected if rc.link_model == "_auto" else rc.link_model,
                synth_model=detected if rc.synth_model == "_auto" else rc.synth_model,
                extract_timeout=rc.extract_timeout,
                link_timeout=rc.link_timeout,
                synth_timeout=rc.synth_timeout,
                chunk_chars=rc.chunk_chars or ctx_chunk_chars,
            )
            say(f"  Ollama: using {detected} (chunk_chars={rc.chunk_chars})")
        except LLMError as exc:
            return _error(str(exc))

    say(f"Scanning {root} ...")
    groups = _detect.collect_files(root, exclude=[output_dir])
    total = sum(len(files) for files in groups.values())
    say(f"  {total} files classified across {len(groups)} types")

    extractions: list[dict[str, Any]] = []

    code_files = groups.get(_detect.FileType.CODE, [])
    if code_files:
        say(f"  AST extraction: {len(code_files)} code files")
    ast_import_error: str | None = None
    for code_path in code_files:
        if ast_import_error is not None:
            break
        cached = cache.load_cached(root, code_path)
        if cached is not None:
            extractions.append(cached)
            continue
        try:
            result = extract_ast.extract_code(code_path)
        except ImportError as exc:
            ast_import_error = str(exc)
            warn(f"AST extraction skipped: {exc}")
            break
        cache.save_cached(root, code_path, result)
        extractions.append(result)

    doc_like: list[tuple[Path, _detect.FileType]] = []
    for ftype in (_detect.FileType.DOCUMENT, _detect.FileType.TRANSCRIPT):
        for p in groups.get(ftype, []):
            doc_like.append((p, ftype))

    if doc_like and use_llm:
        say(f"  LLM extraction: {len(doc_like)} documents")
        documents: list[Document] = []
        for path, ftype in doc_like:
            text = _read_document_text(path, ftype)
            if not text.strip():
                continue
            meta: dict[str, Any] = {"file_type": ftype.value}
            if ftype == _detect.FileType.TRANSCRIPT:
                meta.update(_parse_transcript_meta(path))
            documents.append(
                Document(
                    doc_id=str(path.relative_to(root)),
                    text=text,
                    source=str(path),
                    metadata=meta,
                )
            )
        if documents:
            try:
                llm_results = extract_llm.extract_many(
                    documents,
                    output_dir=output_dir,
                    cache_root=root,
                    model=rc.extract_model,
                    timeout=rc.extract_timeout,
                    chunk_chars=rc.chunk_chars,
                    provider=rc.provider,
                )
            except ImportError as exc:
                warn(f"LLM extraction skipped: {exc}")
                llm_results = []
            if llm_results:
                say(
                    f"  Canonicalize + link: {sum(len(r.get(k, [])) for r in llm_results for k, _ in _ENTITY_KINDS)} raw entities"
                )
                extractions.append(
                    build_semantic_extraction(
                        llm_results,
                        output_dir=output_dir,
                        run_linker=True,
                        documents=documents,
                        link_model=rc.link_model,
                        link_timeout=rc.link_timeout,
                        provider=rc.provider,
                    )
                )
    elif doc_like and not use_llm:
        say(f"  LLM extraction skipped (--no-llm). {len(doc_like)} documents ignored.")

    media_count = (
        len(groups.get(_detect.FileType.AUDIO, []))
        + len(groups.get(_detect.FileType.VIDEO, []))
        + len(groups.get(_detect.FileType.IMAGE, []))
    )
    if media_count:
        say(f"  {media_count} audio/video/image files detected (extraction in v0.2+).")

    if not extractions:
        say("Nothing to extract. Exiting cleanly.")
        return 0

    say(f"  {len(extractions)} extractions; building graph...")
    g = build(extractions)
    communities = cluster(g)

    enrichment_cache_dir = output_dir / "enrichment_cache"
    if use_llm and args.re_enrich:
        removed = _enrich.clear_cache(enrichment_cache_dir)
        if removed:
            say(f"  --re-enrich: cleared {removed} cached enrichment entries")
    if use_llm and not args.no_enrich:
        concept_count = sum(1 for n in g.nodes if g.nodes[n].get("kind") == "concept")
        if concept_count:
            say(f"  Enrichment: {concept_count} concepts (use --no-enrich to skip)")
            try:
                enrich_results = _enrich.enrich_concepts(
                    g,
                    communities,
                    model=args.enrich_model or rc.synth_model,
                    cache_dir=enrichment_cache_dir,
                    provider=rc.provider,
                )
                merges, enriched = _enrich.apply_enrichment(g, enrich_results)
                if merges or enriched:
                    say(
                        f"  Enrichment merged {merges} duplicate concept(s); "
                        f"enriched {enriched} surviving concept(s)."
                    )
                if merges:
                    alive = set(g.nodes)
                    communities = {
                        cid: [m for m in members if m in alive]
                        for cid, members in communities.items()
                    }
            except ImportError as exc:
                warn(f"Enrichment skipped: {exc}")

    inject_code_overview(g, code_files, root)
    inject_categories(g, communities)

    analysis = analyze(g, communities)

    export_json(g, output_dir, communities=communities, analysis=analysis)
    export_html(g, output_dir, communities=communities)
    (output_dir / "GRAPH_REPORT.md").write_text(
        _report.render_report(g, analysis, communities),
        encoding="utf-8",
    )

    from .export_penfield import DEFAULT_THRESHOLDS as _DEFAULT_VAULT_THRESHOLDS

    thresholds: dict[str, int] = dict(_DEFAULT_VAULT_THRESHOLDS)
    if args.threshold_concept is not None:
        thresholds["concept"] = args.threshold_concept
    if args.threshold_category is not None:
        thresholds["category"] = args.threshold_category

    target = _config.OUTPUT_TARGET
    if target in ("penfield", "both"):
        export_penfield(
            g,
            output_dir,
            vault_name="vault-penfield",
            thresholds=thresholds,
            min_confidence=args.min_confidence,
        )
    if target in ("obsidian", "both"):
        export_obsidian(
            g,
            output_dir,
            vault_name="vault-obsidian",
            thresholds=thresholds,
            min_confidence=args.min_confidence,
        )
    if args.export_catalog:
        export_catalog(g, output_dir, communities=communities)
    say(f"Done. Output written to {output_dir}")

    if args.watch:
        return _watch_loop(root, output_dir, args)
    return 0


def _watch_loop(root: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Start the file watcher and keep rebuilding until Ctrl+C."""
    try:
        from . import watch as watch_mod
    except ImportError as exc:
        return _error(str(exc))

    say(f"Watching {root} (Ctrl+C to stop)...")

    def _on_change(changed: set[Path]) -> None:
        say(f"  Changes: {len(changed)} file(s); rebuilding...")
        rebuild_args = argparse.Namespace(**vars(args))
        rebuild_args.watch = False
        try:
            cmd_run(rebuild_args)
        except Exception as exc:  # pragma: no cover — reported to user
            warn(f"rebuild failed: {exc}")

    try:
        observer = watch_mod.watch(
            root,
            _on_change,
            exclude=[output_dir],
        )
    except ImportError as exc:
        return _error(str(exc))

    observer.start()
    try:
        while True:
            try:
                import time

                time.sleep(1)
            except KeyboardInterrupt:
                break
    finally:
        observer.stop()
        observer.join()
        debouncer = getattr(observer, "_pengram_debouncer", None)
        if debouncer is not None:
            debouncer.stop()
    return 0


def cmd_youtube(args: argparse.Namespace) -> int:
    """Pull a YouTube channel's catalog and transcripts."""
    channel_key = args.channel
    channel = _config.YOUTUBE_CHANNELS.get(channel_key)
    if channel is None:
        return _error(
            f"YouTube channel {channel_key!r} is not configured. "
            f"Add it to YOUTUBE_CHANNELS in pengram/config.py."
        )
    try:
        from .youtube import pull_all_transcripts, pull_catalog
    except ImportError as exc:
        return _error(str(exc))
    output_dir = _ensure_output_dir(Path(args.output or _config.OUTPUT_DIR))
    say(f"Pulling catalog for {channel.label} ({channel.url})...")
    catalog = pull_catalog(channel)
    say(f"  {len(catalog)} videos")
    state_file = output_dir / "transcripts" / f"{channel_key}.state.json"
    max_videos: int = args.max_videos
    pull_all_transcripts(
        catalog,
        work_dir=output_dir / "transcripts",
        state_file=state_file,
        sleep_between=2.0,
        max_videos=max_videos,
    )
    say(f"Transcripts written under {output_dir / 'transcripts'}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Re-export an existing ``graph.json`` to another format."""
    graph_path = Path(args.path)
    if not graph_path.exists():
        return _error(f"{graph_path} does not exist")
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    import networkx as nx

    graph_cls = nx.DiGraph if data["meta"].get("directed", True) else nx.Graph
    g: nx.Graph = graph_cls()
    for node in data["nodes"]:
        attrs = {k: v for k, v in node.items() if k != "id"}
        g.add_node(node["id"], **attrs)
    for edge in data["edges"]:
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        g.add_edge(edge["source"], edge["target"], **attrs)
    communities_raw = data.get("communities", {})
    communities = {int(cid): members for cid, members in communities_raw.items()}
    output_dir = _ensure_output_dir(Path(args.output or _config.OUTPUT_DIR))

    fmt = args.format
    if fmt == "penfield":
        export_penfield(g, output_dir)
    elif fmt == "obsidian":
        export_obsidian(g, output_dir)
    elif fmt == "html":
        export_html(g, output_dir, communities=communities)
    elif fmt == "report":
        analysis = data.get("analysis") or analyze(g, communities)
        (output_dir / "GRAPH_REPORT.md").write_text(
            _report.render_report(g, analysis, communities),
            encoding="utf-8",
        )
    else:
        return _error(f"Unknown format: {fmt}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """Maintenance commands for the extraction + enrichment caches."""
    if args.action == "clear":
        root = Path(args.path or ".")
        removed = cache.clear_cache(root)
        say(f"Removed {removed} extraction-cache entries under {root}")
        return 0
    if args.action == "clear-enrichment":
        target = Path(args.path or _config.OUTPUT_DIR) / "enrichment_cache"
        removed = _enrich.clear_cache(target)
        say(f"Removed {removed} enrichment-cache entries under {target}")
        return 0
    return _error(f"Unknown cache action: {args.action}")


def cmd_info(_args: argparse.Namespace) -> int:
    """Print the active configuration and validate it."""
    say(f"pengram {__version__}")
    say(f"  project:     {_config.PROJECT_LABEL} ({_config.PROJECT_NAME})")
    say(f"  output:      {_config.OUTPUT_TARGET}")
    say(f"  whisper:     {_config.WHISPER_MODE} ({_config.WHISPER_MODEL})")
    say(f"  llm:         {_config.LLM_PROVIDER}")
    say(f"  base_dir:    {_config.BASE_DIR}")
    say(f"  output_dir:  {_config.OUTPUT_DIR}")
    errors = _config.validate()
    if errors:
        say("Config errors:")
        for e in errors:
            say(f"  - {e}")
        return 1
    say("Config OK.")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1, got {n} (a concept with 0 mentions is nothing)"
        )
    return n


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="pengram",
        description="PENgram — parse, extract, and normalize a knowledge graph.",
    )
    parser.add_argument("--version", action="version", version=f"pengram {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    parser.add_argument("-o", "--output", help="output directory", default=None)

    subs = parser.add_subparsers(dest="command")

    run = subs.add_parser("run", help="run the full pipeline on a directory")
    run.add_argument("path", help="directory to process")
    run.add_argument(
        "--no-llm", action="store_true", help="skip LLM extraction (AST + deterministic only)"
    )
    run.add_argument(
        "--llm-provider",
        default=None,
        metavar="P",
        choices=sorted(_config.VALID_LLM_PROVIDERS),
        help="LLM provider for extraction, linking, and enrichment "
        "(default: claude-cli; also: openai, openrouter, ollama). "
        "Env: PENGRAM_LLM_PROVIDER.",
    )
    run.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help="override the LLM model used for extraction and linking "
        "(defaults to the provider's extract_model). "
        "Env: PENGRAM_EXTRACT_MODEL / PENGRAM_LINK_MODEL.",
    )
    run.add_argument(
        "--threshold-concept",
        type=_positive_int,
        default=None,
        metavar="N",
        help="minimum mentions for a concept to get a vault note (default 3; minimum 1)",
    )
    run.add_argument(
        "--threshold-category",
        type=_positive_int,
        default=None,
        metavar="N",
        help="minimum members for a category to get a vault note "
        "(default: no filter; pass 2 to drop singleton categories)",
    )
    run.add_argument(
        "--no-enrich",
        action="store_true",
        help="skip the enrichment pass (definitions, quotes, concept dedup)",
    )
    run.add_argument(
        "--re-enrich",
        action="store_true",
        help="delete the enrichment cache before running so every "
        "concept is re-enriched from scratch (use after major "
        "corpus changes)",
    )
    run.add_argument(
        "--chunk-chars",
        type=int,
        default=None,
        metavar="N",
        help="chunk size for oversized documents "
        "(default: 95000 for cloud providers, 20000 for ollama). "
        "Env: PENGRAM_CHUNK_CHARS.",
    )
    run.add_argument(
        "--watch",
        action="store_true",
        help="after the initial build, watch the input tree and rebuild "
        "on file changes (2s debounce). Ctrl+C to stop.",
    )
    run.add_argument(
        "--export-catalog",
        action="store_true",
        help="also write catalog.csv to the output dir — one row per "
        "source file with entity / relationship counts. YouTube "
        "columns are included only when YouTube metadata is "
        "present on any node.",
    )
    run.add_argument(
        "--min-confidence",
        choices=("INFERRED", "EXTRACTED"),
        default=None,
        help="drop vault wikilinks whose edge confidence is below this "
        "level. INFERRED keeps deduced edges; EXTRACTED keeps only "
        "edges stated in source text. AMBIGUOUS is always dropped "
        "upstream (since v0.1.1).",
    )
    run.add_argument(
        "--enrich-model",
        default=None,
        metavar="MODEL",
        help="override the LLM model used for enrichment (defaults to LLM.synth_model)",
    )
    run.set_defaults(func=cmd_run)

    yt = subs.add_parser("youtube", help="pull a configured YouTube channel")
    yt.add_argument("channel", help="channel key from config.YOUTUBE_CHANNELS")
    yt.add_argument(
        "--max-videos",
        type=int,
        default=50,
        help="max new videos to fetch per run (default: 50)",
    )
    yt.set_defaults(func=cmd_youtube)

    ex = subs.add_parser("export", help="re-export an existing graph.json")
    ex.add_argument("format", choices=("penfield", "obsidian", "html", "report"))
    ex.add_argument("path", help="path to graph.json")
    ex.set_defaults(func=cmd_export)

    ca = subs.add_parser("cache", help="cache maintenance")
    ca.add_argument("action", choices=("clear", "clear-enrichment"))
    ca.add_argument("path", nargs="?", default=".")
    ca.set_defaults(func=cmd_cache)

    info = subs.add_parser("info", help="show config and stats")
    info.set_defaults(func=cmd_info)

    return parser


_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "run",
        "youtube",
        "export",
        "cache",
        "info",
    }
)


def _rewrite_default_run(argv: list[str]) -> list[str]:
    """Allow ``pengram <path>`` as a shortcut for ``pengram run <path>``."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-v", "--verbose"):
            i += 1
            continue
        if token in ("-o", "--output"):
            i += 2
            continue
        if token.startswith("--output="):
            i += 1
            continue
        break
    if i >= len(argv):
        return argv
    first = argv[i]
    if first in _SUBCOMMANDS or first.startswith("-"):
        return argv
    if not Path(first).exists():
        return argv
    return argv[:i] + ["run"] + argv[i:]


__all__ = [
    "build_parser",
    "cmd_cache",
    "cmd_export",
    "cmd_info",
    "cmd_run",
    "cmd_youtube",
    "_rewrite_default_run",
]
