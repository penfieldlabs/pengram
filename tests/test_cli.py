# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for the pengram CLI (pengram.__main__)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pengram.__main__ import main
from pengram.cli import build_parser

# tree-sitter-language-pack is a dev/CI dependency (see pyproject [dev]) so
# CLI integration tests that walk real source files always run.


@pytest.fixture(autouse=True)
def _stub_enrichment_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any CLI test from dispatching a real LLM call during the
    enrichment pass. Tests that want to exercise real enrichment behaviour
    override ``pengram.enrich.call_llm`` with their own fake."""
    import pengram.enrich as enrich_mod

    def _safe_enrich(prompt: str, **kw):
        return json.dumps(
            {
                "definition": "",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    monkeypatch.setattr(enrich_mod, "call_llm", _safe_enrich)


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "pengram" in out.lower()


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0


def test_info_runs(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["info"])
    assert code in (0, 1)
    out = capsys.readouterr().out
    assert "pengram" in out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower() or "pengram" in out.lower()


def test_cache_clear_on_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["cache", "clear", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Removed 0" in out


def test_run_on_empty_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "run", str(tmp_path)])
    assert code == 0


def test_run_on_python_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "hello.py"
    src.write_text("def f(): pass\nclass C: pass\n")
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "run", str(tmp_path)])
    assert code == 0
    assert (out_dir / "graph.json").exists()
    data = json.loads((out_dir / "graph.json").read_text())
    assert data["meta"]["nodes"] >= 3  # file + class + function


def test_run_without_tree_sitter_degrades_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI must exit 0 and warn (not fail) when tree-sitter is unavailable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "tree_sitter_language_pack":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    (tmp_path / "hello.py").write_text("def f(): pass\n")
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "run", str(tmp_path), "--no-llm"])
    assert code == 0  # graceful, not a user error


def test_export_missing_file(tmp_path: Path) -> None:
    code = main(["export", "html", str(tmp_path / "missing.json")])
    assert code == 1


def test_export_roundtrip(tmp_path: Path) -> None:
    graph = {
        "meta": {"nodes": 1, "edges": 0, "communities": 0, "directed": True, "version": "0.1"},
        "nodes": [{"id": "a", "label": "A", "kind": "concept"}],
        "edges": [],
        "communities": {},
        "analysis": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "export", "html", str(graph_path)])
    assert code == 0
    assert (out_dir / "graph.html").exists()


def test_youtube_unknown_channel(tmp_path: Path) -> None:
    code = main(["youtube", "definitely_not_configured"])
    assert code == 1


def test_youtube_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_youtube dispatches to pull_catalog / pull_all_transcripts when
    the channel exists, without making any real network call."""
    from pengram.config import YouTubeChannel
    from pengram.youtube import TranscriptResult, VideoMeta

    channel = YouTubeChannel(url="https://youtube.com/@x", label="X")
    monkeypatch.setattr(
        "pengram.config.YOUTUBE_CHANNELS",
        {"x": channel},
        raising=False,
    )

    vid = "dQw4w9WgXcQ"
    fake_catalog = [
        VideoMeta(
            video_id=vid,
            title="V1",
            url="",
            channel="X",
            tab="videos",
        )
    ]
    calls: dict[str, int] = {"catalog": 0, "transcripts": 0}

    def fake_pull_catalog(ch, client=None):
        calls["catalog"] += 1
        return fake_catalog

    def fake_pull_all(catalog, **kw):
        calls["transcripts"] += 1
        return {vid: TranscriptResult(vid, "hi", "ok")}

    import pengram.youtube as yt_mod

    monkeypatch.setattr(yt_mod, "pull_catalog", fake_pull_catalog)
    monkeypatch.setattr(yt_mod, "pull_all_transcripts", fake_pull_all)

    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "youtube", "x"])
    assert code == 0
    assert calls == {"catalog": 1, "transcripts": 1}


def _write_graph_json(tmp_path: Path) -> Path:
    graph = {
        "meta": {"nodes": 2, "edges": 1, "communities": 1, "directed": True, "version": "0.1"},
        "nodes": [
            {"id": "alice", "label": "Alice", "kind": "concept", "mentions": 3},
            {"id": "bob", "label": "Bob", "kind": "concept", "mentions": 3},
        ],
        "edges": [
            {
                "source": "alice",
                "target": "bob",
                "relation": "references",
                "confidence": "EXTRACTED",
            },
        ],
        "communities": {"0": ["alice", "bob"]},
        "analysis": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph))
    return graph_path


def test_export_penfield_format(tmp_path: Path) -> None:
    graph_path = _write_graph_json(tmp_path)
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "export", "penfield", str(graph_path)])
    assert code == 0
    assert (out_dir / "vault-penfield" / "concepts" / "alice.md").exists()


def test_export_obsidian_format(tmp_path: Path) -> None:
    graph_path = _write_graph_json(tmp_path)
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "export", "obsidian", str(graph_path)])
    assert code == 0
    note = (out_dir / "vault-obsidian" / "concepts" / "alice.md").read_text()
    assert "@references" in note  # inline Relationships section present


def test_export_report_format(tmp_path: Path) -> None:
    graph_path = _write_graph_json(tmp_path)
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "export", "report", str(graph_path)])
    assert code == 0
    report = (out_dir / "GRAPH_REPORT.md").read_text()
    assert "# Graph Report" in report


def test_export_unknown_format_rejected(tmp_path: Path) -> None:
    graph_path = _write_graph_json(tmp_path)
    with pytest.raises(SystemExit):
        # argparse rejects unknown choices with SystemExit(2)
        main(["export", "sqlite", str(graph_path)])


def test_cli_re_enrich_flag_clears_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--re-enrich wipes the enrichment cache before running so every
    concept is re-enriched from scratch."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"
    cache_dir = out_dir / "enrichment_cache"
    cache_dir.mkdir(parents=True)
    # Seed a stale cache file.
    (cache_dir / "stale.json").write_text(
        json.dumps(
            {
                "concept_id": "stale",
                "definition": "old",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )
    )
    assert (cache_dir / "stale.json").exists()

    main(["--output", str(out_dir), "run", str(input_dir), "--re-enrich"])
    # Stale entry was wiped by --re-enrich.
    assert not (cache_dir / "stale.json").exists()


def test_cli_cache_clear_enrichment_subcommand(tmp_path: Path) -> None:
    """pengram cache clear-enrichment <output_dir> deletes the cache."""
    cache_dir = tmp_path / "out" / "enrichment_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "a.json").write_text("{}")
    (cache_dir / "b.json").write_text("{}")
    code = main(["cache", "clear-enrichment", str(tmp_path / "out")])
    assert code == 0
    assert list(cache_dir.glob("*.json")) == []


def test_cache_unknown_action() -> None:
    # The argparse layer enforces choices=("clear",) so unknown actions exit
    # with SystemExit (code 2).
    with pytest.raises(SystemExit):
        main(["cache", "wipe"])


def test_verbose_flag_enables_info_logging(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    out_dir = tmp_path / "out"
    rc = main(["-v", "--output", str(out_dir), "run", str(tmp_path), "--no-llm"])
    assert rc == 0


def test_main_internal_error_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected exceptions in a cmd handler should return exit code 2."""
    from pengram import cli as main_mod

    def boom(_args):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(main_mod, "cmd_info", boom)
    assert main(["info"]) == 2


def test_build_parser_basic() -> None:
    parser = build_parser()
    args = parser.parse_args(["info"])
    assert args.command == "info"


def test_run_no_llm_flag_skips_documents(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("def f(): pass\n")
    (tmp_path / "notes.md").write_text("Some text.\n\n" + "word " * 300)
    out_dir = tmp_path / "out"
    # --no-llm must avoid any LLM call — no network, no claude CLI.
    code = main(["--output", str(out_dir), "run", str(tmp_path), "--no-llm"])
    assert code == 0
    data = json.loads((out_dir / "graph.json").read_text())
    # Document should not appear as a node when LLM is skipped.
    assert all(n.get("kind") != "document" for n in data["nodes"])


def test_default_positional_path_runs_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Use a document file so the test works with OR without tree-sitter.
    import pengram.extract_llm as ex_llm

    monkeypatch.setattr(
        ex_llm,
        "call_llm",
        lambda prompt, **kw: json.dumps(
            {
                "concepts": [],
                "events": [],
                "summary": "ok",
            }
        ),
    )
    (tmp_path / "notes.md").write_text("hello " * 300)
    out_dir = tmp_path / "out"
    # No explicit `run` subcommand — the CLI should detect the existing path
    # and insert it.
    code = main(["--output", str(out_dir), str(tmp_path)])
    assert code == 0
    assert (out_dir / "graph.json").exists()


def test_run_with_mocked_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a fake `call_llm` via the extract_llm module path so no external
    # service is contacted.
    import pengram.extract_llm as ex_llm

    def fake(prompt: str, **kw):
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 3, "note": "note"}],
                "events": [],
                "summary": "A note about graphs.",
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake)

    (tmp_path / "notes.md").write_text("graphs " * 300)
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "run", str(tmp_path)])
    assert code == 0
    data = json.loads((out_dir / "graph.json").read_text())
    kinds = {n.get("kind") for n in data["nodes"]}
    assert "document" in kinds
    assert "concept" in kinds


def _install_pipeline_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    link_relation: str = "supports",
    link_confidence: str = "INFERRED",
) -> None:
    """Wire extraction, linking, and enrichment LLM calls to deterministic fakes."""
    import pengram.enrich as enrich_mod
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def fake_extract(prompt: str, **kw):
        # Two concepts with strong mentions ⇒ the v0.1.1 "≥2 mentions per
        # doc" cap keeps them in the link pool.
        return json.dumps(
            {
                "concepts": [
                    {
                        "name": "graphs",
                        "mentions": 4,
                        "note": "Networks of typed connections between entities.",
                    },
                    {"name": "trees", "mentions": 3, "note": "Acyclic graphs with a root."},
                ],
                "events": [],
                "summary": "The note discusses graphs and trees.",
            }
        )

    def fake_link(prompt: str, **kw):
        # The link prompt asks for at most len(targets) entries; respond with one
        # of the 24 semantic types at the chosen confidence so we can assert on it.
        return json.dumps(
            {
                "links": [
                    {
                        "target_index": 0,
                        "relation": link_relation,
                        "confidence": link_confidence,
                        "reason": "test",
                    },
                ]
            }
        )

    def fake_enrich(prompt: str, **kw):
        # Deterministic, non-merging enrichment — no extra LLM cost, no
        # surprising graph mutations. Tests that want to exercise merge
        # behaviour mock pengram.enrich.call_llm separately.
        return json.dumps(
            {
                "definition": "A test definition.",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)
    monkeypatch.setattr(enrich_mod, "call_llm", fake_enrich)


def test_cli_pipeline_uses_linker_not_hardcoded_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for Bug 1: link.py must be called; typed edges must appear."""
    _install_pipeline_mocks(monkeypatch, link_relation="supports")
    (tmp_path / "notes.md").write_text("Alice Bob graphs " * 300)
    out_dir = tmp_path / "out"
    assert main(["--output", str(out_dir), "run", str(tmp_path)]) == 0
    data = json.loads((out_dir / "graph.json").read_text())
    relations = {e.get("relation") for e in data["edges"]}
    # Semantic types from link.py must now appear, not only "references".
    assert "supports" in relations
    assert relations - {"references"} != set()  # something other than references exists


def test_cli_pipeline_emits_non_extracted_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for Bug 2: INFERRED / AMBIGUOUS confidence must be possible."""
    _install_pipeline_mocks(
        monkeypatch,
        link_relation="supports",
        link_confidence="INFERRED",
    )
    (tmp_path / "notes.md").write_text("Alice Bob graphs " * 300)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(tmp_path)])
    data = json.loads((out_dir / "graph.json").read_text())
    confidences = {e.get("confidence") for e in data["edges"]}
    assert "INFERRED" in confidences


def test_cli_document_labels_are_filenames_not_full_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for Bug 4: document labels must be the basename, not an absolute path."""
    _install_pipeline_mocks(monkeypatch)
    (tmp_path / "notes.md").write_text("Alice Bob graphs " * 300)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(tmp_path)])
    data = json.loads((out_dir / "graph.json").read_text())
    doc_nodes = [n for n in data["nodes"] if n.get("kind") == "document"]
    assert doc_nodes
    for doc in doc_nodes:
        assert doc["label"] == "notes.md"
        assert "/" not in doc["label"]


def test_cli_output_target_both_uses_separate_vault_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for Bug 7: OUTPUT_TARGET=both must not overwrite one vault with the other."""
    _install_pipeline_mocks(monkeypatch)
    monkeypatch.setattr("pengram.config.OUTPUT_TARGET", "both", raising=False)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("Alice Bob graphs " * 300)
    out_dir = tmp_path / "out"
    code = main(["--output", str(out_dir), "run", str(input_dir)])
    assert code == 0
    # Two distinct vault trees.
    assert (out_dir / "vault-penfield").is_dir()
    assert (out_dir / "vault-obsidian").is_dir()
    # Penfield vault must not have inline Relationships; Obsidian must.
    pen_notes = list((out_dir / "vault-penfield").rglob("*.md"))
    obs_notes = list((out_dir / "vault-obsidian").rglob("*.md"))
    assert pen_notes and obs_notes
    pen_with_inline = any("## Relationships" in n.read_text() for n in pen_notes)
    obs_with_inline = any("## Relationships" in n.read_text() for n in obs_notes)
    assert not pen_with_inline
    assert obs_with_inline


def test_cli_content_notes_carry_verbatim_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 (4.1): document body passes through verbatim — not summarised."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_text = "# The Graph Note\n\n" + ("Alice discusses graphs. " * 100)
    (input_dir / "notes.md").write_text(source_text)
    out_dir = tmp_path / "out"
    assert main(["--output", str(out_dir), "run", str(input_dir)]) == 0
    notes_md = next((out_dir / "vault-penfield" / "documents").rglob("*.md"))
    body = notes_md.read_text()
    # The full source text (or a large prefix) must appear verbatim — PENgram
    # does not summarise or truncate.
    assert source_text.rstrip() in body


def test_cli_concept_notes_have_extraction_note_and_appearances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1: concept notes get the LLM's `note` + a Discussed-in list.

    Run with --no-enrich so the extraction note survives — v0.2.0
    enrichment would replace it with a (corpus-aware) definition."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 300)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir), "--no-enrich"])
    graphs_md = (out_dir / "vault-penfield" / "concepts" / "graphs.md").read_text()
    assert "Networks of typed connections between entities." in graphs_md
    assert "## Discussed in" in graphs_md
    doc = next((out_dir / "vault-penfield" / "documents").rglob("*.md"))
    doc_slug = doc.stem
    assert f"[[{doc_slug}]]" in graphs_md


def test_cli_graph_has_no_person_org_prediction_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 entity-purge directive: person/org/prediction no longer exist
    at any layer. Even if a stale-prompt LLM returns them, the pipeline
    ignores them so graph.json stays clean."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def stale_extract(prompt: str, **kw):
        # Worst case: an older LLM run returning all the removed kinds.
        return json.dumps(
            {
                "people": [{"name": "Alice", "mentions": 5}],
                "organizations": [{"name": "OpenAI", "mentions": 5}],
                "predictions": [{"statement": "AGI in 3 years"}],
                "topics": ["ai"],
                "concepts": [{"name": "agents", "mentions": 5, "note": "n"}],
                "events": [],
                "summary": "stale",
            }
        )

    def fake_link(prompt: str, **kw):
        return json.dumps({"links": []})

    monkeypatch.setattr(ex_llm, "call_llm", stale_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("agents " * 300)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    # No vault subdirs for the purged kinds.
    assert not (out_dir / "vault-penfield" / "people").exists()
    assert not (out_dir / "vault-penfield" / "organizations").exists()
    assert not (out_dir / "vault-penfield" / "predictions").exists()

    data = json.loads((out_dir / "graph.json").read_text())
    kinds = {n.get("kind") for n in data["nodes"]}
    # Zero person / organization / prediction nodes in graph.json either.
    for banned in ("person", "organization", "prediction"):
        assert banned not in kinds
    # Concepts still flow through.
    assert "concept" in kinds


def test_cli_drops_ambiguous_default_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 (4.2): linker fallback (AMBIGUOUS + default relation) must not appear."""
    # Linker returns the default relation for concepts ("references") at
    # AMBIGUOUS confidence — exactly the fallback we drop.
    _install_pipeline_mocks(
        monkeypatch,
        link_relation="references",
        link_confidence="AMBIGUOUS",
    )
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 300)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])
    data = json.loads((out_dir / "graph.json").read_text())
    # Only doc→entity references edges should survive; no entity→entity
    # AMBIGUOUS defaults.
    concept_ids = {n["id"] for n in data["nodes"] if n.get("kind") == "concept"}
    entity_entity_ambig = [
        e
        for e in data["edges"]
        if e["source"] in concept_ids
        and e["target"] in concept_ids
        and e.get("confidence") == "AMBIGUOUS"
    ]
    assert entity_entity_ambig == []


def test_cli_linker_respects_mention_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 (4.2): only entities with ≥2 mentions in the same doc get linked."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def fake_extract(prompt: str, **kw):
        # 'trees' has only 1 mention — must be filtered out of link pairs.
        return json.dumps(
            {
                "events": [],
                "concepts": [
                    {"name": "graphs", "mentions": 5, "note": "g"},
                    {"name": "trees", "mentions": 1, "note": "t"},
                    {"name": "forests", "mentions": 4, "note": "f"},
                ],
                "summary": "test",
            }
        )

    link_call_count = {"n": 0}

    def fake_link(prompt: str, **kw):
        link_call_count["n"] += 1
        # Capture what the linker sees for assertion.
        assert "trees" not in prompt.lower() or "| trees" not in prompt
        return json.dumps(
            {
                "links": [
                    {
                        "target_index": 0,
                        "relation": "supports",
                        "confidence": "INFERRED",
                        "reason": "ok",
                    }
                ]
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and forests " * 300)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])
    # The linker was called (graphs/forests strong pair) but 'trees' never
    # appeared in any prompt (single-mention → filtered).
    assert link_call_count["n"] > 0


def test_cli_enrichment_renders_definition_and_quotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.0 (3.2): enriched concepts get a definition lead, ## Key Quotes,
    and ## Cross-Source Notes. The extraction note from _install_pipeline_mocks
    (which differs from the enrichment definition) surfaces under
    ## Original Note so nothing is lost."""
    _install_pipeline_mocks(monkeypatch)
    import pengram.enrich as enrich_mod

    def fake_enrich(prompt: str, **kw):
        return json.dumps(
            {
                "definition": "Graphs are networks of typed edges between nodes.",
                "quotes": [
                    {"text": "A graph is a pair (V, E).", "source": "notes.md"},
                ],
                "cross_source_notes": "Discussed consistently across the corpus.",
                "merge_into": None,
            }
        )

    monkeypatch.setattr(enrich_mod, "call_llm", fake_enrich)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"
    assert main(["--output", str(out_dir), "run", str(input_dir)]) == 0
    md = (out_dir / "vault-penfield" / "concepts" / "graphs.md").read_text()
    # The definition is the lead paragraph.
    assert "networks of typed edges" in md
    # Definition must NOT appear twice — once as lead is enough.
    assert md.count("networks of typed edges") == 1
    # Extraction note (differs from definition) appears under Original Note.
    assert "## Original Note" in md
    assert "Networks of typed connections between entities." in md
    assert "## Key Quotes" in md
    assert "A graph is a pair (V, E)." in md
    assert "## Cross-Source Notes" in md


def test_cli_no_enrich_flag_skips_enrichment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-enrich must not call the enrichment LLM at all."""
    _install_pipeline_mocks(monkeypatch)
    import pengram.enrich as enrich_mod

    called = {"n": 0}

    def fake_enrich(prompt: str, **kw):
        called["n"] += 1
        return json.dumps({"definition": "nope"})

    monkeypatch.setattr(enrich_mod, "call_llm", fake_enrich)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir), "--no-enrich"])
    assert called["n"] == 0


def test_cli_enrichment_merges_surface_form_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.0 (3.2): the LLM can redirect a duplicate concept into its
    canonical counterpart. Graph loses the merged node, vault loses the
    merged note, and outgoing wikilinks point at the survivor."""
    import pengram.enrich as enrich_mod
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def fake_extract(prompt: str, **kw):
        return json.dumps(
            {
                "concepts": [
                    {"name": "AMPK", "mentions": 5, "note": "kinase"},
                    {"name": "AMP kinase", "mentions": 3, "note": "same thing"},
                    {"name": "glucose", "mentions": 4, "note": "sugar"},
                ],
                "events": [],
                "summary": "s",
            }
        )

    def fake_link(prompt: str, **kw):
        return json.dumps({"links": []})

    def fake_enrich(prompt: str, **kw):
        # For "AMP kinase" we return merge_into = "AMPK"; everything else
        # gets a plain enrichment.
        if "AMP kinase" in prompt.split("OTHER CONCEPTS", 1)[0]:
            return json.dumps(
                {
                    "definition": "(will be merged)",
                    "quotes": [],
                    "cross_source_notes": "",
                    "merge_into": "AMPK",
                }
            )
        return json.dumps(
            {
                "definition": "Enzyme sensing cellular energy state.",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)
    monkeypatch.setattr(enrich_mod, "call_llm", fake_enrich)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("AMPK AMP kinase glucose " * 200)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    data = json.loads((out_dir / "graph.json").read_text())
    concept_labels = {n["label"] for n in data["nodes"] if n.get("kind") == "concept"}
    assert "AMPK" in concept_labels
    # "AMP kinase" got merged into AMPK — no longer its own node.
    assert "AMP kinase" not in concept_labels
    # AMPK's vault note exists; the duplicate's doesn't.
    assert (out_dir / "vault-penfield" / "concepts" / "ampk.md").exists()
    assert not (out_dir / "vault-penfield" / "concepts" / "amp-kinase.md").exists()


def test_cli_has_no_threshold_event_flag() -> None:
    """v0.2.0 kill-events: --threshold-event no longer exists."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pengram", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "--threshold-event" not in result.stdout


def test_cli_no_events_vault_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the pipeline never writes a vault/events/ directory."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])
    assert not (out_dir / "vault-penfield" / "events").exists()


def test_cli_extraction_output_has_no_events_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-doc extraction dumps on disk never contain an 'events' key
    for fresh runs. (Legacy cached dumps are folded on read.)"""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])
    dumps = list((out_dir / "extractions").glob("*.json"))
    assert dumps
    for dump in dumps:
        data = json.loads(dump.read_text())
        assert "events" not in data


def test_cli_folds_legacy_events_cache_into_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-v0.2.0 content-hash cache entry that still contains an
    ``events`` array must survive as folded concepts — no cache
    invalidation required."""
    _install_pipeline_mocks(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    doc_path = input_dir / "notes.md"
    doc_path.write_text("graphs and trees " * 200)

    # Pre-seed the content-hash cache with a legacy-shape result. This
    # is what a pre-v0.2.0 run would have left behind.
    from pengram import cache as _cache

    _cache.save_cached(
        input_dir,
        doc_path,
        {
            "concepts": [{"name": "graphs", "mentions": 4, "note": "c"}],
            "events": [
                {"name": "CALERIE trial", "mentions": 3, "note": "study"},
                {"name": "cold exposure", "mentions": 2, "note": "practice"},
            ],
            "summary": "legacy",
            "_doc_id": "notes.md",
            "_source": str(doc_path),
        },
    )

    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    data = json.loads((out_dir / "graph.json").read_text())
    concept_labels = {n["label"] for n in data["nodes"] if n.get("kind") == "concept"}
    # Legacy events now surface as concepts.
    assert "CALERIE trial" in concept_labels
    assert "cold exposure" in concept_labels
    # And no 'event'-kind node anywhere.
    kinds = {n.get("kind") for n in data["nodes"]}
    assert "event" not in kinds


def test_cli_ollama_auto_detects_model_and_chunk_chars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Integration test for the cmd_run ollama auto-detect branch.

    When provider=ollama and no --llm-model is given, cmd_run must call
    ``_resolve_ollama_model()`` once, copy the detected name into every
    LLM role, and set ``args.chunk_chars`` from the returned value.
    """
    import pengram.enrich as enrich_mod
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod
    import pengram.llm as llm_mod
    from pengram import config as _config

    resolved = {"calls": 0}
    seen_models: list[str] = []

    def fake_resolve(*a, **kw):
        resolved["calls"] += 1
        return ("qwen2.5:7b", 24_000)

    monkeypatch.setattr(llm_mod, "_resolve_ollama_model", fake_resolve)
    # Pipeline LLMs are mocked so no real HTTP fires.
    # Capture the model kwarg to verify RunConfig threading.
    monkeypatch.setattr(
        ex_llm,
        "call_llm",
        lambda p, **kw: (
            seen_models.append(kw.get("model", "")),
            json.dumps(
                {
                    "concepts": [{"name": "X", "mentions": 3, "note": "n"}],
                    "events": [],
                    "summary": "s",
                }
            ),
        )[1],
    )
    monkeypatch.setattr(
        link_mod,
        "call_llm",
        lambda p, **kw: (
            seen_models.append(kw.get("model", "")),
            json.dumps({"links": []}),
        )[1],
    )
    monkeypatch.setattr(
        enrich_mod,
        "call_llm",
        lambda p, **kw: (
            seen_models.append(kw.get("model", "")),
            json.dumps(
                {
                    "definition": "",
                    "quotes": [],
                    "cross_source_notes": "",
                    "merge_into": None,
                }
            ),
        )[1],
    )

    monkeypatch.setitem(_config.LLM, "extract_model", "_auto")
    monkeypatch.setitem(_config.LLM, "link_model", "_auto")
    monkeypatch.setitem(_config.LLM, "synth_model", "_auto")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("X " * 500)
    out_dir = tmp_path / "out"

    code = main(
        [
            "--output",
            str(out_dir),
            "run",
            str(input_dir),
            "--llm-provider",
            "ollama",
        ]
    )
    assert code == 0
    assert resolved["calls"] == 1
    # All LLM calls received the detected model via RunConfig.
    assert all(m == "qwen2.5:7b" for m in seen_models), seen_models
    # The announcement landed on stdout.
    out = capsys.readouterr().out
    assert "Ollama: using qwen2.5:7b (chunk_chars=24000)" in out


def test_cli_ollama_explicit_llm_model_skips_auto_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing --llm-model bypasses cmd_run's _resolve_ollama_model call."""
    import pengram.enrich as enrich_mod
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod
    import pengram.llm as llm_mod
    from pengram import config as _config

    monkeypatch.setattr("pengram.config.LLM_PROVIDER", _config.LLM_PROVIDER)

    def fake_resolve(*a, **kw):
        raise AssertionError("must not be called when --llm-model is set")

    monkeypatch.setattr(llm_mod, "_resolve_ollama_model", fake_resolve)
    monkeypatch.setattr(
        ex_llm,
        "call_llm",
        lambda p, **kw: json.dumps(
            {
                "concepts": [],
                "events": [],
                "summary": "s",
            }
        ),
    )
    monkeypatch.setattr(
        link_mod,
        "call_llm",
        lambda p, **kw: json.dumps({"links": []}),
    )
    monkeypatch.setattr(
        enrich_mod,
        "call_llm",
        lambda p, **kw: json.dumps(
            {
                "definition": "",
                "quotes": [],
                "cross_source_notes": "",
                "merge_into": None,
            }
        ),
    )

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("x " * 500)
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "--output",
                str(out_dir),
                "run",
                str(input_dir),
                "--llm-provider",
                "ollama",
                "--llm-model",
                "mistral:7b",
            ]
        )
        == 0
    )


def test_cli_watch_mode_runs_initial_build_and_exits_on_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.0 (3.5): --watch runs the initial build, then waits on the
    observer. Stubbing the observer and short-circuiting the time.sleep
    loop exercises the _watch_loop wiring without a real filesystem
    watcher."""
    _install_pipeline_mocks(monkeypatch)

    class _FakeObserver:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.joined = False
            self._pengram_debouncer = None  # set by watch()

        def schedule(self, handler, path, recursive):
            self.scheduled = (handler, path, recursive)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self):
            self.joined = True

    fake_observer = _FakeObserver()
    import pengram.watch as watch_mod

    def fake_watch(root, rebuild, *, debounce=None, exclude=(), observer_factory=None):
        fake_observer.scheduled = (None, str(root), True)
        fake_observer._pengram_debouncer = None
        return fake_observer

    monkeypatch.setattr(watch_mod, "watch", fake_watch)

    # Short-circuit the sleep loop so the _watch_loop exits immediately.
    import time

    loop_iterations = {"n": 0}
    real_sleep = time.sleep

    def fake_sleep(seconds):
        loop_iterations["n"] += 1
        if loop_iterations["n"] >= 2:
            raise KeyboardInterrupt()
        return real_sleep(0)

    monkeypatch.setattr("time.sleep", fake_sleep)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"

    code = main(["--output", str(out_dir), "run", str(input_dir), "--watch"])
    assert code == 0
    # Initial build produced the graph before watch started.
    assert (out_dir / "graph.json").exists()
    # Observer lifecycle honoured.
    assert fake_observer.started
    assert fake_observer.stopped
    assert fake_observer.joined


def test_cli_frontmatter_source_fields_are_basename_and_relative_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.1: source_file = basename; source_path = relative directory.
    No absolute build-machine paths leak into the frontmatter."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    nested = input_dir / "Experts" / "Nathan Bryan"
    nested.mkdir(parents=True)
    (nested / "book.md").write_text("nitric oxide " * 200)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    md = next((out_dir / "vault-penfield" / "documents").rglob("*.md"))
    text = md.read_text()
    fm_end = text.index("\n---", 4)
    fm = text[:fm_end]
    # source_file: no slashes, no absolute path segments.
    assert "source_file: book.md" in fm
    assert str(tmp_path) not in fm
    # source_path: directory only (no filename).
    assert "source_path: Experts/Nathan Bryan" in fm or "source_path: 'Experts/Nathan Bryan'" in fm
    assert "book.md" not in fm.split("source_path:", 1)[1].split("\n", 1)[0]


def test_cli_pdf_lands_in_documents_not_papers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.0: PDFs are plain documents. No ``vault/papers/`` anywhere."""
    _install_pipeline_mocks(monkeypatch)

    # Real tiny PDF via pypdf so extract_pdf_text returns something.
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # pypdf blank pages have no text; write a markdown alongside so the
    # pipeline has at least one LLM-extractable file.
    (input_dir / "book.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (input_dir / "notes.md").write_text("graphs and trees " * 120)

    out_dir = tmp_path / "out"
    assert main(["--output", str(out_dir), "run", str(input_dir)]) == 0
    assert not (out_dir / "vault-penfield" / "papers").exists()
    # notes.md landed under documents/
    assert any((out_dir / "vault-penfield" / "documents").rglob("*.md"))


def test_cli_no_title_field_or_h1_heading_in_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.0: no ``title:`` frontmatter, no synthetic ``# Title`` body heading."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs and trees " * 120)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    for md in (out_dir / "vault-penfield").rglob("*.md"):
        text = md.read_text()
        # title: must NOT appear in the YAML frontmatter (before closing ---).
        fm_end = text.index("\n---", 4)
        assert "title:" not in text[:fm_end]


def test_cli_preserves_input_hierarchy_under_type_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 (4.5): a ``docs/readme.md`` input lands at
    ``vault/documents/docs/readme.md``."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    docs = input_dir / "docs"
    docs.mkdir()
    (docs / "readme.md").write_text("graphs and trees " * 200)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])
    # Filename is the slug of the basename ('readme.md' → 'readme-md.md');
    # the ``docs/`` parent segment mirrors the input hierarchy.
    assert (out_dir / "vault-penfield" / "documents" / "docs" / "readme-md.md").exists()


def test_cli_emits_single_code_overview_not_ast_per_file(
    tmp_path: Path,
) -> None:
    """v0.1.1 (4.6): one ``code`` overview note, not N file/class/function notes."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.py").write_text("def foo(): pass\nclass Bar: pass\n")
    (input_dir / "b.py").write_text("def baz(): pass\n")
    out_dir = tmp_path / "out"
    assert main(["--output", str(out_dir), "run", str(input_dir), "--no-llm"]) == 0
    code_dir = out_dir / "vault-penfield" / "code"
    # Exactly one code note — the overview — not a note per file/class/function.
    notes = list(code_dir.rglob("*.md"))
    assert len(notes) == 1
    assert "## Languages" in notes[0].read_text()


def test_cli_threshold_concept_flag_overrides_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--threshold-concept overrides the default; 0 is rejected at parse time."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def fake_extract(prompt: str, **kw):
        return json.dumps(
            {
                "events": [],
                "concepts": [
                    {"name": "rare", "mentions": 1, "note": "rare"},
                    {"name": "common", "mentions": 5, "note": "common"},
                ],
                "summary": "x",
            }
        )

    def fake_link(prompt: str, **kw):
        return json.dumps({"links": []})

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("rare and common " * 200)

    # Default threshold (3): 'rare' (1 mention) filtered, 'common' (5) kept.
    out1 = tmp_path / "out1"
    main(["--output", str(out1), "run", str(input_dir)])
    assert (out1 / "vault-penfield" / "concepts" / "common.md").exists()
    assert not (out1 / "vault-penfield" / "concepts" / "rare.md").exists()

    # --threshold-concept 0 is rejected at parse time.
    out2 = tmp_path / "out2"
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--output",
                str(out2),
                "run",
                str(input_dir),
                "--threshold-concept",
                "0",
            ]
        )
    assert exc_info.value.code == 2

    # --threshold-concept 1 keeps everything.
    out3 = tmp_path / "out3"
    main(
        [
            "--output",
            str(out3),
            "run",
            str(input_dir),
            "--threshold-concept",
            "1",
        ]
    )
    assert (out3 / "vault-penfield" / "concepts" / "common.md").exists()
    assert (out3 / "vault-penfield" / "concepts" / "rare.md").exists()


def test_cli_extracts_short_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short docs must NOT be silently dropped. The LLM decides what's
    worth extracting — a 30-word dense note is a valid input."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    llm_calls = {"n": 0}

    def fake_extract(prompt: str, **kw):
        llm_calls["n"] += 1
        return json.dumps(
            {
                "concepts": [{"name": "nitric oxide", "mentions": 3, "note": "NO"}],
                "events": [],
                "summary": "A short dense note about nitric oxide.",
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", lambda p, **kw: json.dumps({"links": []}))

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # 30 words — still a valid extraction input.
    (input_dir / "short.md").write_text(
        "Nitric oxide is a gaseous signalling molecule produced by the "
        "endothelium. It relaxes vascular smooth muscle and lowers blood "
        "pressure. Its depletion accompanies aging and many diseases."
    )
    out_dir = tmp_path / "out"
    assert main(["--output", str(out_dir), "run", str(input_dir)]) == 0

    # The short doc was sent to the LLM and extracted.
    assert llm_calls["n"] == 1
    data = json.loads((out_dir / "graph.json").read_text())
    doc_nodes = [n for n in data["nodes"] if n.get("kind") == "document"]
    assert len(doc_nodes) == 1
    concept_labels = {n["label"] for n in data["nodes"] if n.get("kind") == "concept"}
    assert "nitric oxide" in concept_labels


def test_cli_chunks_oversized_documents_instead_of_skipping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.2.0 (3.3): big docs get chunked, not silently dropped. Each
    chunk is a separate LLM call; entities are unioned across chunks."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    llm_calls = {"n": 0}

    def fake_extract(prompt: str, **kw):
        llm_calls["n"] += 1
        # Every chunk reports the same concept with mentions=1 so we can
        # prove the merge step sums them.
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 1, "note": "n"}],
                "events": [],
                "summary": f"chunk {llm_calls['n']}",
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", lambda p, **kw: json.dumps({"links": []}))

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # 25000 chars of text; chunk size 5000 / overlap 500 → ~6 chunks.
    (input_dir / "huge.md").write_text("x " * 12500)
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "--output",
                str(out_dir),
                "run",
                str(input_dir),
                "--chunk-chars",
                "5000",
                "--no-enrich",
            ]
        )
        == 0
    )

    # Multiple chunks ⇒ multiple extraction calls.
    assert llm_calls["n"] > 1
    data = json.loads((out_dir / "graph.json").read_text())
    graph_node = next(
        n for n in data["nodes"] if n.get("kind") == "concept" and n["label"] == "graphs"
    )
    # mentions summed across chunks.
    assert graph_node["mentions"] == llm_calls["n"]
    # Exactly one document node for the whole file (chunks are an
    # extraction detail, not graph nodes).
    doc_nodes = [n for n in data["nodes"] if n.get("kind") == "document"]
    assert len(doc_nodes) == 1


def test_cli_one_bad_file_does_not_kill_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: previously an LLMError on one doc crashed extract_many
    and blanket-except in cmd_run dropped every other cached / in-flight
    result, leaving 'Nothing to extract'. Now the survivors make it
    through."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod
    from pengram.llm import LLMError

    call_order = {"n": 0}

    def fake_extract(prompt: str, **kw):
        call_order["n"] += 1
        if call_order["n"] == 2:
            raise LLMError("line 31 column 74: missing comma")
        return json.dumps(
            {
                "concepts": [{"name": "graphs", "mentions": 3, "note": "n"}],
                "events": [],
                "summary": "ok",
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", lambda p, **kw: json.dumps({"links": []}))

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i, name in enumerate(("a.md", "b.md", "c.md")):
        (input_dir / name).write_text(f"graphs doc-{name} " * 120)
    out_dir = tmp_path / "out"
    # Force workers=1 so the failure is the second call, deterministic.
    monkeypatch.setitem(
        __import__("pengram.config", fromlist=["LLM"]).LLM,
        "parallel_workers",
        1,
    )
    assert main(["--output", str(out_dir), "run", str(input_dir)]) == 0

    # Pipeline produced output instead of 'Nothing to extract'.
    assert (out_dir / "graph.json").exists()
    data = json.loads((out_dir / "graph.json").read_text())
    doc_nodes = [n for n in data["nodes"] if n.get("kind") == "document"]
    # Two out of three documents extracted despite the middle one failing.
    assert len(doc_nodes) == 2


def test_inject_categories_dedups_identical_labels() -> None:
    """v0.2.0 category-dup bugfix: when the gap-closing pass pulls the
    same top-K concepts into two communities, both communities would
    otherwise produce the SAME label. The second one must fold into
    the first, not spawn a `category_N-2` vault note."""
    import networkx as nx

    from pengram.orchestrate.inject import inject_categories as _inject_categories

    g = nx.DiGraph()
    # Three high-degree concepts every community will agree on.
    for cid, label in (
        ("c_burnout", "Burnout"),
        ("c_cancel", "Cancel Culture"),
        ("c_doc", "Documentary"),
    ):
        g.add_node(cid, kind="concept", label=label, mentions=5)

    # Community 0: one doc + the three concepts themselves clustered together.
    g.add_node("doc_a", kind="document", label="A", source_path="a.md", body="x")
    for c in ("c_burnout", "c_cancel", "c_doc"):
        g.add_edge("doc_a", c, relation="references", confidence="EXTRACTED")

    # Community 1: two different docs. Gap-closing pulls c_burnout /
    # c_cancel / c_doc into this community's member set, giving it the
    # same top-K label as community 0.
    g.add_node("doc_b", kind="document", label="B", source_path="b.md", body="y")
    g.add_node("doc_c", kind="document", label="C", source_path="c.md", body="z")
    for c in ("c_burnout", "c_cancel", "c_doc"):
        g.add_edge("doc_b", c, relation="references", confidence="EXTRACTED")
        g.add_edge("doc_c", c, relation="references", confidence="EXTRACTED")

    communities = {
        0: ["doc_a", "c_burnout", "c_cancel", "c_doc"],
        1: ["doc_b", "doc_c"],
    }
    _inject_categories(g, communities)

    # Exactly one category node with that label.
    cat_nodes = [(nid, data) for nid, data in g.nodes(data=True) if data.get("kind") == "category"]
    labels = [data["label"] for _, data in cat_nodes]
    assert labels.count("Burnout / Cancel Culture / Documentary") == 1
    assert len(cat_nodes) == 1

    # Both docs ended up under the single surviving category.
    cat_id, cat_data = cat_nodes[0]
    members = set(cat_data["members"])
    assert "doc_a" in members
    assert "doc_b" in members
    assert "doc_c" in members
    # parent_of edges from the category to every member.
    parent_of = [
        t for _s, t, d in g.out_edges(cat_id, data=True) if d.get("relation") == "parent_of"
    ]
    for member in ("doc_a", "doc_b", "doc_c"):
        assert member in parent_of


def test_inject_categories_dedups_permuted_labels() -> None:
    """Two communities with the same concept set but different
    degree-ordered labels must still dedupe. Canonical comparison key
    is the alphabetically-sorted concept set."""
    import networkx as nx

    from pengram.orchestrate.inject import inject_categories as _inject_categories

    g = nx.DiGraph()
    # Three concepts used by both communities.
    g.add_node("c_a", kind="concept", label="Alpha", mentions=5)
    g.add_node("c_b", kind="concept", label="Beta", mentions=5)
    g.add_node("c_g", kind="concept", label="Gamma", mentions=5)

    # Community 0: doc_a + doc_b + doc_c (size 5) references Alpha most.
    for i in range(3):
        g.add_node(f"doc0_{i}", kind="document", label=f"A{i}", source_path=f"a{i}.md", body="x")
        # doc0_* references Alpha twice (push its degree in the
        # community's subgraph up), plus Beta and Gamma once.
        g.add_edge(f"doc0_{i}", "c_a", relation="references", confidence="EXTRACTED")
    # Add an extra Alpha-puller to tip the degree order in community 0.
    g.add_node("doc0_extra", kind="document", label="A+", source_path="ax.md", body="x")
    g.add_edge("doc0_extra", "c_a", relation="references", confidence="EXTRACTED")
    for cx in ("c_b", "c_g"):
        g.add_edge("doc0_0", cx, relation="references", confidence="EXTRACTED")

    # Community 1: smaller; concepts land with a different degree order
    # (Beta pulled by more docs in its subgraph).
    for i in range(2):
        g.add_node(f"doc1_{i}", kind="document", label=f"B{i}", source_path=f"b{i}.md", body="y")
        g.add_edge(f"doc1_{i}", "c_b", relation="references", confidence="EXTRACTED")
    g.add_edge("doc1_0", "c_a", relation="references", confidence="EXTRACTED")
    g.add_edge("doc1_0", "c_g", relation="references", confidence="EXTRACTED")

    communities = {
        0: ["doc0_0", "doc0_1", "doc0_2", "doc0_extra", "c_a"],
        1: ["doc1_0", "doc1_1", "c_b"],
    }
    _inject_categories(g, communities)

    cat_nodes = [(nid, data) for nid, data in g.nodes(data=True) if data.get("kind") == "category"]
    # Both communities share canonical key ("Alpha", "Beta", "Gamma") →
    # exactly one category node survives.
    assert len(cat_nodes) == 1
    _, cat_data = cat_nodes[0]
    # Every member doc from both communities folded in.
    members = set(cat_data["members"])
    for m in ("doc0_0", "doc0_1", "doc0_2", "doc0_extra", "doc1_0", "doc1_1"):
        assert m in members


def test_inject_categories_bigger_cluster_wins_cat_id() -> None:
    """When two communities dedupe, the one with more members claims
    the surviving cat_id. Reads more sensibly in logs than
    first-come-first-served by raw cid iteration order."""
    import networkx as nx

    from pengram.orchestrate.inject import inject_categories as _inject_categories

    g = nx.DiGraph()
    for nid, label in (("c_x", "X"), ("c_y", "Y"), ("c_z", "Z")):
        g.add_node(nid, kind="concept", label=label, mentions=5)

    # Community 7: 5 docs, referencing the same three concepts.
    for i in range(5):
        g.add_node(f"big_{i}", kind="document", label=f"big{i}", source_path=f"big{i}.md", body="x")
        for c in ("c_x", "c_y", "c_z"):
            g.add_edge(f"big_{i}", c, relation="references", confidence="EXTRACTED")

    # Community 2: 2 docs, referencing the same three concepts.
    for i in range(2):
        g.add_node(f"small_{i}", kind="document", label=f"sm{i}", source_path=f"sm{i}.md", body="y")
        for c in ("c_x", "c_y", "c_z"):
            g.add_edge(f"small_{i}", c, relation="references", confidence="EXTRACTED")

    communities = {
        2: ["small_0", "small_1"],
        7: ["big_0", "big_1", "big_2", "big_3", "big_4"],
    }
    _inject_categories(g, communities)

    cat_ids = [nid for nid, data in g.nodes(data=True) if data.get("kind") == "category"]
    assert len(cat_ids) == 1
    # Bigger community (cid=7) wins the cat_id.
    assert cat_ids[0] == "category_7"


def test_cli_category_parents_concepts_from_member_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 directive: every category must parent_of every concept
    referenced by its member documents, even when clustering splits the
    concept out into a different cluster."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def fake_extract(prompt: str, **kw):
        return json.dumps(
            {
                "events": [],
                "concepts": [
                    {"name": "geopolitics", "mentions": 5, "note": "g"},
                    {"name": "brics", "mentions": 4, "note": "b"},
                    {"name": "dedollarization", "mentions": 3, "note": "d"},
                ],
                "summary": "A note about geopolitics and brics.",
            }
        )

    def fake_link(prompt: str, **kw):
        return json.dumps({"links": []})

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("geopolitics brics dedollarization " * 100)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    data = json.loads((out_dir / "graph.json").read_text())
    category_ids = {n["id"] for n in data["nodes"] if n.get("kind") == "category"}
    concept_ids = {n["id"] for n in data["nodes"] if n.get("kind") == "concept"}
    assert category_ids and concept_ids

    parent_of_by_category: dict[str, set[str]] = {}
    for edge in data["edges"]:
        if edge.get("relation") == "parent_of" and edge["source"] in category_ids:
            parent_of_by_category.setdefault(edge["source"], set()).add(edge["target"])

    # Every concept from the document must be parented by at least one category.
    concept_parents: set[str] = set()
    for targets in parent_of_by_category.values():
        concept_parents |= targets & concept_ids
    assert concept_parents == concept_ids, (
        "every concept referenced by a member doc must be parented by its category"
    )


def test_cli_emits_category_notes_from_clusters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.1.1 (4.3): clusters with concept nodes produce category hub notes."""
    import pengram.extract_llm as ex_llm
    import pengram.link as link_mod

    def fake_extract(prompt: str, **kw):
        return json.dumps(
            {
                "events": [],
                "concepts": [
                    {"name": "graphs", "mentions": 5, "note": "networks"},
                    {"name": "trees", "mentions": 4, "note": "acyclic graphs"},
                    {"name": "forests", "mentions": 3, "note": "collections of trees"},
                ],
                "summary": "graph theory basics",
            }
        )

    def fake_link(prompt: str, **kw):
        return json.dumps(
            {
                "links": [
                    {
                        "target_index": 0,
                        "relation": "sibling_of",
                        "confidence": "INFERRED",
                        "reason": "related",
                    },
                ]
            }
        )

    monkeypatch.setattr(ex_llm, "call_llm", fake_extract)
    monkeypatch.setattr(link_mod, "call_llm", fake_link)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("graphs trees forests " * 100)
    out_dir = tmp_path / "out"
    main(["--output", str(out_dir), "run", str(input_dir)])

    # At least one category note must exist.
    categories_dir = out_dir / "vault-penfield" / "categories"
    assert categories_dir.exists()
    cat_notes = list(categories_dir.glob("*.md"))
    assert cat_notes
    body = cat_notes[0].read_text()
    assert "## Members" in body
    # Category label is derived from top-degree concept labels.
    assert "graphs" in body.lower() or "trees" in body.lower()
    # Category node also appears in graph.json.
    data = json.loads((out_dir / "graph.json").read_text())
    kinds = {n.get("kind") for n in data["nodes"]}
    assert "category" in kinds
    # parent_of edges from category → members.
    parent_of = [e for e in data["edges"] if e.get("relation") == "parent_of"]
    assert parent_of


def test_cli_pipeline_is_idempotent_across_output_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for Bug 3: two runs on the same input produce the same graph."""
    _install_pipeline_mocks(monkeypatch)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.md").write_text("Alice Bob graphs " * 300)

    # Output dirs sit outside the input tree so neither walk picks up the
    # other run's artifacts.
    out1 = tmp_path / "outputs" / "run1"
    out2 = tmp_path / "outputs" / "run2"
    main(["--output", str(out1), "run", str(input_dir)])
    main(["--output", str(out2), "run", str(input_dir)])

    g1 = json.loads((out1 / "graph.json").read_text())
    g2 = json.loads((out2 / "graph.json").read_text())

    ids1 = sorted(n["id"] for n in g1["nodes"])
    ids2 = sorted(n["id"] for n in g2["nodes"])
    edges1 = sorted((e["source"], e["target"], e["relation"]) for e in g1["edges"])
    edges2 = sorted((e["source"], e["target"], e["relation"]) for e in g2["edges"])
    assert ids1 == ids2
    assert edges1 == edges2
