# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.transcribe."""

from __future__ import annotations

from pathlib import Path

import pytest

from pengram import transcribe
from pengram.errors import ConfigError


def test_build_domain_prompt_default() -> None:
    prompt = transcribe.build_domain_prompt(None)
    assert "verbatim" in prompt.lower()
    assert "topics" not in prompt.lower()


def test_build_domain_prompt_with_nodes() -> None:
    prompt = transcribe.build_domain_prompt(["Kubernetes", "Observability"])
    assert "Kubernetes" in prompt
    assert "Observability" in prompt


def test_build_domain_prompt_empty_list_falls_back() -> None:
    prompt = transcribe.build_domain_prompt([])
    assert prompt == transcribe.build_domain_prompt(None)


def test_build_domain_prompt_filters_empty_strings() -> None:
    prompt = transcribe.build_domain_prompt(["", "X", ""])
    assert "X" in prompt


def test_transcribe_unknown_mode(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"\x00")
    with pytest.raises(ValueError):
        transcribe.transcribe(media, mode="telepathy")


def test_transcribe_all_skips_existing_transcript(tmp_path: Path) -> None:
    """When <stem>.transcript already exists, Whisper must not run."""
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")
    transcript = tmp_path / "a.transcript"
    transcript.write_text("pre-existing transcript")

    called: list[Path] = []

    def transcriber(p: Path) -> str:
        called.append(p)
        return "should not be called"

    paths = transcribe.transcribe_all([media], transcriber=transcriber)
    assert paths == [transcript]
    assert called == []


def test_transcribe_all_writes_plain_text_transcript(tmp_path: Path) -> None:
    """Whisper output is written as plain text to <stem>.transcript."""
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")

    def transcriber(p: Path) -> str:
        return "fresh transcript"

    paths = transcribe.transcribe_all([media], transcriber=transcriber)
    assert len(paths) == 1
    tp = paths[0]
    assert tp == tmp_path / "a.transcript"
    assert tp.exists()
    assert tp.read_text(encoding="utf-8") == "fresh transcript"


def test_transcribe_all_returns_both_existing_and_new(tmp_path: Path) -> None:
    """Mix of pre-existing and new transcripts returns all paths."""
    m1 = tmp_path / "a.mp3"
    m1.write_bytes(b"\x00")
    m2 = tmp_path / "b.wav"
    m2.write_bytes(b"\x00")
    (tmp_path / "a.transcript").write_text("existing")

    def transcriber(p: Path) -> str:
        return f"transcript for {p.name}"

    paths = transcribe.transcribe_all([m1, m2], transcriber=transcriber)
    assert len(paths) == 2
    assert (tmp_path / "a.transcript") in paths
    assert (tmp_path / "b.transcript") in paths
    assert (tmp_path / "b.transcript").read_text() == "transcript for b.wav"


def test_transcribe_all_delete_transcript_retriggers_whisper(tmp_path: Path) -> None:
    """Deleting the .transcript file must trigger re-transcription."""
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")

    call_count = {"n": 0}

    def transcriber(p: Path) -> str:
        call_count["n"] += 1
        return f"transcript v{call_count['n']}"

    # First run writes the transcript.
    transcribe.transcribe_all([media], transcriber=transcriber)
    assert call_count["n"] == 1
    assert (tmp_path / "a.transcript").read_text() == "transcript v1"

    # Second run skips (transcript exists).
    transcribe.transcribe_all([media], transcriber=transcriber)
    assert call_count["n"] == 1

    # Delete transcript → third run must re-transcribe.
    (tmp_path / "a.transcript").unlink()
    transcribe.transcribe_all([media], transcriber=transcriber)
    assert call_count["n"] == 2
    assert (tmp_path / "a.transcript").read_text() == "transcript v2"


def test_transcribe_all_failure_skips_file(tmp_path: Path, capsys: object) -> None:
    """A Whisper failure must not crash the batch."""
    m1 = tmp_path / "good.mp3"
    m1.write_bytes(b"\x00")
    m2 = tmp_path / "bad.wav"
    m2.write_bytes(b"\x00")

    def transcriber(p: Path) -> str:
        if "bad" in p.name:
            raise RuntimeError("Whisper crashed")
        return "ok"

    paths = transcribe.transcribe_all([m1, m2], transcriber=transcriber)
    assert len(paths) == 1
    assert paths[0] == tmp_path / "good.transcript"


def test_transcribe_local_import_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="faster-whisper"):
        transcribe.transcribe_local(tmp_path / "x.mp3")


def test_transcribe_api_import_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="openai"):
        transcribe.transcribe_api(tmp_path / "x.mp3", api_key="sk-x")


def test_transcribe_api_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Inject a stub openai module so the import succeeds, then drive the
    # missing-key code path.
    import sys
    import types

    stub = types.SimpleNamespace()

    class _Client:
        def __init__(self, **kw):
            pass

    stub.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", stub)
    monkeypatch.setattr("pengram.config.OPENAI_API_KEY", None, raising=False)
    media = tmp_path / "x.mp3"
    media.write_bytes(b"\x00")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        transcribe.transcribe_api(media, api_key=None)


def test_transcribe_api_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class _Resp:
        text = "hello from api"

    class _Transcriptions:
        def create(self, **kw):
            return _Resp()

    class _Audio:
        transcriptions = _Transcriptions()

    class _Client:
        def __init__(self, **kw):
            pass

        audio = _Audio()

    stub = types.SimpleNamespace(OpenAI=_Client)
    monkeypatch.setitem(sys.modules, "openai", stub)
    media = tmp_path / "x.mp3"
    media.write_bytes(b"\x00")
    out = transcribe.transcribe_api(media, api_key="sk-test")
    assert out == "hello from api"


def test_transcribe_dispatch_openai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"ok": False}

    def fake(path, **kw):
        calls["ok"] = True
        return "ok"

    monkeypatch.setattr(transcribe, "transcribe_api", fake)
    media = tmp_path / "x.mp3"
    media.write_bytes(b"\x00")
    out = transcribe.transcribe(media, mode="openai")
    assert out == "ok" and calls["ok"]


def test_transcribe_dispatch_openrouter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake(path, **kw):
        seen.update(kw)
        return "ok"

    monkeypatch.setattr(transcribe, "transcribe_api", fake)
    monkeypatch.setattr("pengram.config.OPENROUTER_API_KEY", "sk-or", raising=False)
    media = tmp_path / "x.mp3"
    media.write_bytes(b"\x00")
    transcribe.transcribe(media, mode="openrouter")
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
