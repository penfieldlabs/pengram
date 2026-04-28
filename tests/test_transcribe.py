# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.transcribe."""

from __future__ import annotations

import json
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


def test_transcribe_all_uses_sidecar(tmp_path: Path) -> None:
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")
    sidecar = media.with_suffix(media.suffix + ".transcript.json")
    sidecar.write_text(json.dumps({"text": "hello world"}))

    called: list[Path] = []

    def transcriber(p: Path) -> str:
        called.append(p)
        return "should not be called"

    results = transcribe.transcribe_all([media], transcriber=transcriber)
    assert results[media] == "hello world"
    assert called == []


def test_transcribe_all_writes_sidecar(tmp_path: Path) -> None:
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")

    def transcriber(p: Path) -> str:
        return "fresh transcript"

    results = transcribe.transcribe_all([media], transcriber=transcriber)
    assert results[media] == "fresh transcript"
    sidecar = media.with_suffix(media.suffix + ".transcript.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["text"] == "fresh transcript"


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


def test_transcribe_all_cache_sidecar_corrupt_falls_through(tmp_path: Path) -> None:
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")
    sidecar = media.with_suffix(media.suffix + ".transcript.json")
    sidecar.write_text("not json{")  # corrupt

    def runner(p: Path) -> str:
        return "fresh after corrupt sidecar"

    results = transcribe.transcribe_all([media], transcriber=runner)
    assert results[media] == "fresh after corrupt sidecar"


def test_transcribe_all_cache_root_hit(tmp_path: Path) -> None:
    media = tmp_path / "a.mp3"
    media.write_bytes(b"\x00")
    from pengram.cache import save_cached

    save_cached(tmp_path, media, {"text": "from cache root"})

    def runner(p: Path) -> str:
        raise AssertionError("should not be called")

    results = transcribe.transcribe_all([media], cache_root=tmp_path, transcriber=runner)
    assert results[media] == "from cache root"
