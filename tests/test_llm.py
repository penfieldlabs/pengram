# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.llm."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from pengram.llm import LLMError, TransientLLMError, call_llm, parse_json_response


def make_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# claude-cli dispatch
# ---------------------------------------------------------------------------


def test_claude_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram.llm.resolve_tool", lambda *a, **kw: "claude")

    def runner(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        assert args[0] == "claude"
        return make_result(stdout="hello there")

    out = call_llm("hi", provider="claude-cli", model="haiku", runner=runner)
    assert out == "hello there"


def test_claude_cli_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram._tools.shutil.which", lambda name: None)
    monkeypatch.setattr("pengram._tools._venv_bin_dir", lambda: None)
    from pengram._tools import ToolNotFoundError

    with pytest.raises((LLMError, ToolNotFoundError), match="claude|not found"):
        call_llm("hi", provider="claude-cli")


def test_claude_cli_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram.llm.resolve_tool", lambda *a, **kw: "claude")

    def runner(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return make_result(stderr="bad", returncode=2)

    with pytest.raises(LLMError, match="failed"):
        call_llm("hi", provider="claude-cli", runner=runner)


def test_claude_cli_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram.llm.resolve_tool", lambda *a, **kw: "claude")

    def runner(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, 5)

    with pytest.raises(LLMError, match="timed out"):
        call_llm("hi", provider="claude-cli", runner=runner, _sleep=lambda _: None)


# ---------------------------------------------------------------------------
# openai / openrouter dispatch (mocked)
# ---------------------------------------------------------------------------


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.last: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _FakeResp:
        self.last = kwargs
        return _FakeResp("from openai")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(
        self, api_key: str, base_url: str | None = None, timeout: int | None = None
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chat = _FakeChat()


class _FakeOpenAIModule:
    OpenAI = _FakeClient


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAIModule:
    import sys

    fake = _FakeOpenAIModule()
    monkeypatch.setitem(sys.modules, "openai", fake)
    return fake


def test_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setattr("pengram.config.OPENAI_API_KEY", None, raising=False)
    with pytest.raises(LLMError, match="API key"):
        call_llm("hi", provider="openai", model="gpt-4o-mini")


def test_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setattr("pengram.config.OPENAI_API_KEY", "sk-test", raising=False)
    out = call_llm("hi", provider="openai", model="gpt-4o-mini")
    assert out == "from openai"


def test_openrouter_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setattr("pengram.config.OPENROUTER_API_KEY", "sk-or", raising=False)
    out = call_llm("hi", provider="openrouter", model="openai/gpt-4o-mini")
    assert out == "from openai"


def test_unknown_provider() -> None:
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        call_llm("hi", provider="gemini")


# ---------------------------------------------------------------------------
# Ollama provider (mocked HTTP)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_ollama_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from pengram.llm import _call_ollama

    captured: dict[str, Any] = {}

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps({"response": "hello from ollama"}).encode())

    out = _call_ollama(
        "hi",
        "llama3.1:8b",
        60,
        base_url="http://localhost:11434",
        opener=fake_opener,
    )
    assert out == "hello from ollama"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["model"] == "llama3.1:8b"
    assert captured["body"]["prompt"] == "hi"
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["temperature"] == 0.0


def test_ollama_network_failure_raises_llm_error() -> None:
    import urllib.error

    from pengram.llm import _call_ollama

    def bad_opener(request, timeout):
        raise urllib.error.URLError("Connection refused")

    with pytest.raises(LLMError, match="Ollama request failed"):
        _call_ollama("hi", "llama3.1:8b", 60, opener=bad_opener)


def test_ollama_non_json_response_raises_llm_error() -> None:
    from pengram.llm import _call_ollama

    def opener(request, timeout):
        return _FakeResponse(b"not json")

    with pytest.raises(LLMError, match="non-JSON envelope"):
        _call_ollama("hi", "llama3.1:8b", 60, opener=opener)


def test_ollama_missing_response_field_raises() -> None:
    from pengram.llm import _call_ollama

    def opener(request, timeout):
        return _FakeResponse(json.dumps({"done": True}).encode())

    with pytest.raises(LLMError, match="missing 'response'"):
        _call_ollama("hi", "llama3.1:8b", 60, opener=opener)


def test_call_llm_dispatches_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """call_llm(provider='ollama', ...) wires through to _call_ollama."""
    import pengram.llm as llm_mod

    captured: dict[str, Any] = {}

    def fake_ollama(prompt, model, timeout, *, temperature=0.0):
        captured["prompt"] = prompt
        captured["model"] = model
        return "ollama says hi"

    monkeypatch.setattr(llm_mod, "_call_ollama", fake_ollama)
    out = call_llm("hi there", provider="ollama", model="llama3.1:8b")
    assert out == "ollama says hi"
    assert captured["prompt"] == "hi there"
    assert captured["model"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# Ollama auto-detection
# ---------------------------------------------------------------------------


def _make_opener(handlers: dict[str, Any]):
    """Return a fake opener that dispatches by URL path."""

    def _opener(request, timeout):
        for path, payload in handlers.items():
            if request.full_url.endswith(path):
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                return _FakeResponse(body)
        raise AssertionError(f"unmocked URL: {request.full_url}")

    return _opener


def test_resolve_ollama_picks_first_running_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from pengram.llm import _clear_ollama_cache, _resolve_ollama_model

    _clear_ollama_cache()
    opener = _make_opener(
        {
            "/api/ps": {"models": [{"name": "llama3.1:8b"}]},
            "/api/show": {"parameters": "num_ctx 8192\ntemperature 0.7"},
        }
    )
    model, chunk_chars = _resolve_ollama_model(
        "http://fake:11434",
        opener=opener,
    )
    assert model == "llama3.1:8b"
    # 8192 tokens × 3 chars/token
    assert chunk_chars == 8192 * 3


def test_resolve_ollama_reports_multi_loaded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pengram.llm import _clear_ollama_cache, _detect_ollama_model

    _clear_ollama_cache()
    opener = _make_opener(
        {
            "/api/ps": {
                "models": [
                    {"name": "qwen2.5:7b"},
                    {"name": "llama3.1:70b"},
                ]
            },
        }
    )
    model = _detect_ollama_model("http://fake:11434", opener=opener)
    assert model == "qwen2.5:7b"
    out = capsys.readouterr().out
    assert "Multiple Ollama models loaded" in out
    assert "qwen2.5:7b" in out


def test_resolve_ollama_falls_back_to_available_when_none_loaded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pengram.llm import _clear_ollama_cache, _detect_ollama_model

    _clear_ollama_cache()
    opener = _make_opener(
        {
            "/api/ps": {"models": []},
            "/api/tags": {"models": [{"name": "mistral:7b"}]},
        }
    )
    model = _detect_ollama_model("http://fake:11434", opener=opener)
    assert model == "mistral:7b"
    assert "No Ollama model loaded" in capsys.readouterr().out


def test_resolve_ollama_raises_when_no_models() -> None:
    from pengram.llm import _clear_ollama_cache, _detect_ollama_model

    _clear_ollama_cache()
    opener = _make_opener(
        {
            "/api/ps": {"models": []},
            "/api/tags": {"models": []},
        }
    )
    with pytest.raises(LLMError, match="No Ollama models found"):
        _detect_ollama_model("http://fake:11434", opener=opener)


def test_resolve_ollama_caches_result() -> None:
    """Second call doesn't re-hit the HTTP endpoint."""
    from pengram.llm import _clear_ollama_cache, _resolve_ollama_model

    _clear_ollama_cache()
    call_count = {"n": 0}

    def opener(request, timeout):
        call_count["n"] += 1
        if "/api/ps" in request.full_url:
            return _FakeResponse(json.dumps({"models": [{"name": "m"}]}).encode())
        return _FakeResponse(json.dumps({"parameters": "num_ctx 4096"}).encode())

    _resolve_ollama_model("http://fake:11434", opener=opener)
    first = call_count["n"]
    _resolve_ollama_model("http://fake:11434", opener=opener)
    assert call_count["n"] == first, "second call should hit cache, not HTTP"


def test_ollama_context_window_parses_num_ctx() -> None:
    from pengram.llm import _ollama_context_window

    opener = _make_opener(
        {
            "/api/show": {"parameters": "temperature 0.7\nnum_ctx 16384\ntop_p 1"},
        }
    )
    assert _ollama_context_window("http://fake:11434", "m", opener=opener) == 16384


def test_ollama_context_window_fallback_when_missing() -> None:
    from pengram.llm import _ollama_context_window

    opener = _make_opener(
        {
            "/api/show": {"parameters": "temperature 0.7"},
        }
    )
    # Default 8192 when num_ctx is absent.
    assert _ollama_context_window("http://fake:11434", "m", opener=opener) == 8192


def test_call_llm_ollama_auto_resolves_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When model='_auto', call_llm calls _resolve_ollama_model()."""
    import pengram.llm as llm_mod

    llm_mod._clear_ollama_cache()

    resolved = {"n": 0}

    def fake_resolve(*a, **kw):
        resolved["n"] += 1
        return ("detected-model:latest", 30000)

    captured: dict[str, Any] = {}

    def fake_ollama(prompt, model, timeout, *, temperature=0.0):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(llm_mod, "_resolve_ollama_model", fake_resolve)
    monkeypatch.setattr(llm_mod, "_call_ollama", fake_ollama)

    call_llm("hi", provider="ollama", model="_auto")
    assert resolved["n"] == 1
    assert captured["model"] == "detected-model:latest"


def test_call_llm_ollama_explicit_model_skips_auto_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --llm-model bypasses auto-detection."""
    import pengram.llm as llm_mod

    llm_mod._clear_ollama_cache()

    def fake_resolve(*a, **kw):
        raise AssertionError("should not be called")

    def fake_ollama(prompt, model, timeout, *, temperature=0.0):
        return f"called with {model}"

    monkeypatch.setattr(llm_mod, "_resolve_ollama_model", fake_resolve)
    monkeypatch.setattr(llm_mod, "_call_ollama", fake_ollama)

    out = call_llm("hi", provider="ollama", model="qwen2.5:7b")
    assert out == "called with qwen2.5:7b"


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_json_plain() -> None:
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_markdown_fence() -> None:
    text = '```json\n{"a": 1}\n```'
    assert parse_json_response(text) == {"a": 1}


def test_parse_json_bare_fence() -> None:
    text = '```\n{"a": 1}\n```'
    assert parse_json_response(text) == {"a": 1}


def test_parse_json_extracts_from_prose() -> None:
    text = 'Sure, here is the JSON: {"ok": true} hope that helps.'
    assert parse_json_response(text) == {"ok": True}


def test_parse_json_invalid_raises() -> None:
    with pytest.raises(LLMError):
        parse_json_response("definitely not json")


def test_parse_json_array() -> None:
    assert parse_json_response("[1, 2, 3]") == [1, 2, 3]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def test_with_retry_succeeds_on_first_attempt() -> None:
    from pengram.llm import _with_retry

    assert _with_retry(lambda: "ok") == "ok"


def test_with_retry_retries_transient_then_succeeds() -> None:
    from pengram.llm import _with_retry

    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientLLMError("network glitch")
        return "recovered"

    sleeps: list[float] = []
    result = _with_retry(flaky, delays=(0, 0, 0), sleep=lambda s: sleeps.append(s))
    assert result == "recovered"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_with_retry_raises_after_exhausting_attempts() -> None:
    from pengram.llm import _with_retry

    def always_fails() -> str:
        raise TransientLLMError("down")

    with pytest.raises(TransientLLMError, match="down"):
        _with_retry(always_fails, delays=(0, 0, 0), sleep=lambda _: None)


def test_with_retry_does_not_retry_permanent_errors() -> None:
    from pengram.llm import _with_retry

    calls = {"n": 0}

    def permanent() -> str:
        calls["n"] += 1
        raise LLMError("auth failure")

    with pytest.raises(LLMError, match="auth failure"):
        _with_retry(permanent, delays=(0, 0, 0), sleep=lambda _: None)
    assert calls["n"] == 1


def test_call_llm_retries_transient_claude_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram.llm.resolve_tool", lambda *a, **kw: "claude")
    calls = {"n": 0}

    def fake_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        if calls["n"] < 2:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=5)
        return make_result(stdout="ok")

    result = call_llm(
        "hi", provider="claude-cli", runner=fake_runner, model="haiku", _sleep=lambda _: None
    )
    assert result == "ok"
    assert calls["n"] == 2


def test_transient_llm_error_is_subclass_of_llm_error() -> None:
    assert issubclass(TransientLLMError, LLMError)
