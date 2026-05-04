# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.config."""

from __future__ import annotations

import pytest

from pengram import config as c


@pytest.fixture(autouse=True)
def _stub_claude_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests pass llm_provider='claude-cli'. Prevent PATH check failures
    on CI runners that don't have the claude CLI installed."""
    monkeypatch.setattr(
        "pengram.config._shutil.which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )


def test_valid_default_config_passes() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert errors == []


def test_invalid_output_target() -> None:
    errors = c.validate(
        output_target="mongo",
        whisper_mode="local",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("OUTPUT_TARGET" in e for e in errors)


def test_invalid_whisper_mode() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="tinfoil",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("WHISPER_MODE" in e for e in errors)


def test_whisper_openai_without_key() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="openai",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("openai" in e.lower() and "OPENAI_API_KEY" in e for e in errors)


def test_whisper_openrouter_without_key() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="openrouter",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("openrouter" in e.lower() and "OPENROUTER_API_KEY" in e for e in errors)


def test_whisper_openai_with_key_ok() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="openai",
        llm_provider="claude-cli",
        openai_api_key="sk-test",
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert errors == []


def test_invalid_llm_provider() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="llama-3",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("LLM_PROVIDER" in e for e in errors)


def test_llm_openai_requires_key() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="openai",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("OPENAI_API_KEY" in e for e in errors)


def test_llm_openrouter_requires_key() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="openrouter",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={},
    )
    assert any("OPENROUTER_API_KEY" in e for e in errors)


def test_youtube_channel_missing_url() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={"bad": c.YouTubeChannel(url="", label="Bad")},
    )
    assert any("url" in e for e in errors)


def test_youtube_channel_invalid_tab() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={
            "bad": c.YouTubeChannel(url="https://youtube.com/@x", label="X", tabs=["podcasts"])
        },
    )
    assert any("podcasts" in e for e in errors)


def test_youtube_channel_valid_tab() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={
            "x": c.YouTubeChannel(
                url="https://youtube.com/@x", label="X", tabs=["videos", "streams"]
            )
        },
    )
    assert errors == []


def test_youtube_channel_default_tabs_is_videos_only() -> None:
    """Regression pin: default tab is videos. Streams and shorts are opt-in."""
    channel = c.YouTubeChannel(url="https://youtube.com/@x", label="X")
    assert channel.tabs == ["videos"]


def test_youtube_channel_empty_tabs() -> None:
    errors = c.validate(
        output_target="penfield",
        whisper_mode="local",
        llm_provider="claude-cli",
        openai_api_key=None,
        openrouter_api_key=None,
        youtube_channels={"x": c.YouTubeChannel(url="https://youtube.com/@x", label="X", tabs=[])},
    )
    assert any("tabs" in e for e in errors)


def test_provider_defaults_use_cheapest_sensible_model() -> None:
    """v0.2.0 directive: each provider defaults to the cheapest model on its
    platform. Upgrades are opt-in via env or --llm-model."""
    from pengram.config import _DEFAULT_MODELS

    claude = _DEFAULT_MODELS["claude-cli"]
    assert claude["extract_model"] == "haiku"
    assert claude["link_model"] == "haiku"
    assert claude["synth_model"] == "sonnet"

    openai = _DEFAULT_MODELS["openai"]
    assert openai["extract_model"] == "gpt-4o-mini"
    assert openai["link_model"] == "gpt-4o-mini"
    assert openai["synth_model"] == "gpt-4o-mini"

    openrouter = _DEFAULT_MODELS["openrouter"]
    for role in ("extract_model", "link_model", "synth_model"):
        assert openrouter[role] == "openai/gpt-4o-mini"

    ollama = _DEFAULT_MODELS["ollama"]
    for role in ("extract_model", "link_model", "synth_model"):
        assert ollama[role] == "_auto"


def test_llm_dict_has_expected_keys() -> None:
    expected = {
        "extract_model",
        "link_model",
        "synth_model",
        "parallel_workers",
        "extract_timeout",
        "link_timeout",
        "synth_timeout",
    }
    assert expected.issubset(c.LLM.keys())


def test_claude_cli_missing_from_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram.config._shutil.which", lambda cmd: None)  # override autouse
    errors = c.validate(llm_provider="claude-cli")
    assert any("claude" in e and "PATH" in e for e in errors)


def test_claude_cli_present_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pengram.config._shutil.which",
        lambda cmd: "/usr/bin/claude" if cmd == "claude" else None,
    )
    errors = c.validate(llm_provider="claude-cli")
    assert not any("claude" in e and "PATH" in e for e in errors)


def test_constants_exist() -> None:
    assert c.PROJECT_NAME
    assert c.PROJECT_LABEL
    assert c.OUTPUT_TARGET in c.VALID_OUTPUT_TARGETS
    assert c.WHISPER_MODE in c.VALID_WHISPER_MODES
    assert c.LLM_PROVIDER in c.VALID_LLM_PROVIDERS
    # v0.2.0: MAX_DOCUMENT_CHARS replaced by per-provider chunk sizes.
    assert c.default_chunk_chars() > 0
    assert c.default_chunk_overlap() >= 0


def test_llm_config_includes_image_model() -> None:
    from pengram import config as c

    assert "image_model" in c.LLM
    assert isinstance(c.LLM["image_model"], str)
    assert len(c.LLM["image_model"]) > 0


def test_all_providers_have_image_model_default() -> None:
    from pengram.config import _DEFAULT_MODELS

    for provider, models in _DEFAULT_MODELS.items():
        assert "image_model" in models, f"{provider} missing image_model default"
