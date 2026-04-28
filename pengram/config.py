# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Project configuration for PENgram.

Single customization point. Override via environment variables or by editing
the constants below. Run :func:`validate` to check the active configuration.
"""

from __future__ import annotations

import os
import shutil as _shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------

PROJECT_NAME: str = os.environ.get("PENGRAM_PROJECT_NAME", "pengram")
PROJECT_LABEL: str = os.environ.get("PENGRAM_PROJECT_LABEL", "PENgram")

# ---------------------------------------------------------------------------
# Output target
# ---------------------------------------------------------------------------

VALID_OUTPUT_TARGETS: frozenset[str] = frozenset({"penfield", "obsidian", "both", "json"})

OUTPUT_TARGET: str = os.environ.get("PENGRAM_OUTPUT_TARGET", "penfield")

# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

VALID_WHISPER_MODES: frozenset[str] = frozenset({"local", "openai", "openrouter"})

WHISPER_MODE: str = os.environ.get("PENGRAM_WHISPER_MODE", "local")
WHISPER_MODEL: str = os.environ.get("PENGRAM_WHISPER_MODEL", "base.en")

# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

VALID_LLM_PROVIDERS: frozenset[str] = frozenset(
    {
        "claude-cli",
        "openai",
        "openrouter",
        "ollama",
    }
)

LLM_PROVIDER: str = os.environ.get("PENGRAM_LLM_PROVIDER", "claude-cli")

# Ollama base URL. No API key required — auth is network-local. Override
# via env for non-default Ollama hosts.
OLLAMA_BASE_URL: str = os.environ.get(
    "PENGRAM_OLLAMA_BASE_URL",
    "http://localhost:11434",
)

# Provider-specific default models.
# Each provider defaults to its cheapest sensible model. Upgrading is
# the user's decision — we never spend their money by default.
#
# The Ollama row uses the sentinel ``_auto``; it's resolved at call time
# by :func:`pengram.llm._resolve_ollama_model` to whichever model is
# currently loaded (``/api/ps``) or available (``/api/tags``). The user
# already picked their model when they ran ``ollama pull`` — use it.
_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "claude-cli": {
        "extract_model": "haiku",
        "link_model": "haiku",
        "synth_model": "sonnet",
    },
    "openai": {
        "extract_model": "gpt-4o-mini",
        "link_model": "gpt-4o-mini",
        "synth_model": "gpt-4o-mini",
    },
    "openrouter": {
        "extract_model": "openai/gpt-4o-mini",
        "link_model": "openai/gpt-4o-mini",
        "synth_model": "openai/gpt-4o-mini",
    },
    "ollama": {
        "extract_model": "_auto",
        "link_model": "_auto",
        "synth_model": "_auto",
    },
}


def _default_llm_config() -> dict[str, Any]:
    models = _DEFAULT_MODELS.get(LLM_PROVIDER, _DEFAULT_MODELS["claude-cli"])
    return {
        "extract_model": os.environ.get("PENGRAM_EXTRACT_MODEL", models["extract_model"]),
        "link_model": os.environ.get("PENGRAM_LINK_MODEL", models["link_model"]),
        "synth_model": os.environ.get("PENGRAM_SYNTH_MODEL", models["synth_model"]),
        "parallel_workers": int(os.environ.get("PENGRAM_PARALLEL_WORKERS", "4")),
        "extract_timeout": int(os.environ.get("PENGRAM_EXTRACT_TIMEOUT", "300")),
        "link_timeout": int(os.environ.get("PENGRAM_LINK_TIMEOUT", "180")),
        "synth_timeout": int(os.environ.get("PENGRAM_SYNTH_TIMEOUT", "600")),
    }


LLM: dict[str, Any] = _default_llm_config()

# ---------------------------------------------------------------------------
# API keys (read from env — never hardcode)
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# YouTube configuration
# ---------------------------------------------------------------------------

# YouTube tab URL segments. These map to /videos, /streams, /shorts on the
# channel URL. Note: the YouTube UI labels the "streams" tab as "Live" — both
# names refer to the same content (past livestreams). We use yt-dlp's URL-
# segment spelling ("streams") here because that's what the tool consumes.
VALID_YOUTUBE_TABS: frozenset[str] = frozenset({"videos", "streams", "shorts"})


@dataclass
class YouTubeChannel:
    """A single YouTube channel configuration.

    ``tabs`` controls which parts of the channel to enumerate. Defaults to
    regular ``videos`` only. Add ``"streams"`` to also pull past livestreams
    (shown as "Live" in the YouTube UI) or ``"shorts"`` for short-form
    vertical clips. Each tab is a separate yt-dlp playlist pull.
    """

    url: str
    label: str
    tabs: list[str] = field(default_factory=lambda: ["videos"])


# Dict of ``{channel_key: YouTubeChannel}`` used by ``pengram youtube <key>``.
YOUTUBE_CHANNELS: dict[str, YouTubeChannel] = {}

# ---------------------------------------------------------------------------
# Networking / proxy
# ---------------------------------------------------------------------------

PROXY: str | None = os.environ.get("PENGRAM_PROXY")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(os.environ.get("PENGRAM_BASE_DIR", Path.cwd()))
OUTPUT_DIR: Path = Path(os.environ.get("PENGRAM_OUTPUT_DIR", BASE_DIR / "pengram-out"))

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Chunk size ceiling per LLM call. Documents larger than this are split
# into overlapping chunks and extracted chunk-by-chunk, then merged
# (v0.2.0). Defaults vary by provider — cloud models have large context
# windows; local 7B models on Ollama typically don't.
_DEFAULT_CHUNK_CHARS_BY_PROVIDER: dict[str, int] = {
    "claude-cli": 95_000,
    "openai": 95_000,
    "openrouter": 95_000,
    "ollama": 20_000,
}
_DEFAULT_CHUNK_OVERLAP_CHARS: int = 5_000


def default_chunk_chars(provider: str | None = None) -> int:
    """Return the per-provider default chunk size.

    ``PENGRAM_CHUNK_CHARS`` env var overrides. Unknown provider falls
    back to the cloud default (95K).
    """
    env = os.environ.get("PENGRAM_CHUNK_CHARS")
    if env:
        try:
            return max(1000, int(env))
        except ValueError:
            pass
    key = provider or LLM_PROVIDER
    return _DEFAULT_CHUNK_CHARS_BY_PROVIDER.get(key, 95_000)


def default_chunk_overlap() -> int:
    env = os.environ.get("PENGRAM_CHUNK_OVERLAP_CHARS")
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return _DEFAULT_CHUNK_OVERLAP_CHARS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_UNSET: Any = object()


def validate(
    *,
    output_target: str | None = None,
    whisper_mode: str | None = None,
    llm_provider: str | None = None,
    openai_api_key: Any = _UNSET,
    openrouter_api_key: Any = _UNSET,
    youtube_channels: dict[str, YouTubeChannel] | None = None,
) -> list[str]:
    """Validate the active config and return a list of human-readable errors.

    All parameters are optional; the current module-level values are used when
    not provided. Parameters exist so tests can exercise validation without
    mutating module state.
    """

    output_target = OUTPUT_TARGET if output_target is None else output_target
    whisper_mode = WHISPER_MODE if whisper_mode is None else whisper_mode
    llm_provider = LLM_PROVIDER if llm_provider is None else llm_provider
    openai_api_key = OPENAI_API_KEY if openai_api_key is _UNSET else openai_api_key
    openrouter_api_key = OPENROUTER_API_KEY if openrouter_api_key is _UNSET else openrouter_api_key
    youtube_channels = YOUTUBE_CHANNELS if youtube_channels is None else youtube_channels

    errors: list[str] = []

    if output_target not in VALID_OUTPUT_TARGETS:
        errors.append(
            f"OUTPUT_TARGET must be one of {sorted(VALID_OUTPUT_TARGETS)}, got {output_target!r}"
        )

    if whisper_mode not in VALID_WHISPER_MODES:
        errors.append(
            f"WHISPER_MODE must be one of {sorted(VALID_WHISPER_MODES)}, got {whisper_mode!r}"
        )
    elif whisper_mode == "openai" and not openai_api_key:
        errors.append("WHISPER_MODE=openai requires OPENAI_API_KEY to be set")
    elif whisper_mode == "openrouter" and not openrouter_api_key:
        errors.append("WHISPER_MODE=openrouter requires OPENROUTER_API_KEY to be set")

    if llm_provider not in VALID_LLM_PROVIDERS:
        errors.append(
            f"LLM_PROVIDER must be one of {sorted(VALID_LLM_PROVIDERS)}, got {llm_provider!r}"
        )
    elif llm_provider == "openai" and not openai_api_key:
        errors.append("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")
    elif llm_provider == "openrouter" and not openrouter_api_key:
        errors.append("LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY to be set")
    elif llm_provider == "claude-cli" and not _shutil.which("claude"):
        errors.append(
            "LLM_PROVIDER=claude-cli but 'claude' is not on PATH. "
            "Install: npm install -g @anthropic-ai/claude-code"
        )

    for key, channel in youtube_channels.items():
        if not isinstance(channel, YouTubeChannel):
            errors.append(f"YOUTUBE_CHANNELS[{key!r}] must be a YouTubeChannel")
            continue
        if not channel.url:
            errors.append(f"YOUTUBE_CHANNELS[{key!r}].url is required")
        if not channel.label:
            errors.append(f"YOUTUBE_CHANNELS[{key!r}].label is required")
        if not channel.tabs:
            errors.append(f"YOUTUBE_CHANNELS[{key!r}].tabs must be non-empty")
        for tab in channel.tabs:
            if tab not in VALID_YOUTUBE_TABS:
                errors.append(
                    f"YOUTUBE_CHANNELS[{key!r}].tabs contains invalid tab "
                    f"{tab!r}; valid tabs are {sorted(VALID_YOUTUBE_TABS)}"
                )

    return errors


__all__ = [
    "PROJECT_NAME",
    "PROJECT_LABEL",
    "OUTPUT_TARGET",
    "VALID_OUTPUT_TARGETS",
    "WHISPER_MODE",
    "WHISPER_MODEL",
    "VALID_WHISPER_MODES",
    "LLM_PROVIDER",
    "VALID_LLM_PROVIDERS",
    "LLM",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "YOUTUBE_CHANNELS",
    "VALID_YOUTUBE_TABS",
    "YouTubeChannel",
    "PROXY",
    "BASE_DIR",
    "OUTPUT_DIR",
    "default_chunk_chars",
    "default_chunk_overlap",
    "validate",
]
