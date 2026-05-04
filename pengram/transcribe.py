# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Audio/video transcription via Whisper.

Dispatches to local faster-whisper or the OpenAI Whisper API based on
:mod:`pengram.config`. Each media file produces a plain-text
``.transcript`` sidecar next to the source. If the sidecar already
exists, Whisper is skipped — the file IS the cache.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from . import config as _config
from ._ui import say as _ui_say
from ._ui import warn as _ui_warn
from .errors import ConfigError

_FASTER_WHISPER_HINT = (
    "faster-whisper is required for WHISPER_MODE=local. Install with: pip install 'pengram[video]'"
)
_OPENAI_HINT = (
    "openai is required for WHISPER_MODE=openai or openrouter. "
    "Install with: pip install 'pengram[openai]'"
)

_DEFAULT_DOMAIN_PROMPT = (
    "Transcribe this recording verbatim. Preserve technical terminology, "
    "proper nouns, and numbers exactly as spoken."
)


def build_domain_prompt(god_nodes: Iterable[str] | None = None) -> str:
    """Build a Whisper domain-hint prompt from prominent graph nodes."""
    if not god_nodes:
        return _DEFAULT_DOMAIN_PROMPT
    names = [n for n in god_nodes if n]
    if not names:
        return _DEFAULT_DOMAIN_PROMPT
    hint = ", ".join(names[:40])
    return f"{_DEFAULT_DOMAIN_PROMPT} Prominent topics in this corpus include: {hint}."


def _transcript_path(media_path: Path) -> Path:
    """Return the ``.transcript`` sidecar path for a media file."""
    return media_path.with_suffix(".transcript")


def transcribe_local(
    media_path: Path,
    *,
    model: str | None = None,
    prompt: str | None = None,
    language: str | None = None,
) -> str:
    """Transcribe via faster-whisper on CPU/GPU. Returns the full transcript text."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_FASTER_WHISPER_HINT) from exc

    model_name = model or _config.WHISPER_MODEL
    whisper = WhisperModel(model_name)
    segments, _info = whisper.transcribe(
        str(media_path),
        initial_prompt=prompt,
        language=language,
    )
    return "\n".join(segment.text.strip() for segment in segments if segment.text)


def transcribe_api(
    media_path: Path,
    *,
    model: str | None = None,
    prompt: str | None = None,
    language: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Transcribe via the OpenAI (or OpenAI-compatible) Whisper API."""
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_OPENAI_HINT) from exc

    key = api_key or _config.OPENAI_API_KEY
    if not key:
        raise ConfigError("WHISPER_MODE=openai/openrouter requires OPENAI_API_KEY")
    client = openai.OpenAI(api_key=key, base_url=base_url)
    with open(media_path, "rb") as audio:
        resp = client.audio.transcriptions.create(
            model=model or "whisper-1",
            file=audio,
            prompt=prompt,
            language=language,
        )
    # OpenAI client returns a pydantic object with .text
    return getattr(resp, "text", "") or ""


def transcribe(
    media_path: Path,
    *,
    mode: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    language: str | None = None,
) -> str:
    """Dispatch to the configured Whisper backend and return transcript text."""
    resolved_mode = mode or _config.WHISPER_MODE
    if resolved_mode == "local":
        return transcribe_local(media_path, model=model, prompt=prompt, language=language)
    if resolved_mode == "openai":
        return transcribe_api(media_path, model=model, prompt=prompt, language=language)
    if resolved_mode == "openrouter":
        return transcribe_api(
            media_path,
            model=model,
            prompt=prompt,
            language=language,
            api_key=_config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    raise ConfigError(f"Unknown WHISPER_MODE: {resolved_mode!r}")


def transcribe_all(
    media_files: Iterable[Path],
    *,
    transcriber: Callable[[Path], str] | None = None,
    prompt: str | None = None,
) -> list[Path]:
    """Transcribe a batch of media files, writing ``.transcript`` sidecars.

    Each media file produces a plain-text ``<stem>.transcript`` file
    next to the source. If the sidecar already exists, Whisper is
    skipped — the file IS the cache. Deleting the sidecar triggers
    re-transcription on the next run.

    Returns the list of transcript file paths (both pre-existing and
    newly written).
    """
    transcript_paths: list[Path] = []
    already = 0
    for media in media_files:
        tp = _transcript_path(media)
        if tp.exists():
            already += 1
            transcript_paths.append(tp)
            continue
        runner = transcriber or (lambda p: transcribe(p, prompt=prompt))
        try:
            text = runner(media)
        except Exception as exc:
            _ui_warn(f"transcription failed for {media}: {exc}")
            continue
        try:
            tp.write_text(text, encoding="utf-8")
        except OSError as exc:
            _ui_warn(f"transcript write failed for {media}: {exc}")
            continue
        transcript_paths.append(tp)
    wrote = len(transcript_paths) - already
    if wrote:
        _ui_say(f"  Whisper: wrote {wrote} transcript(s)")
    elif already:
        _ui_say(f"  Whisper: {already} transcript(s) already exist, skipped")
    return transcript_paths


__all__ = [
    "build_domain_prompt",
    "transcribe",
    "transcribe_local",
    "transcribe_api",
    "transcribe_all",
]
