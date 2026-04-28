# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Input validation and sanitization.

All external data — URLs from catalogs, paths from user config, labels from
LLM extraction — flows through this module before being stored, written to
disk, or rendered.
"""

from __future__ import annotations

import html
import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

from .errors import PengramError

# Maximum label length after sanitization. Matches the typical Obsidian filename
# ceiling (255 bytes) but leaves room for vault folder prefixes.
_MAX_LABEL_LEN = 200
_MAX_FILENAME_LEN = 120

# Allowed URL schemes for any externally-supplied URL.
_ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Control characters (C0 and C1) except newlines/tabs, plus DEL.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Characters unsafe in filenames across macOS/Linux/Windows.
_FILENAME_UNSAFE_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")

_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


class SecurityError(PengramError, ValueError):
    """Raised when input fails security validation."""


def validate_url(url: str) -> str:
    """Validate that ``url`` has an allowed scheme and return it unchanged.

    Raises :class:`SecurityError` for ``file://``, ``javascript:``, empty
    URLs, or URLs with no host.
    """
    if not url or not isinstance(url, str):
        raise SecurityError("URL must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        raise SecurityError(
            f"URL scheme {parsed.scheme!r} not allowed; "
            f"must be one of {sorted(_ALLOWED_URL_SCHEMES)}"
        )
    if not parsed.netloc:
        raise SecurityError(f"URL {url!r} has no host")
    return url


def validate_path(path: str | Path, allowed_root: str | Path) -> Path:
    """Resolve ``path`` and ensure it stays within ``allowed_root``.

    Returns the resolved :class:`Path`. Raises :class:`SecurityError` on
    directory-traversal attempts (``../``) that escape ``allowed_root``.
    """
    root = Path(allowed_root).resolve()
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SecurityError(f"Path {path!r} escapes allowed root {str(root)!r}") from exc
    return candidate


def sanitize_label(text: str, *, max_len: int = _MAX_LABEL_LEN) -> str:
    """Strip control chars, HTML-escape, and cap length."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _CONTROL_CHAR_RE.sub("", text).strip()
    cleaned = html.escape(cleaned, quote=False)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
    return cleaned


def sanitize_filename(name: str, *, max_len: int = _MAX_FILENAME_LEN) -> str:
    """Return a safe filesystem name derived from ``name``.

    Replaces unsafe characters with ``_``, strips leading dots/whitespace,
    avoids Windows reserved names, and caps length. Raises
    :class:`SecurityError` if the input reduces to an empty string.
    """
    if not isinstance(name, str):
        name = str(name)
    cleaned = _CONTROL_CHAR_RE.sub("", name)
    cleaned = _FILENAME_UNSAFE_RE.sub("_", cleaned)
    cleaned = cleaned.replace(os.sep, "_")
    if os.altsep:
        cleaned = cleaned.replace(os.altsep, "_")
    # Strip leading dots and whitespace so we never produce hidden files
    cleaned = cleaned.lstrip(". \t").rstrip(". \t")
    # Cap length while preserving the extension when possible.
    if len(cleaned) > max_len:
        stem, dot, ext = cleaned.rpartition(".")
        if dot and len(ext) < 20:
            trim = max_len - len(dot) - len(ext)
            cleaned = stem[: max(trim, 1)] + dot + ext
        else:
            cleaned = cleaned[:max_len]
    # Guard against Windows reserved names (CON, PRN, etc.) case-insensitive.
    stem = cleaned.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    if not cleaned:
        raise SecurityError("Sanitized filename is empty")
    return cleaned


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,24}$")


def validate_video_id(video_id: str) -> str:
    """Validate a YouTube video ID and return it unchanged.

    Raises :class:`SecurityError` if the ID doesn't match the expected
    pattern.  This prevents crafted IDs from escaping the output
    directory via glob patterns like ``out_dir.glob(f"{video_id}*.vtt")``.
    """
    if not isinstance(video_id, str) or not _VIDEO_ID_RE.match(video_id):
        raise SecurityError(
            f"Invalid video ID {video_id!r}: must be 8-24 alphanumeric/dash/underscore characters"
        )
    return video_id


_SLUG_MAX_LEN = 200
_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, *, max_len: int = _SLUG_MAX_LEN) -> str:
    """Convert a display label into a safe, predictable filename slug.

    Single source of truth for both vault filenames and wikilink targets —
    both MUST use this function so links never break. Caller is responsible
    for collision detection (append ``-2``, ``-3``, ...) across all slugs it
    produces; this function is pure.

    Rules applied in order:

    1. Unicode NFKD normalise; drop combining marks (accents).
    2. Drop NUL bytes entirely (security).
    3. Lowercase.
    4. Replace every run of non-alphanumeric characters with a single ``-``.
    5. Strip leading/trailing hyphens.
    6. Truncate to ``max_len``; prefer breaking at the last hyphen when the
       break point is past the halfway mark of ``max_len``.
    7. If the result is empty, return ``"_unnamed"``.
    8. If the first hyphen-separated segment is a Windows reserved name
       (``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``-``COM9``, ``LPT1``-
       ``LPT9``), prefix the whole slug with ``_``.
    """
    if not isinstance(name, str):
        name = str(name)
    normalised = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in normalised if not unicodedata.combining(ch) and ch != "\x00")
    lowered = stripped.lower()
    slug = _SLUG_NON_ALNUM_RE.sub("-", lowered).strip("-")

    if len(slug) > max_len:
        truncated = slug[:max_len]
        last_hyphen = truncated.rfind("-")
        if last_hyphen > max_len // 2:
            truncated = truncated[:last_hyphen]
        slug = truncated.strip("-")

    if not slug:
        return "_unnamed"

    first_segment = slug.split("-", 1)[0].upper()
    if first_segment in _WINDOWS_RESERVED:
        slug = f"_{slug}"
    return slug


__all__ = [
    "SecurityError",
    "validate_url",
    "validate_path",
    "sanitize_label",
    "sanitize_filename",
    "slugify",
    "validate_video_id",
]
