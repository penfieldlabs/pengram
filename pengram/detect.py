# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""File discovery and type classification.

Walks a directory, classifies files into :class:`FileType`, honours
``.pengramignore`` patterns, and provides helper functions for extracting
text from PDFs and ePubs.

Classification rules:

- Extensions map to a :class:`FileType` (``CODE``, ``DOCUMENT``, etc.).
- PDFs and ePubs are ``DOCUMENT`` — file format is metadata, not a content
  type (v0.2.0 removed the ``PAPER`` kind).
- Standard noise directories (``node_modules``, ``.git``, ``__pycache__``,
  virtualenvs, build outputs, ``pengram-out``) are always skipped.
- Dotfiles and dot-directories are skipped (including ``.env`` etc.).
- PENgram does NOT second-guess the user's input folder beyond that.
  Deciding what counts as "sensitive" is the user's call; use
  ``.pengramignore`` to exclude files explicitly. The previous
  "sensitive pattern" filter was removed in v0.2.0 after it silently
  dropped health-book filenames containing the word "secret".
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from ._ui import warn as _ui_warn


class FileType(str, Enum):
    """Top-level file classification."""

    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TRANSCRIPT = "transcript"


# ---------------------------------------------------------------------------
# Extension mappings
# ---------------------------------------------------------------------------

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".ts",
        ".js",
        ".jsx",
        ".tsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".hh",
        ".rb",
        ".swift",
        ".cs",
        ".php",
        ".lua",
        ".ex",
        ".exs",
        ".erl",
        ".clj",
        ".cljs",
        ".m",
        ".mm",
        ".jl",
        ".zig",
        ".v",
        ".sv",
        ".dart",
        ".vue",
        ".svelte",
        ".sh",
        ".bash",
        ".sql",
        ".r",
    }
)

# Every text-bearing format is a DOCUMENT. File format is metadata on the
# node, not a separate content type. v0.2.0 folded .pdf (was PAPER) and
# .epub (was silently ignored) into this set.
DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".mdx",
        ".markdown",
        ".txt",
        ".rst",
        ".html",
        ".htm",
        ".adoc",
        ".org",
        ".pdf",
        ".epub",
    }
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }
)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".m4v",
        ".flv",
    }
)

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
    }
)

TRANSCRIPT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".srt",
        ".transcript",
    }
)

# ---------------------------------------------------------------------------
# Directory skip list
# ---------------------------------------------------------------------------

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "venv",
        ".venv",
        "env",
        ".env",
        "node_modules",
        "__pycache__",
        ".git",
        "dist",
        "build",
        "target",
        "out",
        "site-packages",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "pengram-out",
        ".pengram-cache",
    }
)


def _is_noise_dir(name: str) -> bool:
    if name in _SKIP_DIRS:
        return True
    if name.endswith("_venv") or name.endswith("_env"):
        return True
    if name.endswith(".egg-info"):
        return True
    return False


# ---------------------------------------------------------------------------
# .pengramignore support — gitignore-style fnmatch
# ---------------------------------------------------------------------------


def _load_ignore_patterns(root: Path) -> list[str]:
    ignore_file = root / ".pengramignore"
    if not ignore_file.exists():
        return []
    patterns: list[str] = []
    try:
        for line in ignore_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    except OSError:
        return []
    return patterns


def _is_ignored(path: Path, root: Path, patterns: Iterable[str]) -> bool:
    if not patterns:
        return False
    try:
        rel = str(path.relative_to(root)).replace(os.sep, "/")
    except ValueError:
        rel = str(path).replace(os.sep, "/")
    parts = rel.split("/")
    for pat in patterns:
        p = pat.strip("/")
        if not p:
            continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(path.name, p):
            return True
        for i, part in enumerate(parts):
            if fnmatch.fnmatch(part, p):
                return True
            if fnmatch.fnmatch("/".join(parts[: i + 1]), p):
                return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(path: Path) -> FileType | None:
    """Return the :class:`FileType` for ``path``, or ``None`` if unknown/skipped."""
    ext = path.suffix.lower()
    if ext in CODE_EXTENSIONS:
        return FileType.CODE
    if ext in DOCUMENT_EXTENSIONS:
        return FileType.DOCUMENT
    if ext in IMAGE_EXTENSIONS:
        return FileType.IMAGE
    if ext in TRANSCRIPT_EXTENSIONS:
        return FileType.TRANSCRIPT
    if ext in VIDEO_EXTENSIONS:
        return FileType.VIDEO
    if ext in AUDIO_EXTENSIONS:
        return FileType.AUDIO
    return None


# ---------------------------------------------------------------------------
# Walk and collect
# ---------------------------------------------------------------------------


def collect_files(
    root: Path,
    *,
    follow_symlinks: bool = False,
    exclude: Iterable[Path] | None = None,
) -> dict[FileType, list[Path]]:
    """Walk ``root`` and group files by :class:`FileType`.

    Always skips:
      - Hidden files/directories (dot-prefixed) except the root itself.
        This catches ``.env``, ``.git``, etc. — but it is not a security
        filter. Anything non-hidden is processed.
      - Noise directories (see ``_SKIP_DIRS``).
      - Files matching ``.pengramignore`` patterns in ``root``.
      - Any paths listed in ``exclude`` (resolved). This is how ``cmd_run``
        prevents a nested output directory from being re-scanned on the
        next run, which would otherwise pull the previous run's output
        (graph.html, vault-penfield/, …) back into the input set.
    """
    root = root.resolve()
    patterns = _load_ignore_patterns(root)
    excluded: set[Path] = {Path(p).resolve() for p in (exclude or [])}
    groups: dict[FileType, list[Path]] = {ft: [] for ft in FileType}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dp = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and not _is_noise_dir(d)
            and not _is_ignored(dp / d, root, patterns)
            and (dp / d).resolve() not in excluded
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            p = dp / fname
            if _is_ignored(p, root, patterns):
                continue
            ftype = classify(p)
            if ftype is not None:
                groups[ftype].append(p)

    return groups


# ---------------------------------------------------------------------------
# PDF text extraction (optional dep: pypdf)
# ---------------------------------------------------------------------------

_PYPDF_INSTALL_HINT = (
    "pypdf is required to extract text from PDF files. Install with: pip install 'pengram[pdf]'"
)


def extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF via pypdf.

    Returns an empty string if pypdf is unavailable or extraction fails.
    In both cases the caller is expected to log and skip.
    """
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception as exc:
        _ui_warn(f"PDF extraction failed for {path}: {exc}")
        return ""


def require_pypdf() -> None:
    """Raise ``ImportError`` with install hint if pypdf is unavailable."""
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:
        raise ImportError(_PYPDF_INSTALL_HINT) from exc


# ---------------------------------------------------------------------------
# ePub text extraction (optional deps: ebooklib + beautifulsoup4)
# ---------------------------------------------------------------------------

_EPUB_INSTALL_HINT = (
    "ebooklib and beautifulsoup4 are required to extract text from ePub "
    "files. Install with: pip install 'pengram[epub]'"
)


def extract_epub_text(path: Path) -> str:
    """Extract plain text from an ePub via ebooklib + BeautifulSoup.

    Returns an empty string if deps are unavailable or extraction fails.
    Caller is expected to log and skip.
    """
    try:
        import ebooklib  # type: ignore[import-not-found]
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        from ebooklib import epub  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        book = epub.read_epub(str(path))
        parts: list[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html = item.get_content()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    except Exception as exc:
        _ui_warn(f"ePub extraction failed for {path}: {exc}")
        return ""


def require_epub() -> None:
    """Raise ``ImportError`` with install hint if epub deps are unavailable."""
    try:
        import bs4  # noqa: F401
        import ebooklib  # noqa: F401
    except ImportError as exc:
        raise ImportError(_EPUB_INSTALL_HINT) from exc


__all__ = [
    "FileType",
    "CODE_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "TRANSCRIPT_EXTENSIONS",
    "classify",
    "collect_files",
    "extract_pdf_text",
    "extract_epub_text",
    "require_pypdf",
    "require_epub",
]
