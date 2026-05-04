# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""External CLI tool resolution.

When PENgram is invoked via a venv binary directly (without activating
the venv), subprocesses inherit a PATH that doesn't include the venv's
bin directory.  Tools installed alongside PENgram (yt-dlp, claude) then
appear missing even though they're sitting in the same venv.

:func:`resolve_tool` looks in the venv first, then falls back to PATH,
and raises a clear error naming the tool and the install command that
would fix it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .errors import PengramError


class ToolNotFoundError(PengramError, FileNotFoundError):
    """Raised when an external CLI tool isn't on PATH or in the venv bin."""


def _venv_bin_dir() -> Path | None:
    """Return the bin/ dir of the active venv, or ``None``."""
    if sys.prefix == sys.base_prefix:
        return None
    bin_dir = Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin")
    return bin_dir if bin_dir.is_dir() else None


def resolve_tool(name: str, *, install_hint: str) -> str:
    """Return the absolute path to the external CLI tool *name*.

    Lookup order:
      1. ``<venv>/bin/<name>`` (or ``Scripts/<name>.exe`` on Windows).
      2. ``shutil.which(name)`` — inherited PATH.

    Raises :class:`ToolNotFoundError` with *install_hint* if neither
    location resolves.
    """
    venv_bin = _venv_bin_dir()
    if venv_bin is not None:
        for candidate in (venv_bin / name, venv_bin / f"{name}.exe"):
            if candidate.is_file():
                return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    raise ToolNotFoundError(
        f"{name!r} not found in venv ({venv_bin or 'no venv detected'}) or on PATH. {install_hint}"
    )


__all__ = ["ToolNotFoundError", "resolve_tool"]
