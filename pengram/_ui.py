# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Unified user-facing output.

Every message the pipeline prints to the terminal goes through one of
the four functions below.  Internal libraries that need to surface a
warning import :func:`warn`.

Tests can ``monkeypatch.setattr("pengram._ui._handler", ...)`` to
intercept output without touching the real terminal.
"""

from __future__ import annotations

import sys


def _handler(msg: str) -> None:
    print(msg, flush=True)


def _err_handler(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def say(msg: str) -> None:
    """Informational pipeline progress — the default output channel."""
    _handler(msg)


def warn(msg: str) -> None:
    """Non-fatal problem.  Prefixed with ``WARN:`` by convention."""
    _handler(f"  WARN: {msg}")


def step(current: int, total: int, label: str) -> None:
    """Progress tick, e.g. ``Enrichment: 3/10...``"""
    _handler(f"  {label}: {current}/{total}...")


def error(msg: str) -> None:
    """Fatal / user-facing error routed to stderr."""
    _err_handler(f"pengram: error: {msg}")


__all__ = ["say", "warn", "step", "error"]
