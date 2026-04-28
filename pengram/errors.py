# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""PENgram exception hierarchy.

Every PENgram-specific error inherits from :class:`PengramError`, so
callers can ``except PengramError`` to catch anything the library raises
without masking unrelated exceptions.

Subclasses also inherit from the stdlib type they replace
(:class:`ValueError`, :class:`RuntimeError`) so existing ``except
ValueError`` / ``except RuntimeError`` handlers keep working.
"""

from __future__ import annotations


class PengramError(Exception):
    """Base exception for all PENgram errors."""


class ConfigError(PengramError, ValueError):
    """Raised when configuration is invalid or incomplete."""


class ExtractionError(PengramError, ValueError):
    """Raised when an extraction dict fails schema validation."""


__all__ = ["PengramError", "ConfigError", "ExtractionError"]
