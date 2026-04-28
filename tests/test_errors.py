# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.errors — the exception hierarchy."""

from __future__ import annotations

import pytest

from pengram.errors import ConfigError, ExtractionError, PengramError
from pengram.llm import LLMError
from pengram.security import SecurityError


def test_all_errors_inherit_from_pengram_error() -> None:
    for cls in (ConfigError, ExtractionError, SecurityError, LLMError):
        assert issubclass(cls, PengramError)


def test_config_error_is_also_value_error() -> None:
    with pytest.raises(ValueError):
        raise ConfigError("bad config")


def test_extraction_error_is_also_value_error() -> None:
    with pytest.raises(ValueError):
        raise ExtractionError("bad extraction")


def test_security_error_is_also_value_error() -> None:
    with pytest.raises(ValueError):
        raise SecurityError("bad input")


def test_llm_error_is_also_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        raise LLMError("llm failed")


def test_catch_all_pengram_errors() -> None:
    for cls in (ConfigError, ExtractionError, SecurityError, LLMError):
        with pytest.raises(PengramError):
            raise cls("test")
