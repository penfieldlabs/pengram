# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for vision LLM functions in pengram.llm."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pengram.llm import LLMError, _encode_image, call_llm_vision


def _make_png(tmp_path: Path) -> Path:
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return img


def test_encode_image_returns_base64_and_mime(tmp_path: Path) -> None:
    img = _make_png(tmp_path)
    b64, mime = _encode_image(img)
    assert mime == "image/png"
    assert len(b64) > 0
    import base64

    decoded = base64.b64decode(b64)
    assert decoded[:4] == b"\x89PNG"


def test_encode_image_jpeg_mime(tmp_path: Path) -> None:
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    _, mime = _encode_image(img)
    assert mime == "image/jpeg"


def test_call_llm_vision_claude_cli(tmp_path: Path) -> None:
    img = _make_png(tmp_path)
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='{"concepts": []}', stderr=""
    )
    captured_kwargs: dict[str, Any] = {}

    def fake_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_kwargs.update(kwargs)
        return fake_result

    with (
        patch("pengram.config.LLM_PROVIDER", "claude-cli"),
        patch("pengram.llm.resolve_tool", return_value="claude"),
    ):
        result = call_llm_vision(
            "extract entities",
            img,
            model="haiku",
            provider="claude-cli",
            runner=fake_runner,
        )
    assert '{"concepts": []}' in result
    assert "input" in captured_kwargs, "vision prompt must be piped via stdin, not CLI args"
    assert "data:image/png;base64," in captured_kwargs["input"]


def test_encode_image_rejects_unsupported_format(tmp_path: Path) -> None:
    svg = tmp_path / "file.svg"
    svg.write_text("<svg></svg>")
    with pytest.raises(LLMError, match="unsupported image format"):
        _encode_image(svg)


def test_call_llm_vision_unknown_provider(tmp_path: Path) -> None:
    img = _make_png(tmp_path)
    with pytest.raises(LLMError, match="Unknown"):
        call_llm_vision("test", img, provider="nonexistent")


def test_call_llm_vision_default_model_is_image_model(tmp_path: Path) -> None:
    img = _make_png(tmp_path)
    captured_model = None

    def capture_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal captured_model
        captured_model = args[0][3] if len(args[0]) > 3 else None
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="{}", stderr="")

    with (
        patch("pengram.config.LLM_PROVIDER", "claude-cli"),
        patch("pengram.config.LLM", {"image_model": "test-vision-model", "extract_timeout": 300}),
        patch("pengram.llm.resolve_tool", return_value="claude"),
    ):
        call_llm_vision("test", img, provider="claude-cli", runner=capture_runner)
    assert captured_model == "test-vision-model"
