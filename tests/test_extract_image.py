# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.extract_image."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pengram.extract_image import (
    IMAGE_EXTRACT_PROMPT,
    extract_image,
    extract_images,
)


def _fake_vision_llm(prompt: str, image_path: Path, **kwargs: Any) -> str:
    return json.dumps(
        {
            "concepts": [
                {"name": "Neural Network", "mentions": 1, "note": "Diagram of a neural network"},
            ],
            "summary": "An architecture diagram showing a neural network.",
        }
    )


def _failing_vision_llm(prompt: str, image_path: Path, **kwargs: Any) -> str:
    from pengram.llm import LLMError

    raise LLMError("vision model unavailable")


def test_extract_image_returns_expected_keys(tmp_path: Path) -> None:
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    result = extract_image(img, llm=_fake_vision_llm)
    assert result["_doc_id"] == "image:test.png"
    assert result["_source"] == str(img)
    assert len(result["concepts"]) == 1
    assert result["concepts"][0]["name"] == "Neural Network"
    assert result["summary"]


def test_extract_image_prompt_mentions_json() -> None:
    assert "JSON" in IMAGE_EXTRACT_PROMPT
    assert "concepts" in IMAGE_EXTRACT_PROMPT


def test_extract_images_caches_results(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "out"

    results = extract_images(
        [img],
        output_dir=out,
        llm=_fake_vision_llm,
    )
    assert len(results) == 1
    dump = out / "extractions" / "image_a.png.json"
    assert dump.exists()
    data = json.loads(dump.read_text())
    assert "_doc_id" not in data
    assert "_source" not in data
    assert "concepts" in data


def test_extract_images_skips_failures(tmp_path: Path) -> None:
    img = tmp_path / "bad.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "out"

    results = extract_images(
        [img],
        output_dir=out,
        llm=_failing_vision_llm,
    )
    assert results == []


def test_extract_images_empty_input(tmp_path: Path) -> None:
    out = tmp_path / "out"
    results = extract_images([], output_dir=out, llm=_fake_vision_llm)
    assert results == []


def test_extract_images_content_cache_hit(tmp_path: Path) -> None:
    img = tmp_path / "cached.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    out = tmp_path / "out"

    call_count = 0
    original = _fake_vision_llm

    def counting_llm(*args: Any, **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    extract_images([img], output_dir=out, cache_root=tmp_path, llm=counting_llm)
    assert call_count == 1

    call_count = 0
    extract_images([img], output_dir=out, cache_root=tmp_path, llm=counting_llm)
    assert call_count == 0


def test_extract_images_sorted_by_doc_id(tmp_path: Path) -> None:
    for name in ("c.png", "a.png", "b.png"):
        (tmp_path / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    out = tmp_path / "out"

    results = extract_images(
        [tmp_path / "c.png", tmp_path / "a.png", tmp_path / "b.png"],
        output_dir=out,
        llm=_fake_vision_llm,
    )
    ids = [r["_doc_id"] for r in results]
    assert ids == sorted(ids)
