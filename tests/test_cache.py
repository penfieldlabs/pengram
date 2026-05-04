# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.cache."""

from __future__ import annotations

from pathlib import Path

from pengram import cache


def test_load_cached_miss(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    assert cache.load_cached(tmp_path, f) is None


def test_save_then_load(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    cache.save_cached(tmp_path, f, {"nodes": [{"id": "x"}], "edges": []})
    got = cache.load_cached(tmp_path, f)
    assert got is not None
    assert got["nodes"] == [{"id": "x"}]
    assert got["edges"] == []


def test_changed_file_is_cache_miss(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    cache.save_cached(tmp_path, f, {"version": 1})
    f.write_text("x = 2")
    assert cache.load_cached(tmp_path, f) is None


def test_split_cached(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("a = 1")
    b = tmp_path / "b.py"
    b.write_text("b = 2")
    c = tmp_path / "c.py"
    c.write_text("c = 3")
    cache.save_cached(tmp_path, a, {"node": "a"})
    cache.save_cached(tmp_path, b, {"node": "b"})
    cached_pairs, uncached = cache.split_cached(tmp_path, [a, b, c])
    cached_paths = {p for p, _ in cached_pairs}
    assert cached_paths == {a, b}
    assert uncached == [c]


def test_clear_cache_empty(tmp_path: Path) -> None:
    assert cache.clear_cache(tmp_path) == 0


def test_clear_cache_removes_entries(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("a = 1")
    b = tmp_path / "b.py"
    b.write_text("b = 2")
    cache.save_cached(tmp_path, a, {"n": 1})
    cache.save_cached(tmp_path, b, {"n": 2})
    removed = cache.clear_cache(tmp_path)
    assert removed == 2
    assert cache.load_cached(tmp_path, a) is None


def test_save_handles_missing_source(tmp_path: Path) -> None:
    # Hashing a non-existent file raises; load returns None safely.
    ghost = tmp_path / "ghost.py"
    assert cache.load_cached(tmp_path, ghost) is None


def test_cache_corrupt_json_returns_none(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("a = 1")
    cache.save_cached(tmp_path, f, {"ok": True})
    # Corrupt the saved cache entry
    cdir = tmp_path / cache.CACHE_DIR_NAME
    for p in cdir.iterdir():
        p.write_text("not json{")
    assert cache.load_cached(tmp_path, f) is None


def test_cache_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("same")
    b = tmp_path / "b.py"
    b.write_text("same")
    cache.save_cached(tmp_path, a, {"origin": "a"})
    # Same content → same cache key, so we read a's stored result for b
    assert cache.load_cached(tmp_path, b) == {"origin": "a"}


def test_content_hash_cache_excludes_underscore_keys(tmp_path: Path) -> None:
    src = tmp_path / "x.md"
    src.write_text("x" * 100)
    cache.save_cached(tmp_path, src, {"_source": "/secret/x.md", "_doc_id": "x.md", "concepts": []})
    cached_file = next(p for p in (tmp_path / ".pengram-cache").iterdir())
    text = cached_file.read_text()
    assert "/secret" not in text
    assert "_source" not in text
    assert "_doc_id" not in text
    assert "concepts" in text
