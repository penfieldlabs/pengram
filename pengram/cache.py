# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""SHA256 incremental caching for extraction results.

Cache files live under ``<root>/.pengram-cache/`` as ``{sha256}.json``.
Only the file's content hash determines the cache key — renames and moves
hit the same cache entry, which is intentional.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR_NAME = ".pengram-cache"
_CHUNK_SIZE = 1 << 20  # 1 MiB


def _cache_dir(root: Path) -> Path:
    return root / CACHE_DIR_NAME


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cache_path(root: Path, path: Path) -> Path:
    return _cache_dir(root) / f"{_hash_file(path)}.json"


def load_cached(root: Path, path: Path) -> dict[str, Any] | None:
    """Return cached extraction result for ``path`` or ``None`` on miss."""
    try:
        cpath = _cache_path(root, path)
    except OSError:
        return None
    if not cpath.exists():
        return None
    try:
        with open(cpath, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_cached(root: Path, path: Path, result: dict[str, Any]) -> None:
    """Persist ``result`` as the cached extraction for ``path``.

    Cache write failures are non-fatal: this function swallows OSError so
    pipeline runs continue even if the cache dir is not writable.
    """
    try:
        cdir = _cache_dir(root)
        cdir.mkdir(parents=True, exist_ok=True)
        cpath = _cache_path(root, path)
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, sort_keys=True)
    except OSError:
        return


def split_cached(
    root: Path, paths: list[Path]
) -> tuple[list[tuple[Path, dict[str, Any]]], list[Path]]:
    """Partition ``paths`` into cached results and uncached paths."""
    cached_pairs: list[tuple[Path, dict[str, Any]]] = []
    uncached: list[Path] = []
    for p in paths:
        result = load_cached(root, p)
        if result is None:
            uncached.append(p)
        else:
            cached_pairs.append((p, result))
    return cached_pairs, uncached


def clear_cache(root: Path) -> int:
    """Delete all cache entries under ``root``. Returns number of files removed."""
    cdir = _cache_dir(root)
    if not cdir.exists():
        return 0
    removed = 0
    for entry in cdir.iterdir():
        if entry.is_file():
            try:
                entry.unlink()
                removed += 1
            except OSError:
                continue
    return removed


__all__ = [
    "CACHE_DIR_NAME",
    "load_cached",
    "save_cached",
    "split_cached",
    "clear_cache",
]
