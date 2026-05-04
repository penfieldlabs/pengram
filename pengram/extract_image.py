# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Vision-LLM extraction for image files.

Sends each image to a vision-capable model and extracts entities using
the same ``{concepts, summary}`` schema as :mod:`extract_llm`.  Results
feed into the same graph-building pipeline — images become first-class
source documents in the knowledge graph.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import cache as _cache
from . import config as _config
from ._ui import say as _ui_say
from ._ui import step as _ui_step
from ._ui import warn as _ui_warn
from .llm import LLMError, call_llm_vision, parse_json_response
from .security import sanitize_filename

IMAGE_EXTRACT_PROMPT = """\
You are an entity extractor. Examine the image below and produce STRICT JSON
with this schema:

{
  "concepts": [{"name": "...", "mentions": 1, "note": "..."}],
  "summary": "one paragraph describing what this image shows"
}

Rules:
- Extract concepts: ideas, technologies, theories, methods, mechanisms,
  substances, practices, diagrams, charts, data points, labels —
  anything that represents a discrete, nameable piece of knowledge
  visible in the image.
- For diagrams, charts, and infographics: extract the entities and
  relationships depicted, not just the title.
- For screenshots of text: extract the key concepts from the text content.
- For photos: extract the subjects, objects, and context visible.
- Use canonical names ("OpenAI", not "open ai").
- mentions is 1 for each concept (single image = single occurrence).
- note should describe the concept's role or appearance in this image.
- Prefer precision over recall: do not invent entities.
- Respond with JSON only, no commentary, no markdown fences.
"""


def _extraction_path(output_dir: Path, doc_id: str) -> Path:
    return output_dir / "extractions" / f"{sanitize_filename(doc_id)}.json"


def extract_image(
    image_path: Path,
    *,
    model: str | None = None,
    timeout: int | None = None,
    llm: Callable[..., str] | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Extract entities from a single image via a vision-capable LLM."""
    caller = llm if llm is not None else call_llm_vision
    response = caller(
        IMAGE_EXTRACT_PROMPT,
        image_path,
        model=model or _config.LLM["image_model"],
        timeout=timeout or _config.LLM["extract_timeout"],
        provider=provider,
    )
    data = parse_json_response(response)
    if not isinstance(data, dict):
        raise LLMError("image extraction response was not a JSON object")
    doc_id = f"image:{image_path.name}"
    data["_doc_id"] = doc_id
    data["_source"] = str(image_path)
    return data


def extract_images(
    image_paths: Iterable[Path],
    *,
    output_dir: Path,
    cache_root: Path | None = None,
    model: str | None = None,
    timeout: int | None = None,
    workers: int | None = None,
    llm: Callable[..., str] | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Extract entities from many images with caching and parallelism."""
    output_dir = Path(output_dir)
    (output_dir / "extractions").mkdir(parents=True, exist_ok=True)
    workers = workers or int(_config.LLM.get("parallel_workers", 4))
    caller = llm if llm is not None else call_llm_vision

    def _dump_extraction(doc_id: str, result: dict[str, Any]) -> None:
        try:
            clean = {k: v for k, v in result.items() if not k.startswith("_")}
            _extraction_path(output_dir, doc_id).write_text(
                json.dumps(clean, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _run(img: Path) -> dict[str, Any]:
        doc_id = f"image:{img.name}"
        has_content_cache = cache_root is not None and img.exists()

        if has_content_cache:
            cached = _cache.load_cached(cache_root, img)
            if isinstance(cached, dict):
                cached["_doc_id"] = doc_id
                cached["_source"] = str(img)
                _dump_extraction(doc_id, cached)
                return cached
        else:
            dump = _extraction_path(output_dir, doc_id)
            if dump.exists():
                try:
                    cached = json.loads(dump.read_text(encoding="utf-8"))
                    cached["_doc_id"] = doc_id
                    cached["_source"] = str(img)
                    return cached
                except (OSError, json.JSONDecodeError):
                    pass

        result = extract_image(
            img,
            model=model,
            timeout=timeout,
            llm=caller,
            provider=provider,
        )
        _dump_extraction(doc_id, result)

        if has_content_cache:
            _cache.save_cached(cache_root, img, result)  # type: ignore[arg-type]
        return result

    paths = list(image_paths)
    if not paths:
        return []

    results: list[dict[str, Any]] = []
    total = len(paths)
    done = 0
    _ui_say(f"  Extracting entities from {total} image(s)…")

    with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
        futures = {pool.submit(_run, p): p for p in paths}
        for future in as_completed(futures):
            done += 1
            img = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                _ui_warn(f"image extraction failed for {img.name}: {exc}")
            if total >= 10 and done % max(1, total // 10) == 0:
                _ui_step(done, total, "images")

    return sorted(results, key=lambda r: r.get("_doc_id", ""))


__all__ = ["extract_image", "extract_images", "IMAGE_EXTRACT_PROMPT"]
