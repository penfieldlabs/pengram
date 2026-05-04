# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""LLM provider abstraction.

A single :func:`call_llm` function dispatches to the provider configured by
:data:`pengram.config.LLM_PROVIDER`:

- ``claude-cli``  — ``claude -p --model {model} --max-turns 1`` subprocess.
  No API key required. This is the default.
- ``openai``      — OpenAI chat completions API (requires ``OPENAI_API_KEY``).
- ``openrouter``  — OpenRouter endpoint at ``https://openrouter.ai/api/v1``
  via the OpenAI client library (requires ``OPENROUTER_API_KEY``).
- ``ollama``      — local Ollama daemon at
  :data:`pengram.config.OLLAMA_BASE_URL` (default
  ``http://localhost:11434``). No API key — auth is network-local.
  JSON mode is enabled so extraction / linking / enrichment prompts
  get back parseable structured output.

Consumers pass a prompt and a model name; the function returns the response
text. Failures bubble up as :class:`LLMError`.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config as _config
from ._tools import resolve_tool
from .errors import PengramError

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

_OPENAI_HINT = (
    "openai is required for LLM_PROVIDER=openai or openrouter. "
    "Install with: pip install 'pengram[openai]'"
)


class LLMError(PengramError, RuntimeError):
    """Raised when an LLM call fails."""


class TransientLLMError(LLMError):
    """Raised for retriable LLM failures (network, timeout, rate limit)."""


_TRANSIENT_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
)

_RETRY_DELAYS: tuple[int, ...] = (5, 15, 45)


_CLAUDE_HINT = (
    "Install Claude Code (https://github.com/anthropics/claude-code) "
    "or switch LLM_PROVIDER to 'openai', 'openrouter', or 'ollama'."
)


def _call_claude_cli(
    prompt: str,
    model: str,
    timeout: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    claude = resolve_tool("claude", install_hint=_CLAUDE_HINT)
    args = [claude, "-p", "--model", model, "--max-turns", "1", prompt]
    try:
        result = runner(args, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TransientLLMError(f"claude CLI timed out after {timeout}s") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:400]
        raise LLMError(f"claude CLI failed (code {result.returncode}): {stderr}")
    return result.stdout


# ---------------------------------------------------------------------------
# Ollama auto-detection (model + context window)
# ---------------------------------------------------------------------------

# (base_url) → (model_name, chunk_chars_from_context_window)
#
# Cached so auto-detection fires once per process, not per LLM call.
# Lifetime assumption: the loaded Ollama model stays stable for the
# life of the PENgram process. If a ``--watch`` session is running and
# the user stops model X and starts model Y mid-session, subsequent
# rebuilds will keep using X until PENgram restarts. Call
# :func:`_clear_ollama_cache` to force a fresh detection if that ever
# becomes a real workflow.
_OLLAMA_RESOLVED: dict[str, tuple[str, int]] = {}

# Minimum chunk size, even when context window comes back tiny or missing.
_OLLAMA_MIN_CHUNK_CHARS = 2000
# Default context window in tokens when /api/show doesn't expose num_ctx.
_OLLAMA_DEFAULT_CONTEXT_TOKENS = 8192
# Conservative chars-per-token multiplier for chunk sizing.
_OLLAMA_CHARS_PER_TOKEN = 3


def _ollama_get_json(
    url: str,
    *,
    timeout: int = 5,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> Any:
    request = urllib.request.Request(url, method="GET")
    if opener is not None:
        response = opener(request, timeout)
    else:
        response = urllib.request.urlopen(request, timeout=timeout)
    with response as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _ollama_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 5,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if opener is not None:
        response = opener(request, timeout)
    else:
        response = urllib.request.urlopen(request, timeout=timeout)
    with response as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _detect_ollama_model(
    base_url: str,
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
    logger: Callable[[str], None] = print,
) -> str:
    """Pick an Ollama model without asking the user.

    Order of preference:
      1. First currently-loaded model from ``/api/ps``.
      2. First available model from ``/api/tags``.
      3. Raise ``LLMError`` with an actionable install hint.

    Multiple loaded models: use the first and log the choice.
    """
    base = base_url.rstrip("/")

    # 1. Currently-loaded models.
    try:
        data = _ollama_get_json(f"{base}/api/ps", opener=opener)
        running = data.get("models") or []
        if running:
            model = str(running[0].get("name") or "").strip()
            if model:
                if len(running) > 1:
                    logger(f"  Multiple Ollama models loaded, using: {model}")
                return model
    except Exception:
        pass

    # 2. Available models.
    try:
        data = _ollama_get_json(f"{base}/api/tags", opener=opener)
        available = data.get("models") or []
        if available:
            model = str(available[0].get("name") or "").strip()
            if model:
                logger(f"  No Ollama model loaded, using available: {model}")
                return model
    except Exception:
        pass

    raise LLMError("No Ollama models found. Pull one with: ollama pull llama3.1:8b")


def _ollama_context_window(
    base_url: str,
    model: str,
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> int:
    """Query ``/api/show`` and return the model's context window (tokens).

    Falls back to :data:`_OLLAMA_DEFAULT_CONTEXT_TOKENS` (8192 — safe for
    most 7B models) when the endpoint is unreachable or the response
    doesn't expose ``num_ctx``.
    """
    try:
        data = _ollama_post_json(
            f"{base_url.rstrip('/')}/api/show",
            {"name": model},
            opener=opener,
        )
    except Exception:
        return _OLLAMA_DEFAULT_CONTEXT_TOKENS
    params = data.get("parameters") or ""
    if isinstance(params, str):
        for line in params.splitlines():
            line = line.strip()
            if line.startswith("num_ctx"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[-1])
                    except ValueError:
                        break
    # Some Ollama builds expose num_ctx as a top-level integer.
    num_ctx = data.get("num_ctx")
    if isinstance(num_ctx, int) and num_ctx > 0:
        return num_ctx
    return _OLLAMA_DEFAULT_CONTEXT_TOKENS


def _resolve_ollama_model(
    base_url: str | None = None,
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
    logger: Callable[[str], None] = print,
) -> tuple[str, int]:
    """Return ``(model_name, chunk_chars)`` for the Ollama provider.

    Result is cached per ``base_url`` so auto-detection hits the HTTP
    API at most once per process.
    """
    base = (base_url or _config.OLLAMA_BASE_URL).rstrip("/")
    if base in _OLLAMA_RESOLVED:
        return _OLLAMA_RESOLVED[base]
    model = _detect_ollama_model(base, opener=opener, logger=logger)
    ctx_tokens = _ollama_context_window(base, model, opener=opener)
    chunk_chars = max(
        _OLLAMA_MIN_CHUNK_CHARS,
        ctx_tokens * _OLLAMA_CHARS_PER_TOKEN,
    )
    _OLLAMA_RESOLVED[base] = (model, chunk_chars)
    return model, chunk_chars


def _clear_ollama_cache() -> None:
    """Drop the auto-detection cache. Tests call this between scenarios."""
    _OLLAMA_RESOLVED.clear()


def _call_ollama(
    prompt: str,
    model: str,
    timeout: int,
    *,
    base_url: str | None = None,
    temperature: float = 0.0,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> str:
    """Call Ollama's ``/api/generate`` endpoint and return the response text.

    JSON mode is enabled (``format='json'``) so extraction / linking /
    enrichment prompts get back parseable structured output. No API
    key: Ollama auth is network-local. ``opener`` is exposed so tests
    can mock the HTTP boundary without touching the actual service.
    """
    url = (base_url or _config.OLLAMA_BASE_URL).rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        if opener is not None:
            response = opener(request, timeout)
        else:
            response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        raise TransientLLMError(
            f"Ollama request failed: {exc.reason if hasattr(exc, 'reason') else exc}"
        ) from exc
    except TimeoutError as exc:
        raise TransientLLMError(f"Ollama request timed out after {timeout}s") from exc

    with response as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Ollama returned non-JSON envelope: {exc}") from exc
    text = data.get("response")
    if not isinstance(text, str):
        raise LLMError("Ollama response missing 'response' field")
    return text


def _call_openai_compatible(
    prompt: str,
    model: str,
    timeout: int,
    *,
    api_key: str | None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> str:
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_OPENAI_HINT) from exc
    if not api_key:
        raise LLMError("API key missing. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    except Exception as exc:
        cls_name = type(exc).__name__
        if cls_name in _TRANSIENT_EXCEPTION_NAMES:
            raise TransientLLMError(f"{cls_name}: {exc}") from exc
        raise LLMError(f"{cls_name}: {exc}") from exc
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise LLMError("LLM returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return content or ""


def _with_retry(
    fn: Callable[[], str],
    *,
    delays: tuple[int, ...] = _RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Call *fn* up to ``1 + len(delays)`` times, retrying on :class:`TransientLLMError`.

    Permanent errors (:class:`LLMError` that is NOT transient) propagate
    immediately. After exhausting retries the last transient error is
    re-raised.
    """
    last_exc: TransientLLMError | None = None
    for attempt in range(1 + len(delays)):
        try:
            return fn()
        except TransientLLMError as exc:
            last_exc = exc
            if attempt < len(delays):
                sleep(delays[attempt])
    raise last_exc  # type: ignore[misc]


def call_llm(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    _sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Dispatch a prompt to the configured provider and return the text reply.

    ``temperature`` defaults to ``0.0`` to keep extraction, linking, and
    enrichment deterministic across repeat runs. The parameter is honoured
    by the ``openai``, ``openrouter``, and ``ollama`` providers (the
    Ollama payload sets ``options.temperature``). ``claude-cli`` has no
    CLI-level knob for temperature so the value is effectively ignored
    there.
    """
    provider = provider or _config.LLM_PROVIDER
    model = model or _config.LLM.get("extract_model", "haiku")
    timeout = timeout or int(_config.LLM.get("extract_timeout", 300))

    def _dispatch() -> str:
        if provider == "claude-cli":
            return _call_claude_cli(
                prompt,
                model,
                timeout,
                runner=runner if runner is not None else subprocess.run,
            )
        if provider == "openai":
            return _call_openai_compatible(
                prompt,
                model,
                timeout,
                api_key=_config.OPENAI_API_KEY,
                temperature=temperature,
            )
        if provider == "openrouter":
            return _call_openai_compatible(
                prompt,
                model,
                timeout,
                api_key=_config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
            )
        if provider == "ollama":
            return _call_ollama(prompt, model, timeout, temperature=temperature)
        raise LLMError(f"Unknown LLM_PROVIDER: {provider!r}")

    if provider == "ollama" and (not model or model == "_auto"):
        model, _ = _resolve_ollama_model()

    return _with_retry(_dispatch, sleep=_sleep)


_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    suffix = image_path.suffix.lower()
    mime = _MIME_TYPES.get(suffix)
    if mime is None:
        raise LLMError(
            f"unsupported image format {suffix!r}: only {sorted(_MIME_TYPES)} are supported"
        )
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii"), mime


def _call_openai_vision(
    prompt: str,
    image_path: Path,
    model: str,
    timeout: int,
    *,
    api_key: str | None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> str:
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_OPENAI_HINT) from exc
    if not api_key:
        raise LLMError("API key missing. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")
    b64, mime = _encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ],
        }
    ]
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as exc:
        cls_name = type(exc).__name__
        if cls_name in _TRANSIENT_EXCEPTION_NAMES:
            raise TransientLLMError(f"{cls_name}: {exc}") from exc
        raise LLMError(f"{cls_name}: {exc}") from exc
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise LLMError("LLM returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return content or ""


def _call_claude_cli_vision(
    prompt: str,
    image_path: Path,
    model: str,
    timeout: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    claude = resolve_tool("claude", install_hint=_CLAUDE_HINT)
    b64, mime = _encode_image(image_path)
    data_uri = f"data:{mime};base64,{b64}"
    combined = f"{prompt}\n\n[Image: {data_uri}]"
    args = [claude, "-p", "--model", model, "--max-turns", "1"]
    try:
        result = runner(
            args, input=combined, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientLLMError(f"claude CLI timed out after {timeout}s") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:400]
        raise LLMError(f"claude CLI failed (code {result.returncode}): {stderr}")
    return result.stdout


def _call_ollama_vision(
    prompt: str,
    image_path: Path,
    model: str,
    timeout: int,
    *,
    temperature: float = 0.0,
) -> str:
    b64, _mime = _encode_image(image_path)
    base_url = _config.OLLAMA_BASE_URL
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    except urllib.error.URLError as exc:
        raise TransientLLMError(f"Ollama connection failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TransientLLMError(f"Ollama request timed out after {timeout}s") from exc

    with response as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Ollama returned non-JSON envelope: {exc}") from exc
    text = data.get("response")
    if not isinstance(text, str):
        raise LLMError("Ollama response missing 'response' field")
    return text


def call_llm_vision(
    prompt: str,
    image_path: Path,
    *,
    model: str | None = None,
    timeout: int | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    _sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Send a prompt with an image to a vision-capable LLM."""
    provider = provider or _config.LLM_PROVIDER
    model = model or _config.LLM.get("image_model", "sonnet")
    timeout = timeout or int(_config.LLM.get("extract_timeout", 300))

    if provider == "ollama" and (not model or model == "_auto"):
        model, _ = _resolve_ollama_model()

    def _dispatch() -> str:
        if provider == "claude-cli":
            return _call_claude_cli_vision(
                prompt,
                image_path,
                model,  # type: ignore[arg-type]
                timeout,
                runner=runner if runner is not None else subprocess.run,
            )
        if provider == "openai":
            return _call_openai_vision(
                prompt,
                image_path,
                model,  # type: ignore[arg-type]
                timeout,
                api_key=_config.OPENAI_API_KEY,
                temperature=temperature,
            )
        if provider == "openrouter":
            return _call_openai_vision(
                prompt,
                image_path,
                model,  # type: ignore[arg-type]
                timeout,
                api_key=_config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
            )
        if provider == "ollama":
            return _call_ollama_vision(
                prompt,
                image_path,
                model,
                timeout,
                temperature=temperature,  # type: ignore[arg-type]
            )
        raise LLMError(f"Unknown LLM_PROVIDER: {provider!r}")

    return _with_retry(_dispatch, sleep=_sleep)


def parse_json_response(text: str) -> object:
    """Best-effort JSON extraction from an LLM response.

    Strips markdown fences (```json ... ``` or ``` ... ```) before parsing.
    Raises :class:`LLMError` on unrecoverable parse failure.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence and optional language tag.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        # Try to locate the first JSON object/array in the text.
        for start, end_char in (("{", "}"), ("[", "]")):
            i = stripped.find(start)
            j = stripped.rfind(end_char)
            if 0 <= i < j:
                try:
                    return json.loads(stripped[i : j + 1])
                except json.JSONDecodeError:
                    continue
        raise LLMError(f"LLM response was not valid JSON: {exc}") from exc


__all__ = [
    "LLMError",
    "TransientLLMError",
    "call_llm",
    "call_llm_vision",
    "parse_json_response",
]
