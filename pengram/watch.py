# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""File-watching rebuild loop.

Event-based: uses ``watchdog`` (inotify on Linux, FSEvents on macOS).
Debounces bursts of filesystem events — a typical editor save fires a
handful of modifications in quick succession and we don't want to
rebuild for every one. Default debounce: 2 seconds after the last
event.

Code file changes re-run the full pipeline; AST extraction is cheap so
this is fine. Content file changes hit the content-hash cache for
anything unchanged, so only the edited files round-trip through the
LLM.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from . import detect as _detect
from ._ui import warn as _ui_warn

_WATCHDOG_INSTALL_HINT = (
    "watchdog is required for --watch mode. Install with: pip install 'pengram[watch]'"
)

DEFAULT_DEBOUNCE_SECONDS: float = 2.0

_IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pengram-cache",
        "pengram-out",
        ".pytest_cache",
        ".mypy_cache",
    }
)


class Debouncer:
    """Collects events and fires a callback once the stream goes quiet.

    ``schedule()`` resets the quiet-period timer. Multiple calls inside
    the debounce window coalesce into one ``callback()`` invocation
    after the stream has been idle for ``interval`` seconds.

    Thread-safe: the scheduler can be called from any watchdog thread;
    the callback runs on its own fire thread.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        interval: float = DEFAULT_DEBOUNCE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._callback = callback
        self._interval = interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._deadline: float | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False

    def schedule(self) -> None:
        """Record an event and start (or refresh) the fire timer."""
        if self._stopped:
            return
        with self._lock:
            self._deadline = self._clock() + self._interval
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="pengram-watch-debounce",
                    daemon=True,
                )
                self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            self._deadline = None

    def flush(self) -> None:
        """Block until the fire thread has exited (useful in tests)."""
        thread = self._thread
        if thread is not None:
            thread.join()

    def _run(self) -> None:
        while not self._stopped:
            with self._lock:
                deadline = self._deadline
            if deadline is None:
                return
            now = self._clock()
            if now >= deadline:
                with self._lock:
                    self._deadline = None
                try:
                    self._callback()
                except Exception as exc:  # pragma: no cover — reported to user
                    _ui_warn(f"watch callback raised: {exc}")
                return
            self._sleep(min(deadline - now, 0.5))


def classify_change(path: Path) -> str | None:
    """Return ``"code"``, ``"content"``, or ``None`` for a changed path.

    ``None`` means "not interesting" — hidden dirs, noise directories,
    unclassifiable files. ``code`` and ``content`` distinguish which
    re-extraction shortcut the watcher can take (code is AST-only, so
    cheap; content needs the LLM path, which the content-hash cache
    still makes incremental).
    """
    if any(part.startswith(".") for part in path.parts):
        return None
    if any(p in _IGNORED_DIR_NAMES for p in path.parts):
        return None
    ftype = _detect.classify(path)
    if ftype is None:
        return None
    if ftype is _detect.FileType.CODE:
        return "code"
    if ftype in (
        _detect.FileType.DOCUMENT,
        _detect.FileType.TRANSCRIPT,
    ):
        return "content"
    return None


def _require_watchdog():
    try:
        from watchdog.events import FileSystemEventHandler  # type: ignore[import-not-found]
        from watchdog.observers import Observer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_WATCHDOG_INSTALL_HINT) from exc
    return FileSystemEventHandler, Observer


def watch(
    root: Path,
    rebuild: Callable[[set[Path]], None],
    *,
    debounce: float = DEFAULT_DEBOUNCE_SECONDS,
    exclude: Iterable[Path] = (),
    observer_factory: Callable[[], Any] | None = None,
) -> Any:
    """Start a watchdog observer on ``root`` and return it (unstarted).

    The caller is responsible for calling ``observer.start()`` and
    ``observer.stop()``. Exposed at this seam so CLI integration tests
    can drive the observer directly without needing real file events.

    When a relevant file change is seen, ``rebuild(changed_paths)`` is
    called once the debounce window elapses. ``changed_paths`` is the
    set of absolute paths that changed during this window (reset each
    fire).
    """
    FileSystemEventHandler, Observer = _require_watchdog()
    excluded_roots = {Path(p).resolve() for p in exclude}

    pending: set[Path] = set()
    lock = threading.Lock()

    def _fire() -> None:
        with lock:
            batch = set(pending)
            pending.clear()
        if batch:
            rebuild(batch)

    debouncer = Debouncer(_fire, interval=debounce)

    def _event_path(event) -> Path | None:
        p = Path(event.src_path).resolve()
        # Stay out of our own output directory, etc.
        for excluded in excluded_roots:
            try:
                p.relative_to(excluded)
                return None
            except ValueError:
                continue
        return p

    class _Handler(FileSystemEventHandler):
        def _handle(self, event) -> None:
            if getattr(event, "is_directory", False):
                return
            path = _event_path(event)
            if path is None:
                return
            if classify_change(path) is None:
                return
            with lock:
                pending.add(path)
            debouncer.schedule()

        def on_created(self, event):
            self._handle(event)

        def on_modified(self, event):
            self._handle(event)

        def on_moved(self, event):
            self._handle(event)

        def on_deleted(self, event):
            # Deletes still matter — the rebuild will drop the node.
            self._handle(event)

    observer = (observer_factory or Observer)()
    observer.schedule(_Handler(), str(root), recursive=True)
    # Attach the debouncer so callers can flush/stop it alongside the observer.
    observer._pengram_debouncer = debouncer  # type: ignore[attr-defined]
    return observer


__all__ = [
    "DEFAULT_DEBOUNCE_SECONDS",
    "Debouncer",
    "classify_change",
    "watch",
]
