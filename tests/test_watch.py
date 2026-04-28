# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.watch — Debouncer and classify_change.

The full observer/watcher integration is deliberately thin: watchdog's
own observer is well-tested upstream, and our glue (the handler
callback + the debouncer + classify_change) is what we actually own.
Those three are tested directly below; the observer seam is exercised
via an in-process stub in the CLI integration.
"""

from __future__ import annotations

from pathlib import Path

from pengram.watch import DEFAULT_DEBOUNCE_SECONDS, Debouncer, classify_change

# ---------------------------------------------------------------------------
# Debouncer
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


def test_debouncer_fires_after_quiet_window() -> None:
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(dt: float) -> None:
        # Advance the fake clock by exactly ``dt`` whenever the fire thread
        # sleeps. After the first iteration the deadline has passed.
        sleeps.append(dt)
        clock.advance(dt)

    fired = {"n": 0}
    d = Debouncer(
        lambda: fired.__setitem__("n", fired["n"] + 1),
        interval=2.0,
        clock=clock,
        sleep=fake_sleep,
    )
    d.schedule()
    d.flush()  # wait for the fire thread to exit
    assert fired["n"] == 1
    # Sleep was bounded at 0.5s — the debouncer doesn't block forever.
    assert all(s <= 0.5 for s in sleeps)


def test_debouncer_coalesces_rapid_events() -> None:
    """Multiple schedule() calls inside the window → one callback."""
    import threading

    clock = _FakeClock()
    gate = threading.Event()

    def fake_sleep(dt: float) -> None:
        gate.wait()
        clock.advance(dt)

    fires: list[float] = []
    d = Debouncer(
        lambda: fires.append(clock.now),
        interval=1.0,
        clock=clock,
        sleep=fake_sleep,
    )
    d.schedule()
    # The fire thread is now blocked on the gate. Push the deadline
    # forward several times while the thread is parked.
    for _ in range(3):
        clock.advance(0.1)
        d.schedule()
    # Release the fire thread — it will see the latest deadline.
    gate.set()
    d.flush()
    assert len(fires) == 1


def test_debouncer_stop_prevents_future_fires() -> None:
    clock = _FakeClock()
    fired = {"n": 0}

    def fake_sleep(dt: float) -> None:
        clock.advance(dt)

    d = Debouncer(
        lambda: fired.__setitem__("n", fired["n"] + 1),
        interval=1.0,
        clock=clock,
        sleep=fake_sleep,
    )
    d.stop()
    d.schedule()  # no-op after stop
    # Give any rogue thread a moment (there shouldn't be one).
    d.flush()
    assert fired["n"] == 0


def test_debouncer_default_interval_constant() -> None:
    assert DEFAULT_DEBOUNCE_SECONDS == 2.0


# ---------------------------------------------------------------------------
# classify_change
# ---------------------------------------------------------------------------


def test_classify_change_code(tmp_path: Path) -> None:
    p = tmp_path / "main.py"
    p.write_text("x")
    assert classify_change(p) == "code"


def test_classify_change_document(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text("x")
    assert classify_change(p) == "content"


def test_classify_change_transcript(tmp_path: Path) -> None:
    p = tmp_path / "clip.srt"
    p.write_text("x")
    assert classify_change(p) == "content"


def test_classify_change_pdf_is_content(tmp_path: Path) -> None:
    """v0.2.0: PDFs are documents, which count as content for watch."""
    p = tmp_path / "book.pdf"
    p.write_bytes(b"\x00")
    assert classify_change(p) == "content"


def test_classify_change_ignores_hidden(tmp_path: Path) -> None:
    p = tmp_path / ".hidden" / "x.py"
    p.parent.mkdir()
    p.write_text("x")
    assert classify_change(p) is None


def test_classify_change_ignores_pengram_cache(tmp_path: Path) -> None:
    p = tmp_path / ".pengram-cache" / "abc.json"
    p.parent.mkdir()
    p.write_text("x")
    assert classify_change(p) is None


def test_classify_change_ignores_pengram_out(tmp_path: Path) -> None:
    p = tmp_path / "pengram-out" / "graph.json"
    p.parent.mkdir()
    p.write_text("x")
    assert classify_change(p) is None


def test_classify_change_ignores_unknown_extensions(tmp_path: Path) -> None:
    p = tmp_path / "foo.xyz"
    p.write_text("x")
    assert classify_change(p) is None


# ---------------------------------------------------------------------------
# watch() wiring — observer factory lets us stay in-process
# ---------------------------------------------------------------------------


def test_watch_schedules_observer_on_root(tmp_path: Path) -> None:
    from pengram.watch import watch as start_watch

    class _StubObserver:
        def __init__(self) -> None:
            self.scheduled: list[tuple[Any, str, bool]] = []

        def schedule(self, handler, path, recursive):
            self.scheduled.append((handler, path, recursive))

    from typing import Any  # local import for the stub's annotation

    stub = _StubObserver()
    observer = start_watch(
        tmp_path,
        rebuild=lambda changed: None,
        observer_factory=lambda: stub,
    )
    assert observer is stub
    assert stub.scheduled
    handler, path, recursive = stub.scheduled[0]
    assert path == str(tmp_path)
    assert recursive is True


# ---------------------------------------------------------------------------
# End-to-end wiring: event → debouncer → rebuild callback
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal watchdog event: src_path + is_directory."""

    def __init__(self, src_path: str, *, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.is_directory = is_directory


def _drive_watch(
    tmp_path: Path,
    exclude: list[Path] = (),
    debounce: float = 0.0,
):
    """Spin up a fake observer, return (handler, rebuild_calls)."""
    from pengram.watch import watch as start_watch

    class _StubObserver:
        def __init__(self) -> None:
            self.handler = None

        def schedule(self, handler, path, recursive):
            self.handler = handler

    stub = _StubObserver()
    rebuild_calls: list[set] = []
    start_watch(
        tmp_path,
        rebuild=lambda changed: rebuild_calls.append(set(changed)),
        exclude=list(exclude),
        debounce=debounce,
        observer_factory=lambda: stub,
    )
    return stub.handler, rebuild_calls


def test_watch_handler_fires_rebuild_for_relevant_change(tmp_path: Path) -> None:
    """An ``on_modified`` on a code file fires the rebuild with the path."""
    handler, rebuild_calls = _drive_watch(tmp_path, debounce=0.0)
    src = tmp_path / "main.py"
    src.write_text("pass")
    handler.on_modified(_FakeEvent(str(src)))
    # Debouncer runs on a daemon thread with interval=0.0; give it a
    # chance to fire. Poll up to ~500ms.
    import time

    for _ in range(50):
        if rebuild_calls:
            break
        time.sleep(0.01)
    assert rebuild_calls, "rebuild should have fired after the change"
    assert src.resolve() in rebuild_calls[0]


def test_watch_handler_ignores_directory_events(tmp_path: Path) -> None:
    handler, rebuild_calls = _drive_watch(tmp_path, debounce=0.0)
    handler.on_modified(_FakeEvent(str(tmp_path), is_directory=True))
    import time

    time.sleep(0.05)
    assert rebuild_calls == []


def test_watch_handler_ignores_unclassified_files(tmp_path: Path) -> None:
    handler, rebuild_calls = _drive_watch(tmp_path, debounce=0.0)
    junk = tmp_path / "weird.xyz"
    junk.write_text("")
    handler.on_modified(_FakeEvent(str(junk)))
    import time

    time.sleep(0.05)
    assert rebuild_calls == []


def test_watch_handler_ignores_paths_under_exclude(tmp_path: Path) -> None:
    """Changes inside the exclude set (typically the output dir) must not
    trigger a rebuild — otherwise writing graph.json would loop forever."""
    excluded = tmp_path / "out"
    excluded.mkdir()
    handler, rebuild_calls = _drive_watch(
        tmp_path,
        exclude=[excluded],
        debounce=0.0,
    )
    inside = excluded / "graph.json"
    inside.write_text("{}")
    handler.on_modified(_FakeEvent(str(inside)))
    import time

    time.sleep(0.05)
    assert rebuild_calls == []


def test_watch_handler_coalesces_burst_into_single_rebuild(
    tmp_path: Path,
) -> None:
    """Three rapid on_modified events inside one debounce window fire once."""
    handler, rebuild_calls = _drive_watch(tmp_path, debounce=0.05)
    a = tmp_path / "a.py"
    a.write_text("x")
    b = tmp_path / "b.py"
    b.write_text("y")
    handler.on_modified(_FakeEvent(str(a)))
    handler.on_modified(_FakeEvent(str(b)))
    handler.on_modified(_FakeEvent(str(a)))
    import time

    for _ in range(100):
        if rebuild_calls:
            break
        time.sleep(0.01)
    assert len(rebuild_calls) == 1
    assert {a.resolve(), b.resolve()} == rebuild_calls[0]
