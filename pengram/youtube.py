# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""YouTube channel pipeline — catalog and transcript pulling.

Uses yt-dlp for both catalog enumeration (``--flat-playlist``) and subtitle
extraction. Transcripts come from captions; audio is *not* downloaded here.
For audio-only workflows, feed the resulting media files through
:mod:`pengram.transcribe`.

No cookies or authentication tokens are used. The pipeline relies solely on
publicly available metadata and captions.

All network interaction is funnelled through a :class:`YouTubeClient` so tests
can inject a fake.
"""

from __future__ import annotations

import html as _html
import json
import re
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as _config
from ._tools import resolve_tool
from ._ui import step as _ui_step
from ._ui import warn as _ui_warn
from .config import YouTubeChannel
from .security import SecurityError, validate_url, validate_video_id

_YT_DLP_HINT = (
    "yt-dlp is required for the YouTube pipeline. Install with: pip install 'pengram[youtube]'"
)


@dataclass
class VideoMeta:
    """Minimal per-video metadata collected from a channel catalog."""

    video_id: str
    title: str
    url: str
    channel: str
    tab: str  # "videos" | "streams" | "shorts"
    duration: int | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    upload_date: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (flattens ``extra`` into the top level)."""
        return {
            "video_id": self.video_id,
            "title": self.title,
            "url": self.url,
            "channel": self.channel,
            "tab": self.tab,
            "duration": self.duration,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "upload_date": self.upload_date,
            **self.extra,
        }


@dataclass
class TranscriptResult:
    # status is one of: "ok" | "no_subtitles" | "empty_after_cleaning"
    # | "rate_limited" | "timeout" | "error"
    video_id: str
    text: str | None
    status: str
    error: str | None = None

    @property
    def is_transient(self) -> bool:
        """Return True when the failure is worth retrying after a cooldown."""
        return self.status in {"rate_limited", "timeout"}


class YouTubeClient:
    """Thin wrapper around the yt-dlp CLI, injectable in tests."""

    _YT_DLP_HINT = (
        "yt-dlp is required for the YouTube pipeline. Install with: pip install 'pengram[youtube]'"
    )

    def __init__(
        self,
        *,
        proxy: str | None = None,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.proxy = proxy
        self._custom_runner = runner is not None
        self._runner = runner or self._default_runner

    @staticmethod
    def _default_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _base_args(self) -> list[str]:
        if self._custom_runner:
            cmd = "yt-dlp"
        else:
            cmd = resolve_tool("yt-dlp", install_hint=self._YT_DLP_HINT)
        args = [cmd]
        if self.proxy:
            args += ["--proxy", self.proxy]
        return args

    def flat_playlist(self, tab_url: str) -> list[dict[str, Any]]:
        """Return the flat playlist entries (one dict per video) for a tab URL."""
        result = self._runner(
            self._base_args() + ["--flat-playlist", "--dump-json", "--skip-download", tab_url]
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed on {tab_url}: {result.stderr.strip()[:200]}")
        entries: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def fetch_subtitles(self, video_id: str, out_dir: Path) -> Path | None:
        """Download English auto-generated subtitles for a video.

        Returns the .vtt path or None. Single yt-dlp call with
        ``--write-auto-subs`` and ``--sub-langs en`` (exact match).
        Auto-generated captions are available on virtually all videos
        and sufficient for knowledge extraction.
        """
        validate_video_id(video_id)
        url = f"https://www.youtube.com/watch?v={video_id}"
        out_template = str(out_dir / "%(id)s.%(ext)s")
        result = self._runner(
            self._base_args()
            + [
                "--write-auto-subs",
                "--sub-langs",
                "en",
                "--sub-format",
                "vtt",
                "--skip-download",
                "-o",
                out_template,
                url,
            ]
        )
        for candidate in out_dir.glob(f"{video_id}*.vtt"):
            return candidate
        if result.returncode != 0:
            stderr = (result.stderr or "").lower()
            if "sign in" in stderr or "429" in stderr:
                raise RuntimeError("rate_limited: " + (result.stderr or "")[:200])
        return None


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------


def _tab_url(channel_url: str, tab: str) -> str:
    url = channel_url.rstrip("/")
    return f"{url}/{tab}"


def pull_catalog(
    channel: YouTubeChannel,
    *,
    client: YouTubeClient | None = None,
) -> list[VideoMeta]:
    """Enumerate videos for every configured tab on ``channel``."""
    validate_url(channel.url)
    client = client or YouTubeClient(proxy=_config.PROXY)
    catalog: list[VideoMeta] = []
    for tab in channel.tabs:
        if tab not in _config.VALID_YOUTUBE_TABS:
            continue
        try:
            entries = client.flat_playlist(_tab_url(channel.url, tab))
        except RuntimeError as exc:
            if "does not have a" in str(exc):
                _ui_warn(f"Channel {channel.label} has no {tab} tab — skipping")
                continue
            raise
        for entry in entries:
            vid = entry.get("id") or entry.get("video_id")
            if not vid:
                continue
            try:
                validate_video_id(vid)
            except SecurityError:
                continue
            catalog.append(
                VideoMeta(
                    video_id=vid,
                    title=entry.get("title") or vid,
                    url=entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                    channel=channel.label,
                    tab=tab,
                    duration=entry.get("duration"),
                    views=entry.get("view_count"),
                    likes=entry.get("like_count"),
                    comments=entry.get("comment_count"),
                    upload_date=entry.get("upload_date"),
                )
            )
    return catalog


# ---------------------------------------------------------------------------
# Transcript handling
# ---------------------------------------------------------------------------

_VTT_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->")
_VTT_CUE_RE = re.compile(r"^(WEBVTT|NOTE|STYLE|Kind:|Language:)", re.IGNORECASE)
_VTT_TAG_RE = re.compile(r"</?[^>]+>")


def clean_vtt(text: str) -> str:
    """Strip timestamps, cue metadata, and HTML tags from VTT captions."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _VTT_CUE_RE.match(line):
            continue
        if _VTT_TIMESTAMP_RE.match(line):
            continue
        if line.isdigit():  # cue numbers
            continue
        line = _VTT_TAG_RE.sub("", line).strip()
        line = _html.unescape(line)
        if line.startswith(">>"):
            line = line.lstrip(">").strip()
            if out:
                out.append("")
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return "\n".join(out)


def _format_frontmatter(meta: VideoMeta) -> str:
    """Return YAML frontmatter block for a transcript file."""
    title_escaped = meta.title.replace('"', '\\"')
    lines = [
        "---",
        f"video_id: {meta.video_id}",
        f"channel: {meta.channel}",
        f'title: "{title_escaped}"',
        f"url: {meta.url}",
        f'upload_date: "{meta.upload_date}"' if meta.upload_date else "upload_date: NA",
        f"views: {meta.views or 'NA'}",
        f"likes: {meta.likes or 'NA'}",
        f"comments: {meta.comments or 'NA'}",
        f"duration: {meta.duration or 'NA'}",
        f"tab: {meta.tab}",
        "---",
        "",
    ]
    return "\n".join(lines)


def pull_transcript(
    meta: VideoMeta,
    *,
    client: YouTubeClient | None = None,
    work_dir: Path | None = None,
) -> TranscriptResult:
    """Download and clean the English transcript for a video.

    When *meta* carries catalog metadata the saved ``.transcript`` file is
    prefixed with YAML frontmatter (video_id, title, views, …).

    The cleaned text is saved as ``{video_id}.transcript`` and the raw
    ``.vtt`` is deleted. If the ``.transcript`` already exists, the download
    is skipped.
    """
    video_id = meta.video_id
    validate_video_id(video_id)
    client = client or YouTubeClient(proxy=_config.PROXY)
    work_dir = work_dir or (_config.OUTPUT_DIR / "transcripts")
    work_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = work_dir / f"{video_id}.transcript"
    if transcript_path.exists():
        text = transcript_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return TranscriptResult(video_id, text, "ok")

    try:
        vtt = client.fetch_subtitles(video_id, work_dir)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "rate_limited" in msg:
            return TranscriptResult(video_id, None, "rate_limited", str(exc))
        if "timeout" in msg or "timed out" in msg:
            return TranscriptResult(video_id, None, "timeout", str(exc))
        return TranscriptResult(video_id, None, "error", str(exc))
    except subprocess.TimeoutExpired as exc:
        return TranscriptResult(video_id, None, "timeout", str(exc))

    if vtt is None or not vtt.exists():
        return TranscriptResult(video_id, None, "no_subtitles")
    try:
        raw = vtt.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return TranscriptResult(video_id, None, "error", str(exc))

    cleaned = clean_vtt(raw)
    if not cleaned.strip():
        vtt.unlink(missing_ok=True)
        return TranscriptResult(video_id, None, "empty_after_cleaning")

    content = _format_frontmatter(meta) + cleaned
    transcript_path.write_text(content, encoding="utf-8")
    vtt.unlink(missing_ok=True)
    return TranscriptResult(video_id, content, "ok")


def parse_transcript_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from a transcript file's text.

    Returns an empty dict when frontmatter is absent or unparseable.
    """
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    import yaml

    try:
        return yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        return {}


def strip_frontmatter(text: str) -> str:
    """Return *text* with leading YAML frontmatter removed."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + 5 :]


def load_video_meta(work_dir: Path, video_id: str) -> dict[str, Any] | None:
    """Read video metadata from the transcript's YAML frontmatter.

    Falls back to legacy ``.meta.json`` sidecar if frontmatter is absent.
    """
    for ext in (".transcript", ".txt"):
        candidate = work_dir / f"{video_id}{ext}"
        if candidate.exists():
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
                fm = parse_transcript_frontmatter(text)
                if fm:
                    return fm
            except OSError:
                pass
    sidecar = work_dir / f"{video_id}.meta.json"
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError):
            pass
    return None


_RETRY_DELAYS: tuple[float, ...] = (5.0, 15.0, 45.0)


def pull_all_transcripts(
    catalog: Iterable[VideoMeta],
    *,
    client: YouTubeClient | None = None,
    work_dir: Path | None = None,
    state_file: Path | None = None,
    sleep_between: float = 2.0,
    retry_transient: bool = True,
    max_videos: int | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, TranscriptResult]:
    """Pull transcripts for every video, resuming from ``state_file`` if present.

    Permanent failures (``no_subtitles``, ``empty_after_cleaning``) are
    recorded and never retried. Transient failures (``rate_limited``,
    ``timeout``) are retried up to three times with capped exponential
    backoff (5 s / 15 s / 45 s) when ``retry_transient`` is true.

    ``max_videos`` caps how many *new* videos are fetched in a single run
    (already-succeeded entries in ``state_file`` do not count).
    """
    client = client or YouTubeClient(proxy=_config.PROXY)
    work_dir = work_dir or (_config.OUTPUT_DIR / "transcripts")
    work_dir.mkdir(parents=True, exist_ok=True)
    items = list(catalog)
    total = len(items)
    state: dict[str, TranscriptResult] = {}
    if state_file and state_file.exists():
        try:
            raw = json.loads(state_file.read_text(encoding="utf-8"))
            for vid, payload in raw.items():
                status = payload.get("status", "error")
                if status in {"rate_limited", "error"}:
                    transcript = work_dir / f"{vid}.transcript"
                    status = "ok" if transcript.exists() else "no_subtitles"
                state[vid] = TranscriptResult(
                    video_id=vid,
                    text=payload.get("text"),
                    status=status,
                    error=payload.get("error"),
                )
        except (OSError, json.JSONDecodeError) as exc:
            _ui_warn(f"transcript state file unreadable, starting fresh: {exc}")
            state = {}

    def _persist() -> None:
        if not state_file:
            return
        try:
            payload = {
                vid: {
                    "status": r.status,
                    "error": r.error,
                }
                for vid, r in state.items()
            }
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            _ui_warn(f"transcript state file write failed: {exc}")

    progress_step = max(1, total // 10) if total else 1
    processed = 0
    fetched = 0

    def _tick() -> None:
        if processed % progress_step == 0 and processed < total:
            _ui_step(processed, total, "Transcripts")

    for meta in items:
        if max_videos is not None and fetched >= max_videos:
            break
        prior = state.get(meta.video_id)
        if prior and prior.status == "ok":
            processed += 1
            _tick()
            continue
        if prior and not prior.is_transient and prior.status != "error":
            processed += 1
            _tick()
            continue
        transcript_existed = (work_dir / f"{meta.video_id}.transcript").exists()
        result = pull_transcript(meta, client=client, work_dir=work_dir)
        if result.status == "ok" and not prior and transcript_existed:
            state[meta.video_id] = result
            processed += 1
            _tick()
            continue
        if result.is_transient and retry_transient:
            for attempt, delay in enumerate(_RETRY_DELAYS, 1):
                _ui_warn(
                    f"Transient failure for {meta.video_id} ({result.status}), "
                    f"retry {attempt}/{len(_RETRY_DELAYS)} in {delay:.0f}s"
                )
                sleeper(delay)
                result = pull_transcript(
                    meta,
                    client=client,
                    work_dir=work_dir,
                )
                if not result.is_transient:
                    break
        state[meta.video_id] = result
        fetched += 1
        processed += 1
        _persist()
        _tick()
        if sleep_between > 0:
            sleeper(sleep_between)
    if processed:
        _ui_step(processed, total, "Transcripts")
    if state:
        _persist()
    return state


__all__ = [
    "VideoMeta",
    "TranscriptResult",
    "YouTubeClient",
    "YouTubeChannel",
    "pull_catalog",
    "pull_transcript",
    "pull_all_transcripts",
    "load_video_meta",
    "parse_transcript_frontmatter",
    "strip_frontmatter",
    "clean_vtt",
]
