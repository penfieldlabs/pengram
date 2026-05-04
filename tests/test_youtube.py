# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.youtube."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pengram.config import YouTubeChannel
from pengram.youtube import (
    VideoMeta,
    YouTubeClient,
    clean_vtt,
    pull_all_transcripts,
    pull_catalog,
    pull_transcript,
)


def make_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# VTT cleaning
# ---------------------------------------------------------------------------


def test_clean_vtt_basic() -> None:
    raw = (
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: en\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\n"
        "Hello <b>world</b>\n"
        "\n"
        "00:00:02.500 --> 00:00:05.000\n"
        "How are you?\n"
    )
    cleaned = clean_vtt(raw)
    assert "Hello world" in cleaned
    assert "How are you?" in cleaned
    assert "WEBVTT" not in cleaned
    assert "-->" not in cleaned


def test_clean_vtt_deduplicates() -> None:
    raw = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\nhello\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello\n\n"
        "00:00:02.000 --> 00:00:03.000\nworld\n"
    )
    cleaned = clean_vtt(raw)
    assert cleaned.count("hello") == 1
    assert "world" in cleaned


def test_clean_vtt_strips_speaker_markers() -> None:
    """>> speaker-change markers must not render as Markdown blockquotes."""
    raw = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n>> Hello there\n\n"
        "00:00:01.000 --> 00:00:02.000\nI agree\n\n"
        "00:00:02.000 --> 00:00:03.000\n>> Next speaker\n"
    )
    cleaned = clean_vtt(raw)
    assert ">>" not in cleaned
    assert "Hello there" in cleaned
    assert "Next speaker" in cleaned
    lines = cleaned.splitlines()
    hello_idx = lines.index("Hello there")
    assert hello_idx > 0 or lines[0] == "Hello there"


def test_clean_vtt_unescapes_html_entities() -> None:
    raw = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nIt&#39;s a &amp; test &lt;ok&gt;\n"
    cleaned = clean_vtt(raw)
    assert "It's a & test <ok>" in cleaned
    assert "&#39;" not in cleaned
    assert "&amp;" not in cleaned


# ---------------------------------------------------------------------------
# fetch_subtitles: single-call auto-subs only
# ---------------------------------------------------------------------------


_VID = "dQw4w9WgXcQ"


def _meta(video_id: str = _VID, **kwargs: Any) -> VideoMeta:
    """Build a VideoMeta with sane defaults for testing."""
    defaults: dict[str, Any] = {
        "title": f"Test Video {video_id}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel": "test_channel",
        "tab": "videos",
    }
    defaults.update(kwargs)
    return VideoMeta(video_id=video_id, **defaults)


def test_fetch_subtitles_single_call(tmp_path: Path) -> None:
    """Exactly one yt-dlp call per video, using --write-auto-subs only."""
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        (tmp_path / f"{_VID}.en.vtt").write_text("auto")
        return make_result()

    client = YouTubeClient(runner=runner)
    result = client.fetch_subtitles(_VID, tmp_path)

    assert result == tmp_path / f"{_VID}.en.vtt"
    assert len(calls) == 1
    assert "--write-auto-subs" in calls[0]
    assert "--write-subs" not in calls[0]


def test_fetch_subtitles_returns_none_when_no_captions(tmp_path: Path) -> None:
    """yt-dlp succeeds but writes nothing → None."""
    client = YouTubeClient(runner=lambda a: make_result())
    assert client.fetch_subtitles(_VID, tmp_path) is None


def test_fetch_subtitles_uses_plain_en_not_locale_glob(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return make_result()

    client = YouTubeClient(runner=runner)
    client.fetch_subtitles(_VID, tmp_path)

    assert len(calls) == 1
    assert "en.*" not in calls[0]
    idx = calls[0].index("--sub-langs")
    assert calls[0][idx + 1] == "en"


def test_fetch_subtitles_translates_rate_limit(tmp_path: Path) -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result(stderr="HTTP Error 429: Too Many Requests", returncode=1)

    client = YouTubeClient(runner=runner)
    with pytest.raises(RuntimeError, match="rate_limited"):
        client.fetch_subtitles(_VID, tmp_path)


def test_fetch_subtitles_nonzero_exit_with_vtt_succeeds(tmp_path: Path) -> None:
    """yt-dlp exits non-zero (e.g. EJS deprecation warning) but wrote a VTT."""

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        (tmp_path / f"{_VID}.en.vtt").write_text("WEBVTT\n\nhello\n")
        return make_result(stderr="WARNING: no supported javascript runtime found", returncode=1)

    client = YouTubeClient(runner=runner)
    result = client.fetch_subtitles(_VID, tmp_path)
    assert result is not None
    assert result.name.endswith(".vtt")


def test_fetch_subtitles_captionless_video_returns_none(tmp_path: Path) -> None:
    """Captionless video: no VTT, non-zero exit → None, not an error."""

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result(stderr="This video does not have auto-generated subtitles", returncode=1)

    client = YouTubeClient(runner=runner)
    assert client.fetch_subtitles(_VID, tmp_path) is None


def test_fetch_subtitles_ejs_warning_with_vtt_exact_stderr(tmp_path: Path) -> None:
    """Exact EJS deprecation stderr from tester's Gammon run."""
    ejs_stderr = (
        "WARNING: [youtube] no supported javascript runtime could be found. "
        "Install one of: PhantomJS, node.js"
    )

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        (tmp_path / f"{_VID}.en.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n")
        return make_result(stderr=ejs_stderr, returncode=1)

    client = YouTubeClient(runner=runner)
    result = client.fetch_subtitles(_VID, tmp_path)
    assert result is not None


def test_fetch_subtitles_ejs_warning_no_vtt_returns_none(tmp_path: Path) -> None:
    """EJS warning + no VTT = None (captionless), not an error."""
    ejs_stderr = (
        "WARNING: [youtube] no supported javascript runtime could be found. "
        "Install one of: PhantomJS, node.js"
    )

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result(stderr=ejs_stderr, returncode=1)

    client = YouTubeClient(runner=runner)
    assert client.fetch_subtitles(_VID, tmp_path) is None


def test_fetch_subtitles_sign_in_is_rate_limited(tmp_path: Path) -> None:
    """'sign in' in stderr is a genuine rate-limit signal."""

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result(stderr="Please sign in to confirm you're not a bot", returncode=1)

    client = YouTubeClient(runner=runner)
    with pytest.raises(RuntimeError, match="rate_limited"):
        client.fetch_subtitles(_VID, tmp_path)


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------


def test_pull_catalog_multiple_tabs() -> None:
    entries_videos = [
        {"id": "vid1_aaa01", "title": "Video 1"},
        {"id": "vid2_bbb02", "title": "Video 2"},
    ]
    entries_streams = [{"id": "str1_ccc03", "title": "Stream 1"}]

    calls: list[str] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        url = args[-1]
        calls.append(url)
        if url.endswith("/videos"):
            return make_result("\n".join(json.dumps(e) for e in entries_videos))
        if url.endswith("/streams"):
            return make_result("\n".join(json.dumps(e) for e in entries_streams))
        return make_result("", "unknown", 1)

    client = YouTubeClient(runner=runner)
    channel = YouTubeChannel(url="https://youtube.com/@x", label="X", tabs=["videos", "streams"])
    catalog = pull_catalog(channel, client=client)
    tabs = {v.tab for v in catalog}
    assert tabs == {"videos", "streams"}
    assert len(catalog) == 3
    assert any(v.video_id == "vid1_aaa01" for v in catalog)
    assert any(v.video_id == "str1_ccc03" for v in catalog)


def test_pull_catalog_ignores_invalid_tab() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result(json.dumps({"id": "vid1_aaa01", "title": "Only video"}))

    client = YouTubeClient(runner=runner)
    channel = YouTubeChannel(url="https://youtube.com/@x", label="X", tabs=["videos", "podcasts"])
    catalog = pull_catalog(channel, client=client)
    assert {v.tab for v in catalog} == {"videos"}


def test_pull_catalog_yt_dlp_error() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result("", "boom", 1)

    client = YouTubeClient(runner=runner)
    channel = YouTubeChannel(url="https://youtube.com/@x", label="X")
    with pytest.raises(RuntimeError, match="boom"):
        pull_catalog(channel, client=client)


def test_pull_catalog_skips_missing_tab() -> None:
    """Missing tab (e.g. no shorts) should warn and continue, not crash."""

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        url = args[-1]
        if "/shorts" in url:
            return make_result(
                "", "ERROR: [youtube:tab] @X: This channel does not have a shorts tab", 1
            )
        return make_result(json.dumps({"id": "vid1_aaa01", "title": "V"}))

    client = YouTubeClient(runner=runner)
    channel = YouTubeChannel(url="https://youtube.com/@X", label="X", tabs=["videos", "shorts"])
    catalog = pull_catalog(channel, client=client)
    assert len(catalog) == 1
    assert catalog[0].video_id == "vid1_aaa01"


def test_pull_catalog_validates_url() -> None:
    channel = YouTubeChannel(url="file:///etc/passwd", label="Evil")
    with pytest.raises(Exception):
        pull_catalog(channel, client=YouTubeClient(runner=lambda a: make_result("")))


# ---------------------------------------------------------------------------
# Transcript pulling
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(
        self, *, subtitles_path: Path | None = None, raise_exc: Exception | None = None
    ) -> None:
        self.subtitles_path = subtitles_path
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    def fetch_subtitles(self, video_id: str, out_dir: Path) -> Path | None:
        self.calls.append(video_id)
        if self.raise_exc:
            raise self.raise_exc
        return self.subtitles_path


def test_pull_transcript_happy(tmp_path: Path) -> None:
    vtt = tmp_path / f"{_VID}.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello world\n")
    client = FakeClient(subtitles_path=vtt)
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "ok"
    assert result.text and "Hello world" in result.text
    assert (tmp_path / f"{_VID}.transcript").exists()
    assert not vtt.exists()
    saved = (tmp_path / f"{_VID}.transcript").read_text()
    assert saved.startswith("---\n")
    assert f"video_id: {_VID}" in saved
    assert 'title: "Test Video' in saved


def test_pull_transcript_skips_download_if_txt_exists(tmp_path: Path) -> None:
    (tmp_path / f"{_VID}.transcript").write_text("already cleaned")
    client = FakeClient(subtitles_path=None)
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "ok"
    assert result.text == "already cleaned"
    assert client.calls == []


def test_pull_transcript_no_subtitles(tmp_path: Path) -> None:
    client = FakeClient(subtitles_path=None)
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "no_subtitles"


def test_pull_transcript_empty_after_cleaning(tmp_path: Path) -> None:
    vtt = tmp_path / f"{_VID}.en.vtt"
    vtt.write_text("WEBVTT\nKind: captions\nLanguage: en\n")
    client = FakeClient(subtitles_path=vtt)
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "empty_after_cleaning"
    assert not vtt.exists()


def test_pull_transcript_rate_limited(tmp_path: Path) -> None:
    client = FakeClient(raise_exc=RuntimeError("rate_limited: 429"))
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "rate_limited"
    assert result.is_transient


def test_pull_transcript_timeout(tmp_path: Path) -> None:
    client = FakeClient(raise_exc=RuntimeError("timeout while fetching"))
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "timeout"
    assert result.is_transient


def test_pull_transcript_generic_error(tmp_path: Path) -> None:
    client = FakeClient(raise_exc=RuntimeError("unknown failure"))
    result = pull_transcript(_meta(), client=client, work_dir=tmp_path)
    assert result.status == "error"


# ---------------------------------------------------------------------------
# Batch pulling with resume
# ---------------------------------------------------------------------------


def test_pull_all_resumes_from_state(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    vid_a = "aaaabbbb0001"
    vid_b = "aaaabbbb0002"
    state_file.write_text(
        json.dumps(
            {
                vid_a: {"text": "already done", "status": "ok"},
            }
        )
    )

    vtt = tmp_path / f"{vid_b}.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nnew\n")
    client = FakeClient(subtitles_path=vtt)

    catalog = [
        VideoMeta(video_id=vid_a, title="A", url="", channel="ch", tab="videos"),
        VideoMeta(video_id=vid_b, title="B", url="", channel="ch", tab="videos"),
    ]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert result[vid_a].text == "already done"
    assert result[vid_b].status == "ok"
    assert client.calls == [vid_b]


def test_pull_all_retries_transient(tmp_path: Path) -> None:
    attempts = {"count": 0}

    vid = "retrytest01x"

    class RetryClient:
        def fetch_subtitles(self, video_id: str, out_dir: Path) -> Path | None:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("rate_limited: slow down")
            vtt = out_dir / f"{video_id}.en.vtt"
            vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nok\n")
            return vtt

    sleeps: list[float] = []
    catalog = [VideoMeta(video_id=vid, title="X", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=RetryClient(),
        work_dir=tmp_path,
        retry_transient=True,
        sleep_between=0,
        sleeper=sleeps.append,
    )
    assert result[vid].status == "ok"
    assert attempts["count"] == 2
    assert sleeps[0] == 5.0


def test_pull_all_retries_capped_exponential_backoff(tmp_path: Path) -> None:
    """All three retries fire with 5 / 15 / 45 s delays before giving up."""

    vid = "ratelimit01x"

    class AlwaysRateLimited:
        def fetch_subtitles(self, video_id: str, out_dir: Path) -> Path | None:
            raise RuntimeError("rate_limited: 429")

    sleeps: list[float] = []
    catalog = [VideoMeta(video_id=vid, title="X", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=AlwaysRateLimited(),
        work_dir=tmp_path,
        retry_transient=True,
        sleep_between=0,
        sleeper=sleeps.append,
    )
    assert result[vid].status == "rate_limited"
    assert sleeps == [5.0, 15.0, 45.0]


def test_pull_all_max_videos(tmp_path: Path) -> None:
    vtt = tmp_path / "placeholder.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n")
    client = FakeClient(subtitles_path=vtt)
    catalog = [
        VideoMeta(video_id=f"maxvid{i:05d}", title=f"V{i}", url="", channel="c", tab="videos")
        for i in range(10)
    ]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        max_videos=3,
        sleep_between=0,
    )
    assert len(result) == 3


def test_pull_all_skips_permanent_failure(tmp_path: Path) -> None:
    client = FakeClient(subtitles_path=None)
    state_file = tmp_path / "state.json"
    catalog = [VideoMeta(video_id="permfail001x", title="A", url="", channel="c", tab="videos")]
    # First run: records no_subtitles
    pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    # Second run: skips without calling client again
    client.calls.clear()
    pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert client.calls == []


def test_pull_all_writes_frontmatter_to_transcript(tmp_path: Path) -> None:
    vid = "metaside00x1"
    vtt = tmp_path / f"{vid}.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n")
    client = FakeClient(subtitles_path=vtt)
    catalog = [
        VideoMeta(
            video_id=vid,
            title="A Great Title",
            url="https://www.youtube.com/watch?v=" + vid,
            channel="ch",
            tab="videos",
            views=12345,
            upload_date="20240315",
        ),
    ]
    pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        sleep_between=0,
    )
    txt = (tmp_path / f"{vid}.transcript").read_text()
    assert txt.startswith("---\n")
    assert "video_id: metaside00x1" in txt
    assert 'title: "A Great Title"' in txt
    assert "views: 12345" in txt
    assert 'upload_date: "20240315"' in txt
    assert "hello" in txt


def test_load_video_meta_reads_frontmatter(tmp_path: Path) -> None:
    from pengram.youtube import load_video_meta

    vid = "loadmeta001x"
    txt = tmp_path / f"{vid}.txt"
    txt.write_text('---\nvideo_id: loadmeta001x\ntitle: "My Title"\nviews: 99\n---\n\nhello\n')
    meta = load_video_meta(tmp_path, vid)
    assert meta is not None
    assert meta["title"] == "My Title"
    assert meta["views"] == 99


def test_load_video_meta_falls_back_to_sidecar(tmp_path: Path) -> None:
    from pengram.youtube import load_video_meta

    vid = "sidecarfb001"
    (tmp_path / f"{vid}.transcript").write_text("plain text, no frontmatter")
    sidecar = tmp_path / f"{vid}.meta.json"
    sidecar.write_text(json.dumps({"title": "Sidecar Title", "video_id": vid}))
    meta = load_video_meta(tmp_path, vid)
    assert meta is not None
    assert meta["title"] == "Sidecar Title"


def test_load_video_meta_returns_none_when_missing(tmp_path: Path) -> None:
    from pengram.youtube import load_video_meta

    assert load_video_meta(tmp_path, "nonexistent01") is None


def test_parse_transcript_frontmatter() -> None:
    from pengram.youtube import parse_transcript_frontmatter

    text = '---\nvideo_id: abc123\ntitle: "Test"\nviews: 100\n---\n\ncontent here\n'
    fm = parse_transcript_frontmatter(text)
    assert fm["video_id"] == "abc123"
    assert fm["title"] == "Test"
    assert fm["views"] == 100


def test_strip_frontmatter() -> None:
    from pengram.youtube import strip_frontmatter

    text = "---\nvideo_id: abc\n---\n\ncontent here\n"
    assert strip_frontmatter(text) == "\ncontent here\n"
    assert strip_frontmatter("no frontmatter") == "no frontmatter"


def test_transcript_frontmatter_quotes_upload_date() -> None:
    """upload_date must be YAML-quoted so safe_load doesn't parse it as datetime.date."""
    from pengram.youtube import parse_transcript_frontmatter

    text = '---\nvideo_id: v1\nupload_date: "2025-12-01"\n---\n\nbody\n'
    fm = parse_transcript_frontmatter(text)
    assert isinstance(fm["upload_date"], str)


def test_pull_all_disk_cache_hit_skips_sleep(tmp_path: Path) -> None:
    """When .transcript exists on disk but not in state, skip sleep."""
    vid = "diskcache00x"
    transcript = tmp_path / f"{vid}.transcript"
    transcript.write_text("---\nvideo_id: diskcache00x\n---\ncached content")

    client = FakeClient(subtitles_path=None)
    sleeps: list[float] = []
    catalog = [VideoMeta(video_id=vid, title="C", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        sleep_between=2.0,
        sleeper=sleeps.append,
    )
    assert result[vid].status == "ok"
    assert client.calls == []
    assert sleeps == []


def test_pull_all_disk_cache_hit_not_counted_against_max_videos(tmp_path: Path) -> None:
    """Disk cache hits must not decrement max_videos budget."""
    cached_vid = "cached00001x"
    fresh_vid = "freshv00001x"
    transcript = tmp_path / f"{cached_vid}.transcript"
    transcript.write_text("---\nvideo_id: cached00001x\n---\ncached")

    vtt = tmp_path / f"{fresh_vid}.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nfresh\n")
    client = FakeClient(subtitles_path=vtt)

    catalog = [
        VideoMeta(video_id=cached_vid, title="A", url="", channel="c", tab="videos"),
        VideoMeta(video_id=fresh_vid, title="B", url="", channel="c", tab="videos"),
    ]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        max_videos=1,
        sleep_between=0,
    )
    assert cached_vid in result
    assert fresh_vid in result
    assert client.calls == [fresh_vid]


def test_pull_all_disk_cache_hits_persisted_to_state_file(tmp_path: Path) -> None:
    """Disk cache hits are flushed to the state file by the trailing _persist."""
    vid = "diskflush00x"
    transcript = tmp_path / f"{vid}.transcript"
    transcript.write_text("---\nvideo_id: diskflush00x\n---\ntext")

    state_file = tmp_path / "state.json"
    client = FakeClient(subtitles_path=None)
    catalog = [VideoMeta(video_id=vid, title="F", url="", channel="c", tab="videos")]
    pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert state_file.exists()
    saved = json.loads(state_file.read_text())
    assert saved[vid]["status"] == "ok"


def test_pull_all_state_file_omits_transcript_text(tmp_path: Path) -> None:
    """State file must not store full transcript text — only status and error."""
    vid = "statetext00x"
    vtt = tmp_path / f"{vid}.en.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n")
    client = FakeClient(subtitles_path=vtt)
    state_file = tmp_path / "state.json"
    catalog = [VideoMeta(video_id=vid, title="T", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert result[vid].status == "ok"
    saved = json.loads(state_file.read_text())
    assert "text" not in saved[vid]
    assert saved[vid]["status"] == "ok"


def test_pull_all_reclassifies_stale_rate_limited_as_no_subtitles(tmp_path: Path) -> None:
    """Stale rate_limited entries without a .transcript become no_subtitles."""
    vid = "stalerl0001x"
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({vid: {"status": "rate_limited", "error": "old"}}))

    client = FakeClient(subtitles_path=None)
    catalog = [VideoMeta(video_id=vid, title="S", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert result[vid].status == "no_subtitles"
    assert client.calls == []


def test_pull_all_reclassifies_stale_rate_limited_to_ok_when_transcript_exists(
    tmp_path: Path,
) -> None:
    """Stale rate_limited entries WITH a .transcript become ok."""
    vid = "staleok0001x"
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({vid: {"status": "rate_limited", "error": "old"}}))
    (tmp_path / f"{vid}.transcript").write_text("---\nvideo_id: staleok0001x\n---\ncontent")

    client = FakeClient(subtitles_path=None)
    catalog = [VideoMeta(video_id=vid, title="S", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert result[vid].status == "ok"
    assert client.calls == []


def test_pull_all_reclassifies_stale_error_as_no_subtitles(tmp_path: Path) -> None:
    """Stale error entries without a .transcript become no_subtitles."""
    vid = "staleerr001x"
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({vid: {"status": "error", "error": "yt-dlp broke"}}))

    client = FakeClient(subtitles_path=None)
    catalog = [VideoMeta(video_id=vid, title="E", url="", channel="c", tab="videos")]
    result = pull_all_transcripts(
        catalog,
        client=client,
        work_dir=tmp_path,
        state_file=state_file,
        sleep_between=0,
    )
    assert result[vid].status == "no_subtitles"
    assert client.calls == []


def test_fetch_subtitles_captionless_nonzero_exit_returns_none(tmp_path: Path) -> None:
    """No VTT + non-zero exit + no rate-limit signal = None, not RuntimeError."""

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return make_result(stderr="some random yt-dlp error", returncode=1)

    client = YouTubeClient(runner=runner)
    assert client.fetch_subtitles(_VID, tmp_path) is None


def test_pull_all_emits_progress_ticks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """10% progress ticks fire during transcript pulling."""
    ticks: list[tuple[int, int, str]] = []
    monkeypatch.setattr(
        "pengram.youtube._ui_step", lambda cur, tot, lbl: ticks.append((cur, tot, lbl))
    )

    catalog = []
    for i in range(20):
        vid = f"prog{i:07d}x"
        vtt_copy = tmp_path / f"{vid}.en.vtt"
        vtt_copy.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n")
        catalog.append(VideoMeta(video_id=vid, title=f"V{i}", url="", channel="c", tab="videos"))

    class CopyClient:
        def fetch_subtitles(self, video_id: str, out_dir: Path) -> Path | None:
            p = out_dir / f"{video_id}.en.vtt"
            if p.exists():
                return p
            return None

    pull_all_transcripts(
        catalog,
        client=CopyClient(),
        work_dir=tmp_path,
        sleep_between=0,
    )
    assert len(ticks) > 1
    assert ticks[-1] == (20, 20, "Transcripts")
    assert all(label == "Transcripts" for _, _, label in ticks)


def test_pull_all_progress_final_tick_reflects_max_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final progress tick shows actual processed count, not catalog total."""
    ticks: list[tuple[int, int, str]] = []
    monkeypatch.setattr(
        "pengram.youtube._ui_step", lambda cur, tot, lbl: ticks.append((cur, tot, lbl))
    )

    catalog = []
    for i in range(20):
        vid = f"maxt{i:07d}x"
        vtt = tmp_path / f"{vid}.en.vtt"
        vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n")
        catalog.append(VideoMeta(video_id=vid, title=f"V{i}", url="", channel="c", tab="videos"))

    class CopyClient:
        def fetch_subtitles(self, video_id: str, out_dir: Path) -> Path | None:
            p = out_dir / f"{video_id}.en.vtt"
            return p if p.exists() else None

    pull_all_transcripts(
        catalog,
        client=CopyClient(),
        work_dir=tmp_path,
        max_videos=3,
        sleep_between=0,
    )
    final_cur, final_tot, _ = ticks[-1]
    assert final_cur == 3
    assert final_tot == 20
