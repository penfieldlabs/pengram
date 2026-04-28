# ADR-005: Validate YouTube video IDs at the security boundary

## Status
Accepted

## Context
YouTube video IDs from yt-dlp JSON are interpolated into file paths
(`f"{video_id}.transcript"`) and glob patterns (`f"{video_id}*.vtt"`).
A crafted ID containing glob metacharacters or path separators could
match files outside the intended output directory.

The practical exposure is low — input comes from yt-dlp, not the user —
but the security module's posture is "validate at system boundaries,"
and yt-dlp output is an external boundary.

## Decision
Add `validate_video_id()` to `security.py` enforcing
`^[A-Za-z0-9_-]{8,24}$`. Call it at three entry points:
`YouTubeClient.fetch_subtitles`, `pull_transcript`, and `pull_catalog`
(where invalid IDs are silently skipped rather than raising, since
they come from batch enumeration).

## Consequences
- Malformed IDs from a compromised or buggy yt-dlp release are caught
  before they touch the filesystem.
- The 8–24 character range covers standard YouTube IDs (11 chars) with
  headroom for future format changes.
- Test video IDs throughout the suite must be valid-length strings.
