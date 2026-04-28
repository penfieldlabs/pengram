# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.security."""

from __future__ import annotations

from pathlib import Path

import pytest

from pengram import security
from pengram.security import SecurityError

# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------


def test_validate_url_http() -> None:
    assert security.validate_url("http://example.com") == "http://example.com"


def test_validate_url_https() -> None:
    assert security.validate_url("https://example.com/path") == "https://example.com/path"


def test_validate_url_rejects_file() -> None:
    with pytest.raises(SecurityError):
        security.validate_url("file:///etc/passwd")


def test_validate_url_rejects_javascript() -> None:
    with pytest.raises(SecurityError):
        security.validate_url("javascript:alert(1)")


def test_validate_url_rejects_empty() -> None:
    with pytest.raises(SecurityError):
        security.validate_url("")


def test_validate_url_rejects_no_host() -> None:
    with pytest.raises(SecurityError):
        security.validate_url("https://")


# ---------------------------------------------------------------------------
# validate_path
# ---------------------------------------------------------------------------


def test_validate_path_allows_within_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    resolved = security.validate_path("sub/file.txt", tmp_path)
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_validate_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        security.validate_path("../../etc/passwd", tmp_path)


def test_validate_path_rejects_absolute_outside(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        security.validate_path("/etc/passwd", tmp_path)


def test_validate_path_accepts_absolute_inside(tmp_path: Path) -> None:
    inside = tmp_path / "a"
    inside.mkdir()
    resolved = security.validate_path(str(inside), tmp_path)
    assert resolved == inside.resolve()


# ---------------------------------------------------------------------------
# sanitize_label
# ---------------------------------------------------------------------------


def test_sanitize_label_strips_control_chars() -> None:
    assert security.sanitize_label("foo\x00bar\x07baz") == "foobarbaz"


def test_sanitize_label_html_escapes() -> None:
    assert security.sanitize_label("<script>") == "&lt;script&gt;"


def test_sanitize_label_truncates() -> None:
    result = security.sanitize_label("a" * 500, max_len=50)
    assert len(result) <= 50


def test_sanitize_label_trims_whitespace() -> None:
    assert security.sanitize_label("  hello  ") == "hello"


def test_sanitize_label_preserves_unicode() -> None:
    assert security.sanitize_label("café") == "café"


def test_sanitize_label_non_string_input() -> None:
    assert security.sanitize_label(42) == "42"


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


def test_sanitize_filename_replaces_slash() -> None:
    assert "/" not in security.sanitize_filename("a/b/c.md")


def test_sanitize_filename_replaces_unsafe() -> None:
    out = security.sanitize_filename("foo*bar?.txt")
    for ch in '*?<>:"|':
        assert ch not in out


def test_sanitize_filename_strips_leading_dots() -> None:
    assert not security.sanitize_filename("...hidden.md").startswith(".")


def test_sanitize_filename_windows_reserved() -> None:
    for name in ("CON", "PRN", "COM1", "NUL"):
        out = security.sanitize_filename(name)
        assert out.upper() != name


def test_sanitize_filename_preserves_extension() -> None:
    out = security.sanitize_filename("x" * 500 + ".md", max_len=50)
    assert out.endswith(".md")
    assert len(out) <= 50


def test_sanitize_filename_empty_raises() -> None:
    with pytest.raises(SecurityError):
        security.sanitize_filename("...")


def test_sanitize_filename_non_string_input() -> None:
    assert security.sanitize_filename(123) == "123"


# ---------------------------------------------------------------------------
# slugify — single source of truth for vault filenames + wikilink targets.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Clean cases from the spec.
        ("Arne Trautmann", "arne-trautmann"),
        ("Carlson, A.", "carlson-a"),
        ("Typed Relationships", "typed-relationships"),
        ("cache_manager.js", "cache-manager-js"),
        ("__init__()", "init"),
        # Garbage real users actually throw at us.
        ("   ", "_unnamed"),
        ("", "_unnamed"),
        ("...", "_unnamed"),
        ("🔥 Hot Takes 🔥", "hot-takes"),
        ("Ça fait résumé", "ca-fait-resume"),
        ("CON", "_con"),
        ("NUL.txt", "_nul-txt"),
        ("COM1", "_com1"),
        ("../../../etc/passwd", "etc-passwd"),
        ("hello\x00world", "helloworld"),
        ("hello\nworld", "hello-world"),
        ("--flag-looking", "flag-looking"),
        ("-", "_unnamed"),
        ("Mr. Smith (CEO)", "mr-smith-ceo"),
        ("<script>alert(1)</script>", "script-alert-1-script"),
        ("file:///etc/shadow", "file-etc-shadow"),
        ("C:\\Users\\tim\\secrets.txt", "c-users-tim-secrets-txt"),
    ],
)
def test_slugify_spec_examples(raw: str, expected: str) -> None:
    assert security.slugify(raw) == expected


def test_slugify_cjk_strips_to_unnamed() -> None:
    # No latin-ish chars survive NFKD → regex eats everything → _unnamed.
    assert security.slugify("日本語のファイル") == "_unnamed"


def test_slugify_truncates_long_input() -> None:
    result = security.slugify("a" * 500)
    assert len(result) == 200
    assert result == "a" * 200


def test_slugify_prefers_hyphen_break_near_end() -> None:
    # max_len=20; last hyphen after position 10 should become the break.
    label = "mr-smith-ceo-of-acme-corporation-with-a-very-long-title"
    result = security.slugify(label, max_len=20)
    assert len(result) <= 20
    assert not result.endswith("-")
    # break point should be at a hyphen boundary (hyphen-aware truncation).
    assert "-" in result


def test_slugify_lowercase_uniformly() -> None:
    assert security.slugify("HELLO World") == "hello-world"


def test_slugify_non_string_input() -> None:
    assert security.slugify(42) == "42"


def test_slugify_idempotent() -> None:
    # A slug passed through slugify again should be unchanged.
    s = security.slugify("Some Messy Label!")
    assert security.slugify(s) == s


def test_slugify_windows_reserved_with_suffix() -> None:
    # Ensures the check inspects the first segment, not the whole slug.
    assert security.slugify("PRN backup").startswith("_prn")


# ---------------------------------------------------------------------------
# validate_video_id
# ---------------------------------------------------------------------------


def test_validate_video_id_accepts_standard() -> None:
    assert security.validate_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_validate_video_id_accepts_dashes_underscores() -> None:
    assert security.validate_video_id("abc-_DEF-123") == "abc-_DEF-123"


def test_validate_video_id_rejects_short() -> None:
    with pytest.raises(SecurityError):
        security.validate_video_id("abc")


def test_validate_video_id_rejects_long() -> None:
    with pytest.raises(SecurityError):
        security.validate_video_id("a" * 25)


def test_validate_video_id_rejects_glob_injection() -> None:
    with pytest.raises(SecurityError):
        security.validate_video_id("../../../etc")


def test_validate_video_id_rejects_special_chars() -> None:
    with pytest.raises(SecurityError):
        security.validate_video_id("dQw4w9Wg*cQ")


def test_validate_video_id_rejects_empty() -> None:
    with pytest.raises(SecurityError):
        security.validate_video_id("")


def test_validate_video_id_rejects_non_string() -> None:
    with pytest.raises(SecurityError):
        security.validate_video_id(12345678)
