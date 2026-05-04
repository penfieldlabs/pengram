# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.detect."""

from __future__ import annotations

from pathlib import Path

import pytest

from pengram import detect
from pengram.detect import FileType


def test_classify_code(tmp_path: Path) -> None:
    p = tmp_path / "hello.py"
    p.write_text("print('hi')")
    assert detect.classify(p) == FileType.CODE


def test_classify_typescript(tmp_path: Path) -> None:
    p = tmp_path / "x.ts"
    p.write_text("const x = 1;")
    assert detect.classify(p) == FileType.CODE


def test_classify_markdown_document(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text("# Notes\n\nSome ideas.")
    assert detect.classify(p) == FileType.DOCUMENT


def test_classify_pdf_as_document(tmp_path: Path) -> None:
    """v0.2.0: PDFs are ``document``, not a separate ``paper`` kind."""
    p = tmp_path / "book.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    assert detect.classify(p) == FileType.DOCUMENT


def test_classify_epub_as_document(tmp_path: Path) -> None:
    p = tmp_path / "novel.epub"
    p.write_bytes(b"\x00")
    assert detect.classify(p) == FileType.DOCUMENT


def test_classify_video(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00" * 10)
    assert detect.classify(p) == FileType.VIDEO


def test_classify_audio(tmp_path: Path) -> None:
    p = tmp_path / "song.mp3"
    p.write_bytes(b"\x00" * 10)
    assert detect.classify(p) == FileType.AUDIO


def test_classify_image(tmp_path: Path) -> None:
    p = tmp_path / "pic.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert detect.classify(p) == FileType.IMAGE


def test_classify_transcript(tmp_path: Path) -> None:
    p = tmp_path / "talk.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello.")
    assert detect.classify(p) == FileType.TRANSCRIPT


def test_classify_transcript_dot_transcript(tmp_path: Path) -> None:
    p = tmp_path / "abc123.transcript"
    p.write_text("---\nvideo_id: abc123\n---\nHello world.")
    assert detect.classify(p) == FileType.TRANSCRIPT


def test_classify_unknown(tmp_path: Path) -> None:
    p = tmp_path / "data.xyz"
    p.write_text("")
    assert detect.classify(p) is None


def test_academic_markdown_stays_document(tmp_path: Path) -> None:
    """v0.2.0 killed the paper-heuristic reclassification. A paper-like
    .md is still just a ``document`` — file format is metadata, not a
    content type."""
    p = tmp_path / "attention.md"
    p.write_text(
        "arxiv 1706.03762\n"
        "Abstract: We propose a new architecture.\n"
        "Preprint under review. doi: 10.1145/1234.\n"
        "[1] Vaswani et al.\n"
    )
    assert detect.classify(p) == FileType.DOCUMENT


def test_markdown_classifies_as_document(tmp_path: Path) -> None:
    p = tmp_path / "todo.md"
    p.write_text("- buy milk\n- write code\n")
    assert detect.classify(p) == FileType.DOCUMENT


def test_is_sensitive_function_is_gone() -> None:
    """v0.2.0: the pattern-based 'sensitive file' filter was deleted —
    it was silently dropping legitimate corpus files with words like
    'secret' in the filename. Pin it as absent so it doesn't creep
    back in."""
    assert not hasattr(detect, "_is_sensitive")
    assert not hasattr(detect, "_SENSITIVE_PATTERNS")


def test_collect_files_groups_by_type(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# Hi")
    (tmp_path / "c.mp3").write_bytes(b"\x00")
    groups = detect.collect_files(tmp_path)
    assert len(groups[FileType.CODE]) == 1
    assert len(groups[FileType.DOCUMENT]) == 1
    assert len(groups[FileType.AUDIO]) == 1


def test_collect_files_skips_hidden_files(tmp_path: Path) -> None:
    """Dotfiles still don't land in the corpus — catches .env, .git, etc."""
    (tmp_path / ".env").write_text("whatever")
    (tmp_path / "ok.py").write_text("pass")
    groups = detect.collect_files(tmp_path)
    all_paths = [p for files in groups.values() for p in files]
    assert not any(".env" in str(p) for p in all_paths)
    assert any("ok.py" in str(p) for p in all_paths)


def test_collect_files_does_not_filter_on_secret_in_filename(
    tmp_path: Path,
) -> None:
    """Regression for v0.2.0 sensitive-filter removal. Files whose names
    contain words like 'secret', 'password', 'credential' are NOT
    silently dropped — PENgram processes whatever the user put in the
    input folder. Use .pengramignore to exclude specific files."""
    legit = [
        "the_secret_of_nitric_oxide.md",
        "unlocking_secrets_fox.md",
        "password_psychology.md",
        "trust_and_credentials.md",
        "Young Forever - The Secrets to Living Longest.md",
    ]
    for name in legit:
        (tmp_path / name).write_text("body " * 50)
    groups = detect.collect_files(tmp_path)
    document_names = {p.name for p in groups[FileType.DOCUMENT]}
    for name in legit:
        assert name in document_names, (
            f"{name!r} was silently dropped; the sensitive-filter regression"
        )


def test_collect_files_skips_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "mod.js").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"")
    (tmp_path / "real.py").write_text("pass")
    groups = detect.collect_files(tmp_path)
    all_paths = [str(p) for files in groups.values() for p in files]
    assert not any("node_modules" in p for p in all_paths)
    assert not any("__pycache__" in p for p in all_paths)
    assert any("real.py" in p for p in all_paths)


def test_pengramignore_excludes_patterns(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("pass")
    (tmp_path / "drop.py").write_text("pass")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "x.py").write_text("pass")
    (tmp_path / ".pengramignore").write_text("drop.py\nignored_dir/\n")
    groups = detect.collect_files(tmp_path)
    all_paths = [str(p) for files in groups.values() for p in files]
    assert any("keep.py" in p for p in all_paths)
    assert not any("drop.py" in p for p in all_paths)
    assert not any("ignored_dir" in p for p in all_paths)


def test_pengramignore_missing_is_fine(tmp_path: Path) -> None:
    assert detect._load_ignore_patterns(tmp_path) == []


def test_pengramignore_ignores_comments(tmp_path: Path) -> None:
    (tmp_path / ".pengramignore").write_text("# a comment\n\nfoo.py\n")
    patterns = detect._load_ignore_patterns(tmp_path)
    assert patterns == ["foo.py"]


def test_extract_pdf_text_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "missing.pdf"
    assert detect.extract_pdf_text(p) == ""


def test_extract_pdf_text_corrupt_pdf_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not really a pdf")
    assert detect.extract_pdf_text(p) == ""


def test_no_file_type_paper_exists() -> None:
    """v0.2.0: ``FileType.PAPER`` is removed. There is no 'paper' kind."""
    assert not hasattr(FileType, "PAPER")


def test_extract_epub_text_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ebooklib+bs4 installed, extract_epub_text returns ''."""
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in {"ebooklib", "ebooklib.epub", "bs4"}:
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    for mod in ("ebooklib", "ebooklib.epub", "bs4"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = detect.extract_epub_text(Path("/nonexistent.epub"))
    assert result == ""


def test_extract_epub_text_happy_path(tmp_path: Path) -> None:
    """Real ebooklib extraction — dev dep is always installed."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("id-1")
    book.set_title("Test Book")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
    chapter.content = "<h1>Chapter 1</h1><p>The mitochondria is the powerhouse.</p>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    text = detect.extract_epub_text(path)
    assert "mitochondria is the powerhouse" in text


def test_require_epub_raises_without_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in {"ebooklib", "bs4"}:
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    for mod in ("ebooklib", "bs4"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="ebooklib"):
        detect.require_epub()


def test_require_pypdf_when_installed() -> None:
    # pypdf is a dev/CI dependency (see pyproject [dev]) so it is always
    # present when the suite runs. Importing directly documents that.
    import pypdf  # noqa: F401

    detect.require_pypdf()


def test_require_pypdf_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a minimal install to exercise the import-error path."""
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("pypdf not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pypdf", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pypdf"):
        detect.require_pypdf()


def test_collect_files_hidden_ignored(tmp_path: Path) -> None:
    (tmp_path / ".hidden.py").write_text("pass")
    (tmp_path / "visible.py").write_text("pass")
    groups = detect.collect_files(tmp_path)
    all_paths = [str(p) for files in groups.values() for p in files]
    assert any("visible.py" in p for p in all_paths)
    assert not any(".hidden.py" in p for p in all_paths)


def test_detect_no_longer_classifies_svg_as_image() -> None:
    assert ".svg" not in detect.IMAGE_EXTENSIONS
