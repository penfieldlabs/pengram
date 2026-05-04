# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram._tools — external CLI tool resolution."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pengram._tools import ToolNotFoundError, resolve_tool


def test_resolve_tool_prefers_venv_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_venv = tmp_path / "fakevenv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    fake_tool = fake_bin / "mytool"
    fake_tool.touch()
    fake_tool.chmod(0o755)

    monkeypatch.setattr("pengram._tools.sys.prefix", str(fake_venv))
    monkeypatch.setattr("pengram._tools.sys.base_prefix", "/usr")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "mytool").touch()
    (other_dir / "mytool").chmod(0o755)
    monkeypatch.setenv("PATH", str(other_dir))

    resolved = resolve_tool("mytool", install_hint="install it")
    assert resolved == str(fake_tool)


def test_resolve_tool_falls_back_to_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_venv = tmp_path / "fakevenv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)

    monkeypatch.setattr("pengram._tools.sys.prefix", str(fake_venv))
    monkeypatch.setattr("pengram._tools.sys.base_prefix", "/usr")
    path_dir = tmp_path / "path-bin"
    path_dir.mkdir()
    path_tool = path_dir / "mytool"
    path_tool.touch()
    path_tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))

    resolved = resolve_tool("mytool", install_hint="install it")
    assert resolved == str(path_tool)


def test_resolve_tool_raises_with_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_venv = tmp_path / "fakevenv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)

    monkeypatch.setattr("pengram._tools.sys.prefix", str(fake_venv))
    monkeypatch.setattr("pengram._tools.sys.base_prefix", "/usr")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(ToolNotFoundError, match="pengram\\[youtube\\]"):
        resolve_tool("yt-dlp", install_hint="Install with: pip install 'pengram[youtube]'")


def test_tool_not_found_error_is_file_not_found() -> None:
    assert issubclass(ToolNotFoundError, FileNotFoundError)


def test_tool_not_found_error_is_pengram_error() -> None:
    from pengram.errors import PengramError

    assert issubclass(ToolNotFoundError, PengramError)


def test_resolve_tool_no_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pengram._tools.sys.prefix", "/usr")
    monkeypatch.setattr("pengram._tools.sys.base_prefix", "/usr")
    path_dir = tmp_path / "path-bin"
    path_dir.mkdir()
    path_tool = path_dir / "sometool"
    path_tool.touch()
    path_tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))

    resolved = resolve_tool("sometool", install_hint="install it")
    assert resolved == str(path_tool)


def test_resolve_tool_real_venv(tmp_path: Path) -> None:
    """Regression: _venv_bin_dir must not dereference venv symlinks.

    Creates a real venv, drops a fake tool in its bin/, then runs
    resolve_tool from a subprocess inside that venv.
    """
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    fake_tool = bin_dir / "fake-cli"
    fake_tool.write_text("#!/bin/sh\necho ok\n")
    fake_tool.chmod(0o755)

    venv_python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
    project_root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {str(project_root)!r})
        from pengram._tools import resolve_tool
        print(resolve_tool('fake-cli', install_hint='install it'))
    """)
    result = subprocess.run(
        [str(venv_python), "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(fake_tool) in result.stdout.strip()
