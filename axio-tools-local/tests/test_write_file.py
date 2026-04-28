"""Tests for write_file tool handler."""

from __future__ import annotations

import os
import stat
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from axio_tools_local.write_file import write_file


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Generator[Path, None, None]:
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


async def write(filename: str, content: str, **kwargs: Any) -> str:
    return await write_file(file_path=filename, content=content, **kwargs)


class TestWriteFileBasic:
    async def test_creates_file(self, tmp_cwd: Path) -> None:
        await write("out.txt", "hello world")
        assert (tmp_cwd / "out.txt").read_text() == "hello world"

    async def test_returns_bytes_written(self, tmp_cwd: Path) -> None:
        result = await write("out.txt", "hello")
        assert "5" in result
        assert "out.txt" in result

    async def test_overwrites_existing(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("old content")
        await write("f.txt", "new content")
        assert (tmp_cwd / "f.txt").read_text() == "new content"

    async def test_overwrite_shorter_content(self, tmp_cwd: Path) -> None:
        """Overwriting with shorter content must not leave old tail."""
        (tmp_cwd / "f.txt").write_text("long content here")
        await write("f.txt", "short")
        assert (tmp_cwd / "f.txt").read_text() == "short"

    async def test_creates_subdirectories(self, tmp_cwd: Path) -> None:
        await write("sub/dir/file.txt", "nested")
        assert (tmp_cwd / "sub" / "dir" / "file.txt").read_text() == "nested"

    async def test_empty_content(self, tmp_cwd: Path) -> None:
        await write("empty.txt", "")
        assert (tmp_cwd / "empty.txt").read_text() == ""
        assert (tmp_cwd / "empty.txt").exists()

    async def test_unicode_content(self, tmp_cwd: Path) -> None:
        await write("u.txt", "привет мир\n日本語\n")
        assert (tmp_cwd / "u.txt").read_text() == "привет мир\n日本語\n"

    async def test_multiline_newlines_preserved(self, tmp_cwd: Path) -> None:
        content = "line0\nline1\nline2\n"
        await write("f.txt", content)
        assert (tmp_cwd / "f.txt").read_text() == content

    async def test_trailing_newline_preserved(self, tmp_cwd: Path) -> None:
        await write("f.txt", "last\n")
        assert (tmp_cwd / "f.txt").read_text() == "last\n"

    async def test_no_trailing_newline_preserved(self, tmp_cwd: Path) -> None:
        await write("f.txt", "no newline")
        assert (tmp_cwd / "f.txt").read_text() == "no newline"


class TestWriteFilePermissions:
    async def test_default_mode_644(self, tmp_cwd: Path) -> None:
        await write("f.txt", "x")
        mode = stat.S_IMODE((tmp_cwd / "f.txt").stat().st_mode)
        assert mode == 0o644

    async def test_custom_mode(self, tmp_cwd: Path) -> None:
        await write_file(file_path="f.sh", content="#!/bin/sh", mode=0o755)
        mode = stat.S_IMODE((tmp_cwd / "f.sh").stat().st_mode)
        assert mode == 0o755
