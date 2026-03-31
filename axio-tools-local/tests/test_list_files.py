"""Tests for ListFiles tool handler."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from axio_tools_local.list_files import ListFiles


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Path:
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


async def ls(directory: str = ".") -> str:
    return await ListFiles(directory=directory)()


class TestListFilesBasic:
    async def test_lists_files_and_dirs(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "a.txt").write_text("a")
        (tmp_cwd / "b.txt").write_text("b")
        (tmp_cwd / "subdir").mkdir()
        result = await ls()
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir/" in result

    async def test_dirs_listed_before_files(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "z_file.txt").write_text("z")
        (tmp_cwd / "a_dir").mkdir()
        result = await ls()
        lines = result.splitlines()
        dir_idx = next(i for i, l in enumerate(lines) if "a_dir/" in l)
        file_idx = next(i for i, l in enumerate(lines) if "z_file.txt" in l)
        assert dir_idx < file_idx

    async def test_empty_directory(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "empty").mkdir()
        result = await ls("empty")
        assert result == "(empty directory)"

    async def test_default_directory_is_dot(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "file.txt").write_text("x")
        result = await ls()
        assert "file.txt" in result

    async def test_subdirectory_path(self, tmp_cwd: Path) -> None:
        sub = tmp_cwd / "sub"
        sub.mkdir()
        (sub / "inner.txt").write_text("x")
        result = await ls("sub")
        assert "inner.txt" in result

    async def test_not_a_directory_raises(self, tmp_cwd: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await ls("nope")

    async def test_shows_permissions(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("x")
        result = await ls()
        # permissions line starts with '-' for files
        assert any(line.startswith("-") for line in result.splitlines())

    async def test_shows_size(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("hello")
        result = await ls()
        assert "5" in result

    async def test_trailing_slash_on_dirs_only(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "mydir").mkdir()
        (tmp_cwd / "myfile.txt").write_text("x")
        result = await ls()
        assert "mydir/" in result
        assert "myfile.txt/" not in result


class TestListFilesSymlinks:
    async def test_broken_symlink_does_not_crash(self, tmp_cwd: Path) -> None:
        """Broken symlink must produce a [broken symlink] entry, not an exception."""
        link = tmp_cwd / "dead_link"
        link.symlink_to(tmp_cwd / "nonexistent_target")
        result = await ls()
        assert "dead_link" in result
        assert "[broken symlink]" in result

    async def test_valid_symlink_to_file(self, tmp_cwd: Path) -> None:
        target = tmp_cwd / "real.txt"
        target.write_text("x")
        (tmp_cwd / "link.txt").symlink_to(target)
        result = await ls()
        assert "link.txt" in result
        assert "[broken symlink]" not in result

    async def test_mixed_broken_and_valid(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "good.txt").write_text("x")
        (tmp_cwd / "bad_link").symlink_to(tmp_cwd / "nowhere")
        result = await ls()
        assert "good.txt" in result
        assert "bad_link" in result
        assert "[broken symlink]" in result


class TestListFilesMisc:
    async def test_repr(self) -> None:
        assert "." in repr(ListFiles())
        assert "src" in repr(ListFiles(directory="src"))
