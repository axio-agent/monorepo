"""Tests for PatchFile tool handler.

All line numbers are 1-indexed, both inclusive:
  from_line=2, to_line=4 replaces lines 2, 3, 4.
  Insert without deleting: to_line = from_line - 1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from axio_tools_local.patch_file import PatchFile


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Path:
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


async def patch(path: Path, from_line: int, to_line: int, content: str) -> str:
    handler = PatchFile(file_path=path.name, from_line=from_line, to_line=to_line, content=content)
    return await handler()


class TestPatchBasic:
    async def test_replace_middle_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\nline4\n")
        await patch(f, 2, 3, "replaced\n")
        assert f.read_text() == "line1\nreplaced\nline4\n"

    async def test_replace_first_line(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\n")
        await patch(f, 1, 1, "new1\n")
        assert f.read_text() == "new1\nline2\nline3\n"

    async def test_replace_last_line(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("line1\nline2\nline3\n")
        await patch(f, 3, 3, "new3\n")
        assert f.read_text() == "line1\nline2\nnew3\n"

    async def test_replace_all_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 1, 3, "x\ny\n")
        assert f.read_text() == "x\ny\n"

    async def test_replace_with_more_lines(self, tmp_cwd: Path) -> None:
        """Replacing 1 line with 3 lines."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "x\ny\nz\n")
        assert f.read_text() == "a\nx\ny\nz\nc\n"

    async def test_replace_with_fewer_lines(self, tmp_cwd: Path) -> None:
        """Replacing 3 lines with 1 line."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\nd\n")
        await patch(f, 1, 3, "only\n")
        assert f.read_text() == "only\nd\n"


class TestPatchInsertDelete:
    async def test_insert_without_deleting(self, tmp_cwd: Path) -> None:
        """to_line = from_line - 1 inserts before from_line without deleting."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 1, "inserted\n")
        assert f.read_text() == "a\ninserted\nb\nc\n"

    async def test_insert_at_start(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\n")
        await patch(f, 1, 0, "before_a\n")
        assert f.read_text() == "before_a\na\nb\n"

    async def test_append_to_end(self, tmp_cwd: Path) -> None:
        """Insert after last line: from_line = N+1, to_line = N."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\n")
        await patch(f, 3, 2, "appended\n")
        assert f.read_text() == "a\nb\nappended\n"

    async def test_delete_lines_empty_content(self, tmp_cwd: Path) -> None:
        """Empty content string deletes the specified lines."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "")
        assert f.read_text() == "a\nc\n"

    async def test_delete_multiple_lines(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\nd\n")
        await patch(f, 2, 3, "")
        assert f.read_text() == "a\nd\n"


class TestNewlineHandling:
    async def test_content_without_trailing_newline(self, tmp_cwd: Path) -> None:
        """Content without \\n must not corrupt next line."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "replaced")  # no trailing newline
        assert f.read_text() == "a\nreplaced\nc\n"

    async def test_multiline_content_no_trailing_newline(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "x\ny")  # no trailing newline on last line
        assert f.read_text() == "a\nx\ny\nc\n"

    async def test_adjacent_lines_not_corrupted(self, tmp_cwd: Path) -> None:
        """Lines before and after patch range are exactly preserved."""
        f = tmp_cwd / "f.txt"
        f.write_text("first\nsecond\nthird\nfourth\nfifth\n")
        await patch(f, 3, 3, "REPLACED\n")
        lines = f.read_text().splitlines()
        assert lines == ["first", "second", "REPLACED", "fourth", "fifth"]

    async def test_content_extra_trailing_newlines(self, tmp_cwd: Path) -> None:
        """Content with extra trailing newlines creates blank lines."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 2, 2, "x\n\n")
        assert f.read_text() == "a\nx\n\nc\n"

    async def test_single_line_file_replace(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("only line\n")
        await patch(f, 1, 1, "new line\n")
        assert f.read_text() == "new line\n"

    async def test_single_line_file_no_trailing_newline(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("only line")
        await patch(f, 1, 1, "replaced\n")
        assert f.read_text() == "replaced\n"

    async def test_file_no_trailing_newline_patch_middle(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc")  # c has no trailing newline
        await patch(f, 2, 2, "B\n")
        assert f.read_text() == "a\nB\nc"


class TestEdgeCases:
    async def test_file_not_found(self, tmp_cwd: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await patch(tmp_cwd / "missing.txt", 1, 1, "x")

    async def test_returns_bytes_written_message(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\n")
        result = await patch(f, 1, 1, "x\n")
        assert "bytes written" in result
        assert "f.txt" in result

    async def test_empty_file_append(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("")
        await patch(f, 1, 0, "new\n")
        assert f.read_text() == "new\n"

    async def test_indented_code_preserved(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.py"
        f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        await patch(f, 2, 2, "    return 42\n")
        assert f.read_text() == "def foo():\n    return 42\n\ndef bar():\n    return 2\n"

    async def test_unicode_content(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("привет\nмир\n")
        await patch(f, 1, 1, "hello\n")
        assert f.read_text() == "hello\nмир\n"

    async def test_repr(self) -> None:
        h = PatchFile(file_path="f.py", from_line=5, to_line=10, content="code")
        r = repr(h)
        assert "f.py" in r
        assert "5:10" in r
