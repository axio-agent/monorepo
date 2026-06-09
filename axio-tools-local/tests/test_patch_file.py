"""Tests for PatchFile tool handler.

All line numbers are 1-indexed, both inclusive:
  from_line=2, to_line=4 replaces lines 2, 3, 4.
  Insert without deleting: to_line = from_line - 1.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from axio_tools_local.patch_file import patch_file
from axio_tools_local.read_file import read_file


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Generator[Path, None, None]:
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


async def patch(path: Path, from_line: int, to_line: int, content: str) -> str:
    return await patch_file(file_path=path.name, from_line=from_line, to_line=to_line, content=content)


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


class TestValidation:
    async def test_from_line_zero_rejected(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        with pytest.raises(ValueError, match="from_line=0"):
            await patch(f, 0, 1, "x")

    async def test_from_line_negative_rejected(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        with pytest.raises(ValueError, match="from_line=-1"):
            await patch(f, -1, 1, "x")

    async def test_from_line_beyond_end_rejected(self, tmp_cwd: Path) -> None:
        """from_line > N+1 is not allowed (N+1 is valid for append-insert)."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")  # 3 lines
        with pytest.raises(ValueError, match="from_line=5"):
            await patch(f, 5, 5, "x")

    async def test_to_line_beyond_end_rejected(self, tmp_cwd: Path) -> None:
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")  # 3 lines
        with pytest.raises(ValueError, match="to_line=4"):
            await patch(f, 2, 4, "x")

    async def test_gap_range_rejected(self, tmp_cwd: Path) -> None:
        """from_line > to_line + 1 is a gap range and must be rejected."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        with pytest.raises(ValueError, match="to_line=2"):
            await patch(f, 4, 2, "x")

    async def test_file_unchanged_on_validation_error(self, tmp_cwd: Path) -> None:
        """File must not be modified if validation fails."""
        f = tmp_cwd / "f.txt"
        original = "a\nb\nc\n"
        f.write_text(original)
        with pytest.raises(ValueError):
            await patch(f, 0, 1, "x")
        assert f.read_text() == original

    async def test_from_line_n_plus_1_allowed(self, tmp_cwd: Path) -> None:
        """from_line = N+1, to_line = N is the valid append-insert pattern."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")  # 3 lines; from_line=4, to_line=3 is valid
        await patch(f, 4, 3, "appended\n")
        assert f.read_text() == "a\nb\nc\nappended\n"


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


class TestDoublePatching:
    async def test_second_patch_rejected(self, tmp_cwd: Path) -> None:
        """Patching the same file twice without re-reading must raise RuntimeError."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 1, 1, "A\n")
        with pytest.raises(RuntimeError, match="already patched"):
            await patch(f, 2, 2, "B\n")

    async def test_read_clears_block(self, tmp_cwd: Path) -> None:
        """Re-reading the file after a patch allows patching again."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 1, 1, "A\n")
        await read_file(filename=f.name)
        await patch(f, 2, 2, "B\n")
        assert f.read_text() == "A\nB\nc\n"

    async def test_failed_patch_does_not_block(self, tmp_cwd: Path) -> None:
        """A patch that fails validation must not consume the single-patch slot."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        with pytest.raises(ValueError):
            await patch(f, 0, 1, "x")
        # No error — file was never successfully patched.
        await patch(f, 1, 1, "A\n")
        assert f.read_text() == "A\nb\nc\n"

    async def test_different_files_independent(self, tmp_cwd: Path) -> None:
        """Patching different files does not interfere with each other's slots."""
        a = tmp_cwd / "a.txt"
        b = tmp_cwd / "b.txt"
        a.write_text("a1\na2\n")
        b.write_text("b1\nb2\n")
        await patch_file(file_path=a.name, from_line=1, to_line=1, content="A1\n")
        await patch_file(file_path=b.name, from_line=1, to_line=1, content="B1\n")
        assert a.read_text() == "A1\na2\n"
        assert b.read_text() == "B1\nb2\n"

    async def test_rejected_second_patch_leaves_file_with_only_first_patch(self, tmp_cwd: Path) -> None:
        """When the second patch is rejected, the file must contain exactly the first patch result."""
        f = tmp_cwd / "f.txt"
        f.write_text("a\nb\nc\n")
        await patch(f, 1, 1, "A\n")
        with pytest.raises(RuntimeError):
            await patch(f, 2, 2, "B\n")
        assert f.read_text() == "A\nb\nc\n"


class TestNullBytesAndBinaryFiles:
    async def test_null_bytes_in_preserved_lines_survive(self, tmp_cwd: Path) -> None:
        """Null bytes in lines that are NOT being replaced must not be stripped."""
        f = tmp_cwd / "f.txt"
        f.write_bytes(b"line1\nline2\x00null\nline3\n")
        await patch(f, 1, 1, "replaced\n")
        result = f.read_bytes()
        assert b"line2\x00null" in result, "null byte in preserved line was stripped"
        assert result.startswith(b"replaced\n")

    async def test_binary_file_raises_clear_error(self, tmp_cwd: Path) -> None:
        """Patching a non-UTF-8 binary file must raise an error before any write."""
        f = tmp_cwd / "f.bin"
        f.write_bytes(bytes(range(128, 200)))
        original = f.read_bytes()
        with pytest.raises(UnicodeDecodeError):
            await patch(f, 1, 1, "hello\n")
        assert f.read_bytes() == original, "binary file was modified despite decode error"


class TestSymlinks:
    async def test_symlink_to_file_is_followed(self, tmp_cwd: Path) -> None:
        """patch_file follows symlinks and patches the target file."""
        target = tmp_cwd / "target.txt"
        link = tmp_cwd / "link.txt"
        target.write_text("a\nb\nc\n")
        link.symlink_to(target)
        await patch_file(file_path=link.name, from_line=2, to_line=2, content="B\n")
        assert target.read_text() == "a\nB\nc\n"
