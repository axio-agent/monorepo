"""Tests for read_file tool handler."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from axio.blocks import ImageBlock, TextBlock, VideoBlock

from axio_tools_local.read_file import read_file


@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> Generator[Path, None, None]:
    old = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old)


async def read(tmp_cwd: Path, filename: str, **kwargs: Any) -> str:
    result = await read_file(filename=filename, **kwargs)
    assert isinstance(result, str)
    return result


class TestReadFilePlain:
    """line_numbers=False (default) - plain text, no line numbers."""

    async def test_single_line(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("content here")
        assert await read(tmp_cwd, "f.txt") == "content here"

    async def test_multiline(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\n")
        assert await read(tmp_cwd, "f.txt") == "a\nb\nc\n"

    async def test_no_trailing_newline_preserved(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("no newline")
        assert await read(tmp_cwd, "f.txt") == "no newline"

    async def test_empty_file(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("")
        assert await read(tmp_cwd, "f.txt") == ""

    async def test_indentation_preserved(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.py").write_text("def f():\n    pass\n")
        result = await read(tmp_cwd, "f.py")
        assert "    pass" in result

    async def test_unicode(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("привет\nмир\n")
        result = await read(tmp_cwd, "f.txt")
        assert "привет" in result
        assert "мир" in result


class TestReadFileIndexed:
    """line_numbers=True - each line prefixed with 1-based number."""

    async def test_first_line_is_1(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("hello\n")
        result = await read(tmp_cwd, "f.txt", line_numbers=True)
        assert result.startswith("1\t")

    async def test_numbers_are_one_indexed(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\n")
        result = await read(tmp_cwd, "f.txt", line_numbers=True)
        lines = result.splitlines()
        assert lines[0] == "1\ta"
        assert lines[1] == "2\tb"
        assert lines[2] == "3\tc"

    async def test_no_line_numbers_without_indexed(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\n")
        result = await read(tmp_cwd, "f.txt")
        assert not any(line[0].isdigit() for line in result.splitlines())

    async def test_single_line_no_trailing_newline(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("only")
        result = await read(tmp_cwd, "f.txt", line_numbers=True)
        assert result == "1\tonly"

    async def test_indentation_preserved_with_index(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.py").write_text("def f():\n    pass\n")
        result = await read(tmp_cwd, "f.py", line_numbers=True)
        assert "2\t    pass" in result


class TestReadFileRange:
    async def test_start_line_1_is_first(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\n")
        result = await read(tmp_cwd, "f.txt", start_line=1)
        assert "a" in result and "b" in result and "c" in result

    async def test_start_line_skips_before(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\nd\n")
        result = await read(tmp_cwd, "f.txt", start_line=3)
        assert "a" not in result
        assert "b" not in result
        assert "c" in result
        assert "d" in result

    async def test_end_line_is_inclusive(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\nd\n")
        result = await read(tmp_cwd, "f.txt", end_line=2)
        assert "a" in result
        assert "b" in result
        assert "c" not in result
        assert "d" not in result

    async def test_start_and_end_line(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\nd\ne\n")
        result = await read(tmp_cwd, "f.txt", start_line=2, end_line=4)
        assert "a" not in result
        assert "b" in result
        assert "c" in result
        assert "d" in result
        assert "e" not in result

    async def test_indexed_numbers_reflect_file_position(self, tmp_cwd: Path) -> None:
        """Line numbers must reflect position in the file, not position in the slice."""
        lines = [f"line{i}\n" for i in range(1, 11)]
        (tmp_cwd / "f.txt").write_text("".join(lines))
        result = await read(tmp_cwd, "f.txt", start_line=5, end_line=7, line_numbers=True)
        out = result.splitlines()
        assert out[0] == "5\tline5"
        assert out[1] == "6\tline6"
        assert out[2] == "7\tline7"

    async def test_single_line_range(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a\nb\nc\n")
        result = await read(tmp_cwd, "f.txt", start_line=2, end_line=2)
        assert result.strip() == "b"


class TestReadFileTruncation:
    async def test_default_limit_rejects_large_file(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a" * 32769)
        with pytest.raises(ValueError, match="explicit max_chars"):
            await read_file(filename="f.txt")

    async def test_truncated_at_max_chars(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("a" * 200)
        result = await read(tmp_cwd, "f.txt", max_chars=10)
        assert "[truncated]" in result

    async def test_no_truncation_within_limit(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("hello\n")
        result = await read(tmp_cwd, "f.txt")
        assert "[truncated]" not in result

    async def test_line_range_can_read_small_slice_from_large_file(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "f.txt").write_text("first\n" + ("a" * 40000) + "\nlast\n")
        result = await read(tmp_cwd, "f.txt", start_line=1, end_line=1)
        assert result == "first\n"


class TestReadFileBinary:
    async def test_binary_as_hex(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "b.dat").write_bytes(b"\x80\x81\xff")
        result = await read(tmp_cwd, "b.dat", binary_as_hex=True)
        assert "8081ff" in result

    async def test_binary_hex_truncated_to_max_chars(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "b.dat").write_bytes(bytes(range(256)))
        result = await read(tmp_cwd, "b.dat", binary_as_hex=True, max_chars=20)
        assert len(result) < 600
        assert "[truncated]" in result

    async def test_binary_default_limit_rejects_large_file(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "b.dat").write_bytes(b"\xff" * 16385)
        with pytest.raises(ValueError, match="explicit max_chars"):
            await read_file(filename="b.dat", binary_as_hex=True)

    async def test_binary_raises_without_hex(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "b.dat").write_bytes(b"\x80\x81\xff")
        with pytest.raises(UnicodeDecodeError):
            await read_file(filename="b.dat", binary_as_hex=False)


class TestReadFileImage:
    async def test_png_returns_image_block(self, tmp_cwd: Path) -> None:
        img_data = b"\x89PNG\r\n\x1a\nfake-png-data"
        (tmp_cwd / "photo.png").write_bytes(img_data)
        result = await read_file(filename="photo.png")
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], TextBlock)
        assert "photo.png" in result[0].text
        assert isinstance(result[1], ImageBlock)
        assert result[1].media_type == "image/png"
        assert result[1].data == img_data

    async def test_jpeg_returns_image_block(self, tmp_cwd: Path) -> None:
        img_data = b"\xff\xd8\xff\xe0fake-jpeg"
        (tmp_cwd / "photo.jpg").write_bytes(img_data)
        result = await read_file(filename="photo.jpg")
        assert isinstance(result, list)
        assert isinstance(result[1], ImageBlock)
        assert result[1].media_type == "image/jpeg"

    async def test_webp_returns_image_block(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "img.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        result = await read_file(filename="img.webp")
        assert isinstance(result, list)
        assert isinstance(result[1], ImageBlock)
        assert result[1].media_type == "image/webp"

    async def test_large_image_returns_text_only(self, tmp_cwd: Path) -> None:
        """Images larger than 20MB should not be inlined."""
        large = b"\x89PNG" + b"\x00" * (20 * 1024 * 1024 + 1)
        (tmp_cwd / "huge.png").write_bytes(large)
        result = await read_file(filename="huge.png")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextBlock)
        assert "too large" in result[0].text


class TestReadFileVideo:
    async def test_mp4_returns_video_block(self, tmp_cwd: Path) -> None:
        video_data = b"\x00\x00\x00\x1cftypisom"
        (tmp_cwd / "clip.mp4").write_bytes(video_data)
        result = await read_file(filename="clip.mp4")
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], TextBlock)
        assert "clip.mp4" in result[0].text
        assert isinstance(result[1], VideoBlock)
        assert result[1].media_type == "video/mp4"
        assert result[1].data == video_data

    async def test_webm_returns_video_block(self, tmp_cwd: Path) -> None:
        (tmp_cwd / "clip.webm").write_bytes(b"\x1aE\xdf\xa3webm")
        result = await read_file(filename="clip.webm")
        assert isinstance(result, list)
        assert isinstance(result[1], VideoBlock)
        assert result[1].media_type == "video/webm"

    async def test_large_video_returns_text_only(self, tmp_cwd: Path) -> None:
        large = b"\x00" * (20 * 1024 * 1024 + 1)
        (tmp_cwd / "huge.mp4").write_bytes(large)
        result = await read_file(filename="huge.mp4")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextBlock)
        assert "too large" in result[0].text


class TestReadFileMisc:
    async def test_file_not_found(self, tmp_cwd: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await read(tmp_cwd, "nope.txt")
