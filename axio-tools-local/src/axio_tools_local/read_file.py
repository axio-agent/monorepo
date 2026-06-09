import asyncio
import os

from axio.blocks import AudioBlock, AudioMediaType, ImageBlock, ImageMediaType, TextBlock, VideoBlock, VideoMediaType
from axio.field import StrictStr

from .patch_file import clear_patched

type ReadFileResult = str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock]
type MediaFileContent = list[TextBlock | ImageBlock | AudioBlock | VideoBlock]

_AUDIO_EXTENSIONS: dict[str, AudioMediaType] = {
    ".aac": "audio/x-aac",
    ".flac": "audio/flac",
    ".mp3": "audio/mp3",
    ".m4a": "audio/m4a",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpga",
    ".ogg": "audio/ogg",
    ".pcm": "audio/pcm",
    ".wav": "audio/wav",
    ".weba": "audio/webm",
}

_IMAGE_EXTENSIONS: dict[str, ImageMediaType] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_VIDEO_EXTENSIONS: dict[str, VideoMediaType] = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpg",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
}

# 20 MB limit for inline media
_MAX_MEDIA_BYTES = 20 * 1024 * 1024


def _read_audio(path: str, filename: str, ext: str) -> MediaFileContent:
    size = os.path.getsize(path)
    if size > _MAX_MEDIA_BYTES:
        return [TextBlock(text=f"Audio file too large to inline ({size} bytes): {filename}")]
    with open(path, "rb") as f:
        data = f.read()
    media_type = _AUDIO_EXTENSIONS[ext]
    return [
        TextBlock(text=f"Audio file: {filename} ({len(data)} bytes)"),
        AudioBlock(media_type=media_type, data=data),
    ]


def _read_image(path: str, filename: str, ext: str) -> MediaFileContent:
    size = os.path.getsize(path)
    if size > _MAX_MEDIA_BYTES:
        return [TextBlock(text=f"Image file too large to inline ({size} bytes): {filename}")]
    with open(path, "rb") as f:
        data = f.read()
    media_type = _IMAGE_EXTENSIONS[ext]
    return [
        TextBlock(text=f"Image file: {filename} ({len(data)} bytes)"),
        ImageBlock(media_type=media_type, data=data),
    ]


def _read_video(path: str, filename: str, ext: str) -> MediaFileContent:
    size = os.path.getsize(path)
    if size > _MAX_MEDIA_BYTES:
        return [TextBlock(text=f"Video file too large to inline ({size} bytes): {filename}")]
    with open(path, "rb") as f:
        data = f.read()
    media_type = _VIDEO_EXTENSIONS[ext]
    return [
        TextBlock(text=f"Video file: {filename} ({len(data)} bytes)"),
        VideoBlock(media_type=media_type, data=data),
    ]


def _read_text(
    path: str,
    max_chars: int,
    binary_as_hex: bool,
    start_line: int | None,
    end_line: int | None,
    line_numbers: bool,
) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        if binary_as_hex:
            return "Encoded binary data HEX: " + raw[:max_chars].hex()
        raise
    all_lines = text.splitlines(keepends=True)
    start = 0 if start_line is None else start_line - 1
    end = len(all_lines) if end_line is None else end_line
    lines = all_lines[start:end]
    if line_numbers:
        result = "".join(f"{start + 1 + i}\t{line}" for i, line in enumerate(lines))
    else:
        result = "".join(lines)
    if len(result) > max_chars:
        return result[:max_chars] + "\n...[truncated]"
    return result


async def read_file(
    filename: StrictStr,
    max_chars: int = 32768,
    binary_as_hex: bool = True,
    start_line: int | None = None,
    end_line: int | None = None,
    line_numbers: bool = False,
) -> ReadFileResult:
    """Read file contents. Returns text for text files, hex for binaries.
    Audio files (mp3/wav/ogg/flac/etc.), image files (jpg/png/gif/webp) and
    video files (mp4/webm/etc.) are returned as multimodal content blocks for
    capable models. Lines are 1-indexed: start_line=1 is the first line,
    end_line=3 includes line 3. Pass line_numbers=True to prefix each line with
    its 1-based line number (tab-separated) — required before calling
    patch_file. Large files are truncated to max_chars. Always read the file
    before editing it with write_file or patch_file."""

    def _blocking() -> ReadFileResult:
        path = os.path.join(os.getcwd(), filename)
        clear_patched(os.path.realpath(path))
        ext = os.path.splitext(path)[1].lower()
        if ext in _AUDIO_EXTENSIONS:
            return _read_audio(path, filename, ext)
        if ext in _IMAGE_EXTENSIONS:
            return _read_image(path, filename, ext)
        if ext in _VIDEO_EXTENSIONS:
            return _read_video(path, filename, ext)
        return _read_text(path, max_chars, binary_as_hex, start_line, end_line, line_numbers)

    return await asyncio.to_thread(_blocking)
