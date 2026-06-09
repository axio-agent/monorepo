import asyncio
import os
import tempfile
import threading
from pathlib import Path

from axio.field import StrictStr

# Tracks absolute paths patched since the last read_file call for that path.
# Cleared per-path when read_file reads the file; prevents double-patching
# with stale line numbers within a single agent turn.
_patched_files: set[str] = set()
_patched_files_lock = threading.Lock()


def _mark_patched(path: str) -> None:
    with _patched_files_lock:
        _patched_files.add(path)


def _check_and_mark(path: str) -> None:
    with _patched_files_lock:
        if path in _patched_files:
            raise RuntimeError(
                f"{path!r} was already patched since the last read. "
                "Re-read the file with line_numbers=True to get updated line numbers before patching again."
            )
        _patched_files.add(path)


def clear_patched(path: str) -> None:
    """Remove path from the patched-files tracker. Called by read_file."""
    with _patched_files_lock:
        _patched_files.discard(path)


async def patch_file(
    file_path: StrictStr,
    from_line: int,
    to_line: int,
    content: str,
    mode: int = 0o644,
) -> str:
    """Replace a range of lines in an existing file. Lines are 1-indexed:
    from_line and to_line are both inclusive (from_line=2, to_line=4 replaces
    lines 2, 3, 4). To insert without deleting, set to_line = from_line - 1.
    Always read the file first with line_numbers=True to get correct line numbers.
    Patch each file at most once per read — re-read with line_numbers=True after
    patching before issuing another patch to the same file.
    Use this for surgical edits instead of rewriting the whole file with
    write_file."""

    def _blocking() -> str:
        # Resolve symlinks so reads and writes go to the real file.
        path = (Path(os.getcwd()) / file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file")

        resolved = str(path)
        _check_and_mark(resolved)
        try:
            with path.open("r") as f:
                lines = f.readlines()

            n = len(lines)
            if not (1 <= from_line <= n + 1):
                raise ValueError(
                    f"from_line={from_line} out of range; file has {n} lines (valid: 1..{n + 1})"
                )
            if not (from_line - 1 <= to_line <= n):
                raise ValueError(
                    f"to_line={to_line} out of range (valid: {from_line - 1}..{n})"
                )

            content_lines = content.splitlines(keepends=True)
            if content_lines and not content_lines[-1].endswith("\n"):
                content_lines[-1] += "\n"

            new_lines = lines[: from_line - 1] + content_lines + lines[to_line:]
            content_str = "".join(new_lines)

            fd, tmp_path_str = tempfile.mkstemp(dir=path.parent)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(content_str)
                    byte_count = f.tell()
                os.chmod(tmp_path_str, mode)
                os.replace(tmp_path_str, path)
            except Exception:
                try:
                    os.unlink(tmp_path_str)
                except OSError:
                    pass
                raise
        except Exception:
            # Roll back the "patched" marker so the caller can still read and retry.
            clear_patched(resolved)
            raise

        return f"{byte_count} bytes written to {file_path}"

    return await asyncio.to_thread(_blocking)
