import asyncio
import os
from pathlib import Path

from axio.tool import ToolHandler


class PatchFile(ToolHandler):
    """Replace a range of lines in an existing file. Lines are 1-indexed:
    from_line and to_line are both inclusive (from_line=2, to_line=4 replaces
    lines 2, 3, 4). To insert without deleting, set to_line = from_line - 1.
    Always read the file first with indexed=True to get correct line numbers.
    Use this for surgical edits instead of rewriting the whole file with
    write_file."""

    file_path: str
    mode: int = 0o644
    from_line: int
    to_line: int
    content: str

    def __repr__(self) -> str:
        return (
            f"PatchFile(file_path={self.file_path!r},"
            f" lines={self.from_line}:{self.to_line}, content=<{len(self.content)} chars>)"
        )

    def _blocking(self) -> str:
        path = Path(os.getcwd()) / self.file_path
        if not path.is_file():
            raise FileNotFoundError(f"{self.file_path} is not a valid file")

        # read all lines
        with path.open("r") as f:
            lines = f.readlines()

        # content lines — preserve existing newlines
        content_lines = self.content.splitlines(keepends=True)
        if content_lines and not content_lines[-1].endswith("\n"):
            content_lines[-1] += "\n"

        # patch lines (from_line/to_line are 1-indexed, both inclusive)
        new_lines = lines[: self.from_line - 1] + content_lines + lines[self.to_line :]
        with path.open("w") as f:
            f.writelines(new_lines)
            return f"{f.tell()} bytes written to {self.file_path}"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._blocking)
