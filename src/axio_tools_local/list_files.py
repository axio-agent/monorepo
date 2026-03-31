import asyncio
import os
import stat as stat_module
from datetime import datetime
from pathlib import Path

from axio.tool import ToolHandler


class ListFiles(ToolHandler):
    """List files and directories. Shows permissions, size, modification time,
    and name for each entry. Directories are listed first and marked with
    a trailing slash. Use this to explore the project structure before
    reading or editing files."""

    directory: str = "."

    def __repr__(self) -> str:
        return f"ListFiles(directory={self.directory!r})"

    def _blocking(self) -> str:
        path = Path(os.getcwd()) / self.directory
        if not path.is_dir():
            raise FileNotFoundError(f"{self.directory} is not a valid directory")
        lines: list[str] = []
        for fpath in sorted(path.glob("*"), key=lambda p: (not p.is_dir(), p.name)):
            try:
                st = fpath.stat()
            except OSError:
                lines.append(f"{'?':10} {'?':>8} {'?':12} {fpath.name} [broken symlink]")
                continue
            mode = stat_module.filemode(st.st_mode)
            size = st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%b %d %H:%M")
            name = fpath.name + ("/" if fpath.is_dir() else "")
            lines.append(f"{mode} {size:>8} {mtime} {name}")
        return "\n".join(lines) or "(empty directory)"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._blocking)
