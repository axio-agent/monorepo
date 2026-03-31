import asyncio
import os

from axio.tool import ToolHandler


class WriteFile(ToolHandler):
    """Create or overwrite a file with the given content. Parent directories
    are created automatically. Use this for new files or full rewrites.
    For partial edits prefer patch_file instead."""

    file_path: str
    content: str
    mode: int = 0o644

    def __repr__(self) -> str:
        return f"WriteFile(file_path={self.file_path!r}, content=<{len(self.content)} chars>)"

    def _blocking(self) -> str:
        path = os.path.join(os.getcwd(), self.file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.content)
        os.chmod(path, mode=self.mode)
        return f"Wrote {len(self.content)} bytes to {self.file_path}"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._blocking)
