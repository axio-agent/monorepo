import asyncio
import os

from axio.field import StrictStr


async def write_file(
    file_path: StrictStr,
    content: str,
    mode: int = 0o644,
) -> str:
    """Create or overwrite a file with the given content. Parent directories
    are created automatically. Use this for new files or full rewrites.
    For partial edits prefer patch_file instead."""

    def _blocking() -> str:
        path = os.path.join(os.getcwd(), file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, mode=mode)
        return f"Wrote {len(content)} bytes to {file_path}"

    return await asyncio.to_thread(_blocking)
