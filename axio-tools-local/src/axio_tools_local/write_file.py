import asyncio
import os

from axio.field import StrictStr


async def write_file(
    path: StrictStr,
    content: str,
    mode: int = 0o644,
) -> str:
    """Create or overwrite a file with the given content. Parent directories
    are created automatically. Use this for new files or full rewrites.
    For partial edits prefer patch_file instead."""

    def _blocking() -> str:
        resolved = os.path.join(os.getcwd(), path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)
        os.chmod(resolved, mode=mode)
        return f"Wrote {len(content)} bytes to {path}"

    return await asyncio.to_thread(_blocking)
