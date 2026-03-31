import asyncio
import os
import subprocess
import sys
import tempfile

from axio.tool import ToolHandler

from . import _short


class RunPython(ToolHandler):
    """Run a Python code snippet in a subprocess and return stdout/stderr.
    The code is written to a temp file and executed with the current
    interpreter. Use for calculations, data processing, or testing
    small scripts. Optionally pass stdin data. Non-zero exit codes
    and tracebacks are returned as-is."""

    code: str
    cwd: str = "."
    timeout: int = 5
    stdin: str | None = None

    def __repr__(self) -> str:
        return f"RunPython(code={_short(self.code)!r}, cwd={self.cwd!r})"

    def _blocking(self) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
            f.write(self.code)
            f.flush()

            path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.cwd,
                    input=self.stdin if self.stdin is not None else None,
                    stdin=subprocess.DEVNULL if self.stdin is None else None,
                )
            except subprocess.TimeoutExpired:
                return f"[timeout: code exceeded {self.timeout}s]"
            finally:
                os.unlink(path)
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._blocking)
