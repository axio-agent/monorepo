import asyncio
import os
import subprocess
import sys
import tempfile


async def run_python(
    code: str,
    cwd: str = ".",
    timeout: int = 5,
    stdin: str | None = None,
) -> str:
    """Run a Python code snippet in a subprocess and return stdout/stderr.
    The code is written to a temp file and executed with the current
    interpreter. Use for calculations, data processing, or testing
    small scripts. Optionally pass stdin data. Non-zero exit codes
    and tracebacks are returned as-is."""

    def _blocking() -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                input=stdin if stdin is not None else None,
                stdin=subprocess.DEVNULL if stdin is None else None,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout: code exceeded {timeout}s]"
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

    return await asyncio.to_thread(_blocking)
