import asyncio
import subprocess

from axio.field import StrictStr


async def shell(
    command: StrictStr,
    timeout: int = 5,
    cwd: StrictStr = ".",
    stdin: str | None = None,
) -> str:
    """Run a shell command and return combined stdout/stderr. Use for git,
    build tools, grep, tests, or any CLI operation. Non-zero exit codes
    are reported. Optionally pass stdin data for commands that read from
    standard input. Prefer short timeouts and avoid interactive commands."""

    def _blocking() -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                input=stdin if stdin is not None else None,
                stdin=subprocess.DEVNULL if stdin is None else None,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout: command exceeded {timeout}s]"
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"

    return await asyncio.to_thread(_blocking)
