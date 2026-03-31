import asyncio
import subprocess

from axio.tool import ToolHandler

from . import _short


class Shell(ToolHandler):
    """Run a shell command and return combined stdout/stderr. Use for git,
    build tools, grep, tests, or any CLI operation. Non-zero exit codes
    are reported. Optionally pass stdin data for commands that read from
    standard input. Prefer short timeouts and avoid interactive commands."""

    command: str
    timeout: int = 5
    cwd: str = "."
    stdin: str | None = None

    def __repr__(self) -> str:
        return f"Shell(command={_short(self.command)!r}, cwd={self.cwd!r})"

    def _blocking(self) -> str:
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
                input=self.stdin if self.stdin is not None else None,
                stdin=subprocess.DEVNULL if self.stdin is None else None,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout: command exceeded {self.timeout}s]"
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
