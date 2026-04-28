"""Tests for shell tool handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axio_tools_local.shell import shell


async def sh(command: str, **kwargs: Any) -> str:
    return await shell(command=command, **kwargs)


class TestShellBasic:
    async def test_echo(self) -> None:
        assert "hello" in await sh("echo hello")

    async def test_stderr_included(self) -> None:
        result = await sh("echo err >&2")
        assert "err" in result

    async def test_nonzero_exit_reported(self) -> None:
        result = await sh("exit 42")
        assert "exit code: 42" in result

    async def test_stdout_and_stderr_combined(self) -> None:
        result = await sh("echo out; echo err >&2")
        assert "out" in result
        assert "err" in result

    async def test_no_output_returns_sentinel(self) -> None:
        result = await sh("true")
        assert result == "(no output)"

    async def test_multiline_output(self) -> None:
        result = await sh("printf 'a\\nb\\nc\\n'")
        assert "a" in result
        assert "b" in result
        assert "c" in result


class TestShellCwd:
    async def test_cwd_affects_command(self, tmp_path: Path) -> None:
        result = await sh("pwd", cwd=str(tmp_path))
        assert str(tmp_path) in result

    async def test_relative_default_cwd(self) -> None:
        result = await sh("pwd")
        assert "/" in result


class TestShellStdin:
    async def test_stdin_devnull_by_default(self) -> None:
        """stdin must be /dev/null so subprocesses can't steal TUI key events."""
        result = await sh("cat", timeout=2)
        assert result == "(no output)"

    async def test_stdin_passthrough(self) -> None:
        result = await sh("cat", stdin="hello from stdin")
        assert "hello from stdin" in result

    async def test_stdin_multiline(self) -> None:
        result = await sh("wc -l", stdin="a\nb\nc\n")
        assert "3" in result


class TestShellTimeout:
    async def test_timeout_returns_message_not_exception(self) -> None:
        """Timeout must return a clean message, not raise TimeoutExpired."""
        result = await sh("sleep 10", timeout=1)
        assert "timeout" in result
        assert "1s" in result

    async def test_fast_command_not_timed_out(self) -> None:
        result = await sh("echo quick", timeout=5)
        assert "quick" in result
