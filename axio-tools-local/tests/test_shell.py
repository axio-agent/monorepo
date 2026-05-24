"""Tests for shell tool handler."""

from __future__ import annotations

import re
import shlex
import sys
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any, cast

from axio_tools_local.shell import shell


async def sh(command: str, **kwargs: Any) -> str:
    return await shell(command=command, **kwargs)


def shell_stream(**kwargs: Any) -> AsyncGenerator[tuple[str, str], None]:
    stream = cast(Callable[..., AsyncGenerator[tuple[str, str], None]], getattr(shell, "stream"))
    return stream(**kwargs)


def large_output_command(line_chars: int = 300, lines: int = 20) -> str:
    code = f"for i in range({lines}): print(f'line{{i:02d}}-' + 'x' * {line_chars})"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def failing_large_output_command() -> str:
    code = "import sys\nfor i in range(20): print(f'line{i:02d}-' + 'x' * 300)\nsys.exit(7)"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


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

    async def test_output_text_matching_large_output_notice_is_not_special(self) -> None:
        result = await sh("printf '[output is too large, not internal\\nkeep\\n'")
        assert "[output is too large, not internal" in result
        assert result.index("[output is too large, not internal") < result.index("keep")

    async def test_large_output_saved_to_file(self, tmp_path: Path) -> None:
        result = await sh(large_output_command(), cwd=str(tmp_path))
        match = re.search(r"saved to ([^;]+);", result)
        assert match is not None
        output_path = Path(match.group(1))
        assert output_path.parent == tmp_path
        assert output_path.exists()
        assert "output is too large" in result
        assert "showing first 5 and last 5 lines within 4096 chars" in result
        assert len(result) <= 4096
        for i in range(5):
            assert f"line{i:02d}-" in result
        for i in range(5, 15):
            assert f"line{i:02d}-" not in result
        for i in range(15, 20):
            assert f"line{i:02d}-" in result
        saved = output_path.read_text()
        assert sum(1 for line in saved.splitlines() if "stdout] line" in line) == 20
        assert "inline output cap: 4096 chars" in saved

    async def test_large_output_nonzero_keeps_five_tail_lines(self, tmp_path: Path) -> None:
        result = await sh(failing_large_output_command(), cwd=str(tmp_path))
        assert "exit code: 7" in result
        for i in range(15, 20):
            assert f"line{i:02d}-" in result
        assert len(result) <= 4096

    async def test_large_output_does_not_duplicate_threshold_crossing_tail(self, tmp_path: Path) -> None:
        result = await sh(large_output_command(line_chars=900, lines=6), cwd=str(tmp_path))
        assert result.count("line05-") == 1
        assert len(result) <= 4096


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


class TestShellStreaming:
    async def test_stream_yields_keyed_lines(self) -> None:
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command="printf 'a\\nb\\nc\\n'"):
            chunks.append(chunk)
        assert len(chunks) == 3
        assert all(k == "stdout" for k, _ in chunks)
        assert chunks[0] == ("stdout", "a\n")
        assert chunks[1] == ("stdout", "b\n")
        assert chunks[2] == ("stdout", "c\n")

    async def test_stream_yields_initial_output_before_process_exits(self) -> None:
        code = "import time; print('start', flush=True); time.sleep(0.5); print('end', flush=True)"
        started = time.monotonic()
        first_elapsed: float | None = None
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}", timeout=2):
            chunks.append(chunk)
            if chunk == ("stdout", "start\n") and first_elapsed is None:
                first_elapsed = time.monotonic() - started
        assert first_elapsed is not None
        assert first_elapsed < 0.4
        assert chunks == [("stdout", "start\n"), ("stdout", "end\n")]

    async def test_stream_yields_small_output_after_first_five_lines_before_process_exits(self) -> None:
        code = "import time\nfor i in range(7):\n print(f'line{i}', flush=True)\n time.sleep(0.2)"
        started = time.monotonic()
        line5_elapsed: float | None = None
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}", timeout=3):
            chunks.append(chunk)
            if chunk == ("stdout", "line5\n") and line5_elapsed is None:
                line5_elapsed = time.monotonic() - started
        assert line5_elapsed is not None
        assert line5_elapsed < 1.3
        assert chunks == [(("stdout", f"line{i}\n")) for i in range(7)]

    async def test_stream_stderr_separate_key(self) -> None:
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command="echo out; echo err >&2"):
            chunks.append(chunk)
        stdout = [t for k, t in chunks if k == "stdout"]
        stderr = [t for k, t in chunks if k == "stderr"]
        assert any("out" in t for t in stdout)
        assert any("err" in t for t in stderr)

    async def test_stream_exit_code_on_stderr_key(self) -> None:
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command="echo out; exit 1"):
            chunks.append(chunk)
        stdout = [t for k, t in chunks if k == "stdout"]
        stderr = [t for k, t in chunks if k == "stderr"]
        assert any("out" in t for t in stdout)
        assert any("exit code: 1" in t for t in stderr)

    async def test_stream_timeout(self) -> None:
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command="sleep 10", timeout=1):
            chunks.append(chunk)
        all_text = "".join(t for _, t in chunks)
        assert "timeout" in all_text
        assert "1s" in all_text

    async def test_stream_no_output(self) -> None:
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command="true"):
            chunks.append(chunk)
        assert chunks == []

    async def test_call_structured_log_records(self) -> None:
        """shell() produces timestamped log records with stream keys."""
        result = await shell(command="echo out; echo err >&2")
        assert "stdout]" in result
        assert "stderr]" in result
        assert "out" in result
        assert "err" in result

    async def test_stream_caps_large_output(self, tmp_path: Path) -> None:
        chunks: list[tuple[str, str]] = []
        async for chunk in shell_stream(command=large_output_command(), cwd=str(tmp_path)):
            chunks.append(chunk)
        assert {key for key, _ in chunks} <= {"stdout", "stderr"}
        text = "".join(t for _, t in chunks)
        assert "output is too large" in text
        assert "line00-" in text
        assert "line19-" in text
        assert sum(1 for line in text.splitlines() if line.startswith("line")) < 20
        match = re.search(r"saved to ([^;]+);", text)
        assert match is not None
        assert Path(match.group(1)).exists()

    async def test_stream_head_tail_content_stays_within_budget(self, tmp_path: Path) -> None:
        result = await sh(large_output_command(line_chars=1000), cwd=str(tmp_path))
        assert len(result) <= 4096
        assert "line00-" in result
        assert "line19-" in result
        assert "[truncated]" in result
