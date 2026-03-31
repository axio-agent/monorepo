"""Tests for RunPython tool handler."""

from __future__ import annotations

import tempfile
from pathlib import Path

from axio_tools_local.run_python import RunPython


async def run(code: str, **kwargs: object) -> str:
    return await RunPython(code=code, **kwargs)()


class TestRunPythonBasic:
    async def test_simple_print(self) -> None:
        assert "hello" in await run("print('hello')")

    async def test_multiline_code(self) -> None:
        code = "x = 1 + 2\nprint(x)"
        assert "3" in await run(code)

    async def test_no_output_returns_sentinel(self) -> None:
        assert await run("x = 1") == "(no output)"

    async def test_stdout_returned(self) -> None:
        result = await run("print('line1'); print('line2')")
        assert "line1" in result
        assert "line2" in result


class TestRunPythonErrors:
    async def test_exception_in_stderr(self) -> None:
        result = await run("raise ValueError('boom')")
        assert "ValueError" in result
        assert "exit code:" in result

    async def test_syntax_error(self) -> None:
        result = await run("def (broken")
        assert "SyntaxError" in result
        assert "exit code:" in result

    async def test_stderr_included(self) -> None:
        result = await run("import sys; sys.stderr.write('err output\\n')")
        assert "err output" in result

    async def test_nonzero_exit_reported(self) -> None:
        result = await run("import sys; sys.exit(42)")
        assert "exit code: 42" in result

    async def test_zero_exit_not_reported(self) -> None:
        result = await run("print('ok')")
        assert "exit code" not in result


class TestRunPythonStdin:
    async def test_stdin_devnull_by_default(self) -> None:
        """stdin must be /dev/null so subprocesses can't steal TUI key events."""
        result = await run("import sys; data = sys.stdin.read(); print(repr(data))")
        assert "''" in result

    async def test_stdin_passthrough(self) -> None:
        result = await run("import sys; print(sys.stdin.read())", stdin="hello from stdin")
        assert "hello from stdin" in result

    async def test_stdin_multiline(self) -> None:
        result = await run("import sys; print(len(sys.stdin.readlines()))", stdin="a\nb\nc\n")
        assert "3" in result


class TestRunPythonTimeout:
    async def test_timeout_returns_message_not_exception(self) -> None:
        """Timeout must return a clean message, not raise TimeoutExpired."""
        result = await run("import time; time.sleep(10)", timeout=1)
        assert "timeout" in result
        assert "1s" in result

    async def test_fast_code_not_timed_out(self) -> None:
        result = await run("print('fast')", timeout=5)
        assert "fast" in result


class TestRunPythonCwd:
    async def test_cwd_affects_execution(self, tmp_path: Path) -> None:
        result = await run("import os; print(os.getcwd())", cwd=str(tmp_path))
        assert str(tmp_path) in result


class TestRunPythonCleanup:
    async def test_temp_file_deleted_after_run(self) -> None:
        """The temp .py file must be cleaned up after execution."""
        before = set(Path(tempfile.gettempdir()).glob("tmp*.py"))
        await run("print('x')")
        after = set(Path(tempfile.gettempdir()).glob("tmp*.py"))
        assert after == before

    async def test_temp_file_deleted_on_timeout(self) -> None:
        """The temp file must be cleaned up even when execution times out."""
        before = set(Path(tempfile.gettempdir()).glob("tmp*.py"))
        await run("import time; time.sleep(10)", timeout=1)
        after = set(Path(tempfile.gettempdir()).glob("tmp*.py"))
        assert after == before


class TestRunPythonMisc:
    async def test_repr_contains_code(self) -> None:
        assert "x = 1" in repr(RunPython(code="x = 1"))

    async def test_repr_truncates_long_code(self) -> None:
        long_code = "x = " + "1" * 100
        r = repr(RunPython(code=long_code))
        assert len(r) < 200
