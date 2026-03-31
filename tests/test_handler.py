"""Tests for handler builders."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

from axio_tools_docker.handler import build_sandbox_exec, build_sandbox_read, build_sandbox_write
from axio_tools_docker.manager import SandboxManager


async def test_exec_handler_forwards_to_manager() -> None:
    manager = SandboxManager()
    manager.exec = AsyncMock(return_value="output")  # type: ignore[method-assign]

    handler_cls = build_sandbox_exec(manager)
    instance = cast(type[Any], handler_cls)(command="echo hi", timeout=10)

    with patch.object(SandboxManager, "docker_available", return_value=True):
        result = await instance()

    assert result == "output"
    manager.exec.assert_awaited_once_with("echo hi", timeout=10)


async def test_write_handler_forwards_to_manager() -> None:
    manager = SandboxManager()
    manager.write_file = AsyncMock(return_value="Wrote /workspace/test.py")  # type: ignore[method-assign]

    handler_cls = build_sandbox_write(manager)
    instance = cast(type[Any], handler_cls)(path="/workspace/test.py", content="print('hi')")

    with patch.object(SandboxManager, "docker_available", return_value=True):
        result = await instance()

    assert result == "Wrote /workspace/test.py"
    manager.write_file.assert_awaited_once_with("/workspace/test.py", "print('hi')")


async def test_read_handler_forwards_to_manager() -> None:
    manager = SandboxManager()
    manager.read_file = AsyncMock(return_value="file content")  # type: ignore[method-assign]

    handler_cls = build_sandbox_read(manager)
    instance = cast(type[Any], handler_cls)(path="/workspace/test.py")

    with patch.object(SandboxManager, "docker_available", return_value=True):
        result = await instance()

    assert result == "file content"
    manager.read_file.assert_awaited_once_with("/workspace/test.py")


async def test_exec_handler_returns_error_when_docker_missing() -> None:
    manager = SandboxManager()
    handler_cls = build_sandbox_exec(manager)
    instance = cast(type[Any], handler_cls)(command="echo hi")

    with patch.object(SandboxManager, "docker_available", return_value=False):
        result = await instance()

    assert "not installed" in result


async def test_write_handler_returns_error_when_docker_missing() -> None:
    manager = SandboxManager()
    handler_cls = build_sandbox_write(manager)
    instance = cast(type[Any], handler_cls)(path="/test", content="x")

    with patch.object(SandboxManager, "docker_available", return_value=False):
        result = await instance()

    assert "not installed" in result


async def test_read_handler_returns_error_when_docker_missing() -> None:
    manager = SandboxManager()
    handler_cls = build_sandbox_read(manager)
    instance = cast(type[Any], handler_cls)(path="/test")

    with patch.object(SandboxManager, "docker_available", return_value=False):
        result = await instance()

    assert "not installed" in result


def test_exec_handler_is_tool_handler_subclass() -> None:
    from axio.tool import ToolHandler

    manager = SandboxManager()
    handler_cls = build_sandbox_exec(manager)
    assert issubclass(handler_cls, ToolHandler)


def test_handlers_have_docstrings() -> None:
    manager = SandboxManager()
    for builder in (build_sandbox_exec, build_sandbox_read, build_sandbox_write):
        handler_cls = builder(manager)
        assert handler_cls.__doc__ is not None
        assert len(handler_cls.__doc__) > 10
