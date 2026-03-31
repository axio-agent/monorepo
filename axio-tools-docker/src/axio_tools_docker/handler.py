"""Dynamic ToolHandler builders for Docker sandbox tools."""

from __future__ import annotations

from typing import ClassVar

from axio.tool import ToolHandler

from .manager import SandboxManager


def build_sandbox_exec(manager: SandboxManager) -> type[ToolHandler]:
    """Create a ToolHandler that executes commands in the sandbox."""

    class SandboxExecHandler(ToolHandler):
        """Execute a shell command inside a Docker sandbox container.
        Returns combined stdout/stderr. Non-zero exit codes are reported.
        Use for running code, tests, or CLI tools in an isolated environment."""

        command: str
        timeout: int = 30

        _manager: ClassVar[SandboxManager]

        async def __call__(self) -> str:
            if not self._manager.docker_available():
                return "Error: Docker is not installed or not on PATH"
            return await self._manager.exec(self.command, timeout=self.timeout)

    SandboxExecHandler._manager = manager
    return SandboxExecHandler


def build_sandbox_write(manager: SandboxManager) -> type[ToolHandler]:
    """Create a ToolHandler that writes files in the sandbox."""

    class SandboxWriteHandler(ToolHandler):
        """Write content to a file inside the Docker sandbox container.
        Creates parent directories as needed. Use for placing source code,
        config files, or test data in the sandbox."""

        path: str
        content: str

        _manager: ClassVar[SandboxManager]

        async def __call__(self) -> str:
            if not self._manager.docker_available():
                return "Error: Docker is not installed or not on PATH"
            return await self._manager.write_file(self.path, self.content)

    SandboxWriteHandler._manager = manager
    return SandboxWriteHandler


def build_sandbox_read(manager: SandboxManager) -> type[ToolHandler]:
    """Create a ToolHandler that reads files from the sandbox."""

    class SandboxReadHandler(ToolHandler):
        """Read the contents of a file from the Docker sandbox container.
        Returns the full file content as text."""

        path: str

        _manager: ClassVar[SandboxManager]

        async def __call__(self) -> str:
            if not self._manager.docker_available():
                return "Error: Docker is not installed or not on PATH"
            return await self._manager.read_file(self.path)

    SandboxReadHandler._manager = manager
    return SandboxReadHandler
