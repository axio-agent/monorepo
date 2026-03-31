"""DockerPlugin: ToolsPlugin implementation for Docker sandbox."""

from __future__ import annotations

import logging
from typing import Any

from axio.tool import Tool

from .config import SandboxConfig
from .handler import build_sandbox_exec, build_sandbox_read, build_sandbox_write
from .manager import SandboxManager

logger = logging.getLogger(__name__)


class DockerPlugin:
    """Dynamic tool provider for Docker sandbox execution.

    Discovered via the ``axio.tools.settings`` entry point group.
    The TUI interacts with this class through the ToolsPlugin protocol.
    """

    def __init__(self) -> None:
        self._manager = SandboxManager()
        self._tools: list[Tool] = []
        self._config: Any = None
        self._global_config: Any = None

    @property
    def label(self) -> str:
        return "Docker Sandbox"

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        self._config = config
        self._global_config = global_config
        await self._load_config()
        self._build_tools()

    async def _load_config(self) -> None:
        """Load sandbox config from DB (project first, then global)."""
        for db in (self._config, self._global_config):
            if db is None:
                continue
            raw = await db.get_prefix("docker.")
            if not raw:
                continue
            data: dict[str, str] = {}
            for full_key, value in raw.items():
                parts = full_key.split(".", 1)
                if len(parts) == 2:
                    data[parts[1]] = value
            if data:
                self._manager.config = SandboxConfig.from_dict(data)
                return

    def _build_tools(self) -> None:
        """Create the three sandbox tools."""
        self._tools = [
            Tool(
                name="sandbox_exec",
                description="Execute a shell command inside a Docker sandbox container",
                handler=build_sandbox_exec(self._manager),
            ),
            Tool(
                name="sandbox_write",
                description="Write content to a file inside the Docker sandbox container",
                handler=build_sandbox_write(self._manager),
            ),
            Tool(
                name="sandbox_read",
                description="Read a file from the Docker sandbox container",
                handler=build_sandbox_read(self._manager),
            ),
        ]

    @property
    def all_tools(self) -> list[Tool]:
        return self._tools

    def settings_screen(self) -> Any:
        from .settings import DockerSettingsScreen

        return DockerSettingsScreen(self._manager, config=self._config, global_config=self._global_config)

    async def close(self) -> None:
        await self._manager.close()
