"""MCPPlugin: ToolsPlugin implementation for the TUI plugin system."""

from __future__ import annotations

from typing import Any

from axio.tool import Tool

from .registry import MCPRegistry


class MCPPlugin:
    """Dynamic tool provider that bridges MCP servers into Axio.

    Discovered via the ``axio.tools.settings`` entry point group.
    The TUI interacts with this class through the ToolsPlugin protocol.
    """

    def __init__(self) -> None:
        self._registry = MCPRegistry()
        self._config: Any = None
        self._global_config: Any = None

    @property
    def label(self) -> str:
        return "MCP Servers"

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        self._config = config
        self._global_config = global_config
        await self._registry.init(config=config, global_config=global_config)

    @property
    def all_tools(self) -> list[Tool]:
        return self._registry.all_tools

    def settings_screen(self) -> Any:
        from .settings import MCPHubScreen

        return MCPHubScreen(self._registry, config=self._config, global_config=self._global_config)

    async def close(self) -> None:
        await self._registry.close()
