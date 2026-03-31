"""MCPSession: wraps mcp.ClientSession with explicit lifecycle management."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPSession:
    """Manages connection lifecycle for a single MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def config(self) -> MCPServerConfig:
        return self._config

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        """Connect to the MCP server and initialize the session."""
        if self._session is not None:
            return

        if self._config.command is not None:
            params = StdioServerParameters(
                command=self._config.command,
                args=self._config.args,
                env=self._config.env,
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        elif self._config.url is not None:
            http_client = httpx.AsyncClient(
                headers=self._config.headers or None,
                timeout=self._config.timeout,
            )
            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(self._config.url, http_client=http_client),
            )
        else:
            raise ValueError("No command or url configured")

        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Connected to MCP server %r", self._config.name)

    async def list_tools(self) -> list[Tool]:
        """List available tools from the connected MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected")
        result = await self._session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, object]) -> CallToolResult:
        """Call a tool on the connected MCP server."""
        if self._session is None:
            raise RuntimeError("Not connected")
        return await self._session.call_tool(name, arguments=arguments)

    async def close(self) -> None:
        """Close the session and release resources."""
        self._session = None
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        logger.info("Disconnected from MCP server %r", self._config.name)
