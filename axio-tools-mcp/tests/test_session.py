"""Tests for MCPSession."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.session import MCPSession


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    return MCPServerConfig(name="test", command="echo", args=["hello"])


@pytest.fixture
def http_config() -> MCPServerConfig:
    return MCPServerConfig(name="remote", url="http://localhost:8000/mcp")


def _mock_client_session() -> MagicMock:
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=ListToolsResult(
            tools=[
                MCPTool(name="add", description="Add numbers", inputSchema={"type": "object"}),
            ],
        ),
    )
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="3")],
            isError=False,
        ),
    )
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


async def test_connect_stdio(stdio_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        mock_read, mock_write = MagicMock(), MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()

        assert session.is_connected
        mock_session.initialize.assert_awaited_once()
        await session.close()
        assert not session.is_connected


async def test_connect_http(http_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.streamable_http_client") as mock_http,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        mock_read, mock_write = MagicMock(), MagicMock()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, lambda: "sid"))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_http.return_value = ctx

        session = MCPSession(http_config)
        await session.connect()

        assert session.is_connected
        mock_session.initialize.assert_awaited_once()
        await session.close()


async def test_list_tools(stdio_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()
        tools = await session.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "add"
        await session.close()


async def test_call_tool(stdio_config: MCPServerConfig) -> None:
    mock_session = _mock_client_session()

    with (
        patch("axio_tools_mcp.session.stdio_client") as mock_stdio,
        patch("axio_tools_mcp.session.ClientSession", return_value=mock_session),
    ):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = ctx

        session = MCPSession(stdio_config)
        await session.connect()
        result = await session.call_tool("add", {"a": 1, "b": 2})

        assert not result.isError
        assert result.content[0].text == "3"  # type: ignore[union-attr]
        await session.close()


async def test_call_tool_not_connected(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.call_tool("add", {})


async def test_list_tools_not_connected(stdio_config: MCPServerConfig) -> None:
    session = MCPSession(stdio_config)
    with pytest.raises(RuntimeError, match="Not connected"):
        await session.list_tools()
