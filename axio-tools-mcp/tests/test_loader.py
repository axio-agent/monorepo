"""Tests for load_mcp_tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.loader import load_mcp_tools


def _make_mock_session_cls(tools: list[MCPTool], fail: bool = False) -> type:
    """Create a mock MCPSession class."""

    class MockSession:
        def __init__(self, config: MCPServerConfig) -> None:
            self.config = config
            self.is_connected = False

        async def connect(self) -> None:
            if fail:
                raise ConnectionError("Failed to connect")
            self.is_connected = True

        async def list_tools(self) -> list[MCPTool]:
            return tools

        async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            from mcp.types import CallToolResult

            return CallToolResult(
                content=[TextContent(type="text", text="ok")],
                isError=False,
            )

        async def close(self) -> None:
            self.is_connected = False

    return MockSession


async def test_single_server() -> None:
    tools = [
        MCPTool(name="read", description="Read file", inputSchema={"type": "object", "properties": {}}),
        MCPTool(name="write", description="Write file", inputSchema={"type": "object", "properties": {}}),
    ]
    configs = [MCPServerConfig(name="fs", command="python")]

    with patch("axio_tools_mcp.loader.MCPSession") as mock_cls:
        mock_session = MagicMock()
        mock_session.connect = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=tools)
        mock_session.close = AsyncMock()
        mock_cls.return_value = mock_session

        result_tools, sessions = await load_mcp_tools(configs)

    assert len(result_tools) == 2
    assert result_tools[0].name == "fs__read"
    assert result_tools[1].name == "fs__write"
    assert len(sessions) == 1


async def test_namespacing() -> None:
    tools_a = [MCPTool(name="list", description="List", inputSchema={"type": "object", "properties": {}})]
    tools_b = [MCPTool(name="list", description="List", inputSchema={"type": "object", "properties": {}})]
    configs = [
        MCPServerConfig(name="server_a", command="a"),
        MCPServerConfig(name="server_b", command="b"),
    ]

    with patch("axio_tools_mcp.loader.MCPSession") as mock_cls:
        call_count = 0

        def create_session(config: MCPServerConfig) -> MagicMock:
            nonlocal call_count
            session = MagicMock()
            session.connect = AsyncMock()
            session.list_tools = AsyncMock(return_value=tools_a if call_count == 0 else tools_b)
            session.close = AsyncMock()
            call_count += 1
            return session

        mock_cls.side_effect = create_session
        result_tools, sessions = await load_mcp_tools(configs)

    assert len(result_tools) == 2
    names = {t.name for t in result_tools}
    assert "server_a__list" in names
    assert "server_b__list" in names


async def test_failed_server_graceful() -> None:
    """Failed server is skipped, other servers still load."""
    good_tools = [MCPTool(name="ping", description="Ping", inputSchema={"type": "object", "properties": {}})]
    configs = [
        MCPServerConfig(name="bad", command="fail"),
        MCPServerConfig(name="good", command="ok"),
    ]

    with patch("axio_tools_mcp.loader.MCPSession") as mock_cls:
        call_count = 0

        def create_session(config: MCPServerConfig) -> MagicMock:
            nonlocal call_count
            session = MagicMock()
            if call_count == 0:
                session.connect = AsyncMock(side_effect=ConnectionError("refused"))
            else:
                session.connect = AsyncMock()
                session.list_tools = AsyncMock(return_value=good_tools)
            session.close = AsyncMock()
            call_count += 1
            return session

        mock_cls.side_effect = create_session
        result_tools, sessions = await load_mcp_tools(configs)

    assert len(result_tools) == 1
    assert result_tools[0].name == "good__ping"
    assert len(sessions) == 1


async def test_empty_servers() -> None:
    result_tools, sessions = await load_mcp_tools([])
    assert result_tools == []
    assert sessions == []
