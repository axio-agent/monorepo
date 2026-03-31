"""Tests for MCPRegistry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

from axio_tools_mcp.config import MCPServerConfig
from axio_tools_mcp.registry import MCPRegistry


def _mock_session(tools: list[MCPTool] | None = None) -> MagicMock:
    if tools is None:
        tools = [MCPTool(name="test", description="Test tool", inputSchema={"type": "object", "properties": {}})]
    session = MagicMock()
    session.connect = AsyncMock()
    session.list_tools = AsyncMock(return_value=tools)
    session.call_tool = AsyncMock(
        return_value=CallToolResult(content=[TextContent(type="text", text="ok")], isError=False),
    )
    session.close = AsyncMock()
    session.is_connected = True
    return session


async def test_add_server() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="myserver", command="python")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        tools = await registry.add_server(config)

    assert len(tools) == 1
    assert tools[0].name == "myserver__test"
    assert "myserver" in registry.server_names
    assert registry.server_status("myserver") == "connected"
    assert registry.server_tool_count("myserver") == 1


async def test_remove_server() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="myserver", command="python")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.add_server(config)
        await registry.remove_server("myserver")

    assert "myserver" not in registry.server_names
    assert registry.all_tools == []


async def test_update_server() -> None:
    registry = MCPRegistry()
    config1 = MCPServerConfig(name="myserver", command="python")
    config2 = MCPServerConfig(name="myserver", command="node")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.add_server(config1)

        new_tools = [
            MCPTool(name="new_tool", description="New", inputSchema={"type": "object", "properties": {}}),
        ]
        mock_cls.return_value = _mock_session(new_tools)
        tools = await registry.update_server("myserver", config2)

    assert len(tools) == 1
    assert tools[0].name == "myserver__new_tool"


async def test_all_tools_aggregation() -> None:
    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        tools_a = [MCPTool(name="a", description="A", inputSchema={"type": "object", "properties": {}})]
        mock_cls.return_value = _mock_session(tools_a)
        await registry.add_server(MCPServerConfig(name="server_a", command="a"))

        tools_b = [MCPTool(name="b", description="B", inputSchema={"type": "object", "properties": {}})]
        mock_cls.return_value = _mock_session(tools_b)
        await registry.add_server(MCPServerConfig(name="server_b", command="b"))

    all_tools = registry.all_tools
    assert len(all_tools) == 2
    names = {t.name for t in all_tools}
    assert "server_a__a" in names
    assert "server_b__b" in names


async def test_server_status_error() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="bad", command="fail")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        session = _mock_session()
        session.connect = AsyncMock(side_effect=ConnectionError("refused"))
        mock_cls.return_value = session
        await registry.add_server(config)

    assert registry.server_status("bad") == "error"
    assert registry.server_tool_count("bad") == 0


async def test_duplicate_server_raises() -> None:
    registry = MCPRegistry()
    config = MCPServerConfig(name="myserver", command="python")

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.add_server(config)

    import pytest

    with pytest.raises(ValueError, match="already exists"):
        await registry.add_server(config)


async def test_close_all() -> None:
    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        session = _mock_session()
        mock_cls.return_value = session
        await registry.add_server(MCPServerConfig(name="s1", command="a"))
        await registry.close()

    session.close.assert_awaited()


async def test_init_from_config_db() -> None:
    """Registry loads saved configs from config DB on init."""
    mock_db = MagicMock()
    mock_db.get_prefix = AsyncMock(
        return_value={
            "mcp.myserver.command": "python",
            "mcp.myserver.args": '["-m", "server"]',
        }
    )
    mock_db.delete_prefix = AsyncMock()
    mock_db.set = AsyncMock()

    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.init(mock_db)

    assert "myserver" in registry.server_names
    mock_db.get_prefix.assert_awaited_once_with("mcp.")


async def test_dual_config_init() -> None:
    """Registry loads servers from both global and project config DBs."""
    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(
        return_value={
            "mcp.global_server.command": "python",
            "mcp.global_server.args": '["-m", "global_mod"]',
        }
    )
    global_db.delete_prefix = AsyncMock()
    global_db.set = AsyncMock()

    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(
        return_value={
            "mcp.project_server.command": "node",
        }
    )
    project_db.delete_prefix = AsyncMock()
    project_db.set = AsyncMock()

    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.init(config=project_db, global_config=global_db)

    assert "global_server" in registry.server_names
    assert "project_server" in registry.server_names

    # Verify scope tracking
    assert registry.get_server_scope("global_server") is global_db
    assert registry.get_server_scope("project_server") is project_db


async def test_dual_config_persist_to_correct_scope() -> None:
    """Adding a server with explicit scope persists to the correct DB."""
    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(return_value={})
    global_db.delete_prefix = AsyncMock()
    global_db.set = AsyncMock()

    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(return_value={})
    project_db.delete_prefix = AsyncMock()
    project_db.set = AsyncMock()

    registry = MCPRegistry()
    await registry.init(config=project_db, global_config=global_db)

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.add_server(MCPServerConfig(name="proj_srv", command="python"), scope=project_db)

    # Should persist to project DB, not global
    proj_set_calls = {call.args[0] for call in project_db.set.call_args_list}
    assert "mcp.proj_srv.command" in proj_set_calls
    assert not any("proj_srv" in str(call) for call in global_db.set.call_args_list)
    assert registry.get_server_scope("proj_srv") is project_db


async def test_scope_only_change_skips_reconnect() -> None:
    """Changing scope without modifying config should not reconnect."""
    global_db = MagicMock()
    global_db.get_prefix = AsyncMock(
        return_value={
            "mcp.srv.command": "python",
        }
    )
    global_db.delete_prefix = AsyncMock()
    global_db.set = AsyncMock()

    project_db = MagicMock()
    project_db.get_prefix = AsyncMock(return_value={})
    project_db.delete_prefix = AsyncMock()
    project_db.set = AsyncMock()

    registry = MCPRegistry()

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_session = _mock_session()
        mock_cls.return_value = mock_session
        await registry.init(config=project_db, global_config=global_db)

        # Reset to track calls during update
        mock_cls.reset_mock()
        mock_session.close.reset_mock()

        # Change scope only (same config)
        same_config = MCPServerConfig(name="srv", command="python")
        await registry.update_server("srv", same_config, scope=project_db)

    # Should NOT have created a new session or closed the old one
    mock_cls.assert_not_called()
    mock_session.close.assert_not_awaited()

    # Should be persisted to project DB now
    assert registry.get_server_scope("srv") is project_db
    proj_set_calls = {call.args[0] for call in project_db.set.call_args_list}
    assert "mcp.srv.command" in proj_set_calls

    # Old scope (global) should have been cleaned up
    global_db.delete_prefix.assert_awaited_with("mcp.srv.")


async def test_persistence_roundtrip() -> None:
    """Add server persists config, remove deletes it."""
    mock_db = MagicMock()
    mock_db.get_prefix = AsyncMock(return_value={})
    mock_db.delete_prefix = AsyncMock()
    mock_db.set = AsyncMock()

    registry = MCPRegistry()
    await registry.init(mock_db)

    with patch("axio_tools_mcp.registry.MCPSession") as mock_cls:
        mock_cls.return_value = _mock_session()
        await registry.add_server(MCPServerConfig(name="persisted", command="python"))

    # Check that set was called with mcp.persisted.command
    set_calls = {call.args[0]: call.args[1] for call in mock_db.set.call_args_list}
    assert "mcp.persisted.command" in set_calls
    assert set_calls["mcp.persisted.command"] == "python"

    # Remove should delete prefix
    await registry.remove_server("persisted")
    mock_db.delete_prefix.assert_awaited_with("mcp.persisted.")
