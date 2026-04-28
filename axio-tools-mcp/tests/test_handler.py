"""Tests for build_handler() - MCP session forwarding."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from axio.tool import Tool
from mcp.types import CallToolResult, TextContent

from axio_tools_mcp.handler import build_handler
from axio_tools_mcp.session import MCPSession


def _make_mock_session() -> MCPSession:
    session = MagicMock(spec=MCPSession)
    session.call_tool = AsyncMock()
    return session


async def test_call_forwarding() -> None:
    """Handler forwards kwargs to MCP session.call_tool."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="hello world")],
        isError=False,
    )

    handler = build_handler("echo__say", "say", "Say something", session)
    result = await handler(message="hi")

    assert result == "hello world"
    cast(AsyncMock, session.call_tool).assert_awaited_once_with("say", {"message": "hi"})


async def test_error_handling() -> None:
    """Handler raises RuntimeError when isError=True."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="not found")],
        isError=True,
    )

    handler = build_handler("fs__read", "read", "Read file", session)
    with pytest.raises(RuntimeError, match="not found"):
        await handler(path="/missing")


async def test_empty_result() -> None:
    """Handler returns empty string when MCP content is empty."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(content=[], isError=False)

    handler = build_handler("sys__status", "status", "Get status", session)
    assert await handler() == ""


async def test_multipart_content_joined() -> None:
    """Multiple TextContent parts are joined with newlines."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[
            TextContent(type="text", text="line1"),
            TextContent(type="text", text="line2"),
        ],
        isError=False,
    )

    handler = build_handler("t__t", "t", "t", session)
    assert await handler() == "line1\nline2"


def test_handler_metadata() -> None:
    """Handler has correct __name__ and __doc__."""
    session = _make_mock_session()
    handler = build_handler("my_server__my_tool", "my_tool", "Does stuff", session)
    assert handler.__name__ == "my_server__my_tool"
    assert handler.__doc__ == "Does stuff"


async def test_unknown_extras_filtered_by_schema() -> None:
    """Unknown kwargs are filtered to schema properties before reaching the MCP server.

    Tool.__call__ filters **kwargs handlers to declared schema properties before
    guards and before execution, so unknown extras are not forwarded.
    """
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        isError=False,
    )

    mcp_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read file", session)
    tool: Tool[Any] = Tool(
        name="fs__read",
        description="Read file",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
    )

    await tool(path="/tmp/file.txt", _unknown_extra="should-be-dropped")

    # Only the declared schema property must reach the MCP server.
    cast(AsyncMock, session.call_tool).assert_awaited_once_with("read", {"path": "/tmp/file.txt"})


def test_mcp_schema_passed_through_to_tool() -> None:
    """Tool.input_schema is the original MCP schema - not re-derived from annotations.

    The handler has no parameter annotations; the schema comes exclusively from
    Tool(schema=MappingProxyType(input_schema)).
    """
    session = _make_mock_session()
    mcp_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "required_field": {"type": "string"},
            "optional_no_default": {"type": "string"},
            "optional_with_default": {"type": "string", "default": "hello"},
        },
        "required": ["required_field"],
    }
    handler = build_handler("test__defaults", "defaults", "Default test", session)
    tool: Tool[Any] = Tool(
        name="test__defaults",
        description="Default test",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
    )

    required: list[str] = tool.input_schema.get("required", [])
    props: dict[str, Any] = tool.input_schema["properties"]

    assert required == ["required_field"]
    assert "optional_no_default" not in required
    assert "optional_with_default" not in required
    assert "default" not in props["optional_no_default"]
    assert props["optional_with_default"].get("default") == "hello"


async def test_schema_type_validated_at_runtime() -> None:
    """Tool enforces MCP schema types at call time - wrong type raises HandlerError."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        isError=False,
    )

    mcp_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    handler = build_handler("t__count", "count_tool", "Count", session)
    tool: Tool[Any] = Tool(
        name="t__count",
        description="Count",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
    )

    from axio.exceptions import HandlerError

    with pytest.raises(HandlerError, match="requires int"):
        await tool(count="not-an-int")

    # Correct type passes and reaches the server.
    cast(AsyncMock, session.call_tool).reset_mock()
    await tool(count=42)
    cast(AsyncMock, session.call_tool).assert_awaited_once_with("count_tool", {"count": 42})


async def test_schema_default_injected() -> None:
    """Schema property defaults are injected before the call reaches the MCP server."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        isError=False,
    )

    mcp_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "encoding": {"type": "string", "default": "utf-8"},
        },
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read", session)
    tool: Tool[Any] = Tool(
        name="fs__read",
        description="Read",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
    )

    await tool(path="/tmp/file.txt")
    cast(AsyncMock, session.call_tool).assert_awaited_once_with("read", {"path": "/tmp/file.txt", "encoding": "utf-8"})


async def test_empty_properties_strips_all_extras() -> None:
    """A schema with properties:{} accepts no kwargs - extras are silently dropped."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        isError=False,
    )

    mcp_schema: dict[str, Any] = {"type": "object", "properties": {}}
    handler = build_handler("sys__ping", "ping", "Ping", session)
    tool: Tool[Any] = Tool(
        name="sys__ping",
        description="Ping",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
    )

    await tool(_extra="ignored")
    cast(AsyncMock, session.call_tool).assert_awaited_once_with("ping", {})


async def test_guard_injected_extras_not_forwarded_to_mcp_server() -> None:
    """Guard-injected keys outside the schema are stripped before the MCP call.

    Tool.__call__ applies the schema-based post-guard strip for explicit-schema
    handlers, so guards cannot accidentally inject kwargs that reach the MCP server.
    """
    from axio.permission import PermissionGuard

    class _Inject(PermissionGuard):
        async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
            return {**kwargs, "_audit_tag": "injected-by-guard"}

    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        isError=False,
    )

    mcp_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read", session)
    tool: Tool[Any] = Tool(
        name="fs__read",
        description="Read",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
        guards=(_Inject(),),
    )

    await tool(path="/tmp/file.txt")
    cast(AsyncMock, session.call_tool).assert_awaited_once_with("read", {"path": "/tmp/file.txt"})
