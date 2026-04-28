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
