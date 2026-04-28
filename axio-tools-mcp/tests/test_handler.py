"""Tests for build_handler() dynamic function creation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from axio_tools_mcp.handler import build_handler
from axio_tools_mcp.session import MCPSession


def _make_mock_session() -> MCPSession:
    session = MagicMock(spec=MCPSession)
    session.call_tool = AsyncMock()
    return session


def test_schema_fidelity() -> None:
    """Built handler has correct fields from JSON schema."""
    session = _make_mock_session()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "count": {"type": "integer", "description": "Number of items"},
            "verbose": {"type": "boolean"},
        },
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read a file", schema, session)
    from axio.schema import build_tool_schema

    json_schema = build_tool_schema(handler)

    assert "path" in json_schema["properties"]
    assert "count" in json_schema["properties"]
    assert "verbose" in json_schema["properties"]


async def test_call_forwarding() -> None:
    """Handler forwards to MCP session.call_tool."""
    session = _make_mock_session()
    mock_call = cast(AsyncMock, session.call_tool)
    mock_call.return_value = CallToolResult(
        content=[TextContent(type="text", text="hello world")],
        isError=False,
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }
    handler = build_handler("echo__say", "say", "Say something", schema, session)
    result = await handler(message="hi")

    assert result == "hello world"
    mock_call.assert_awaited_once_with("say", {"message": "hi"})


async def test_error_handling() -> None:
    """Handler raises RuntimeError when isError=True."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="not found")],
        isError=True,
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    handler = build_handler("fs__read", "read", "Read file", schema, session)

    with pytest.raises(RuntimeError, match="not found"):
        await handler(path="/missing")


async def test_empty_schema() -> None:
    """Handler works with empty input schema (no params)."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="done")],
        isError=False,
    )

    handler = build_handler("sys__status", "status", "Get status", {}, session)
    result = await handler()
    assert result == "done"


def test_type_mapping() -> None:
    """All JSON schema types are mapped to Python types."""
    from axio.schema import build_tool_schema

    session = _make_mock_session()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "a": {"type": "array"},
            "o": {"type": "object"},
        },
        "required": ["s", "i", "n", "b", "a", "o"],
    }
    handler = build_handler("test__types", "types", "Type test", schema, session)
    json_schema = build_tool_schema(handler)
    props: dict[str, Any] = json_schema["properties"]
    assert props["s"] == {"type": "string"}
    assert props["i"] == {"type": "integer"}
    assert props["n"] == {"type": "number"}
    assert props["b"] == {"type": "boolean"}


def test_mcp_schema_passed_through_to_tool() -> None:
    """Tool.input_schema must be the original MCP schema, not a re-derived one.

    Re-deriving from handler annotations incorrectly marks optional params as
    required (no FieldInfo default -> added to required list by build_tool_schema).
    Passing input_schema directly via Tool(schema=...) preserves the MCP contract.
    """
    from types import MappingProxyType

    from axio.tool import Tool

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
    handler = build_handler("test__defaults", "defaults", "Default test", mcp_schema, session)
    tool: Tool[Any] = Tool(
        name="test__defaults",
        description="Default test",
        handler=handler,
        schema=MappingProxyType(mcp_schema),
    )
    props: dict[str, Any] = tool.input_schema["properties"]
    required: list[str] = tool.input_schema.get("required", [])

    # required list must match MCP schema exactly
    assert required == ["required_field"]
    assert "optional_no_default" not in required
    assert "optional_with_default" not in required

    # optional param with no default must not have 'default' key
    assert "default" not in props["optional_no_default"]
    # optional param with explicit default must surface it
    assert props["optional_with_default"].get("default") == "hello"
