"""Tests for build_handler() dynamic ToolHandler creation."""

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
    """Built handler class has correct fields from JSON schema."""
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
    handler_cls = build_handler("fs__read", "read", "Read a file", schema, session)
    json_schema = handler_cls.model_json_schema()

    assert "path" in json_schema["properties"]
    assert "count" in json_schema["properties"]
    assert "verbose" in json_schema["properties"]


def test_required_vs_optional() -> None:
    """Required fields must be provided, optional fields have defaults."""
    session = _make_mock_session()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "default": 25},
        },
        "required": ["name"],
    }
    handler_cls = build_handler("test__greet", "greet", "Greet", schema, session)

    instance = cast(type[Any], handler_cls)(name="Alice")
    assert instance.name == "Alice"
    assert instance.age is None or instance.age == 25


async def test_call_forwarding() -> None:
    """Handler __call__ forwards to MCP session.call_tool."""
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
    handler_cls = build_handler("echo__say", "say", "Say something", schema, session)
    instance = cast(type[Any], handler_cls)(message="hi")
    result = await instance()

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
    handler_cls = build_handler("fs__read", "read", "Read file", schema, session)
    instance = cast(type[Any], handler_cls)(path="/missing")

    with pytest.raises(RuntimeError, match="not found"):
        await instance()


async def test_empty_schema() -> None:
    """Handler works with empty input schema (no params)."""
    session = _make_mock_session()
    cast(AsyncMock, session.call_tool).return_value = CallToolResult(
        content=[TextContent(type="text", text="done")],
        isError=False,
    )

    handler_cls = build_handler("sys__status", "status", "Get status", {}, session)
    instance = handler_cls()
    result = await instance()
    assert result == "done"


def test_type_mapping() -> None:
    """All JSON schema types are mapped to Python types."""
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
    handler_cls = build_handler("test__types", "types", "Type test", schema, session)
    instance = cast(type[Any], handler_cls)(s="x", i=1, n=1.5, b=True, a=[1, 2], o={"k": "v"})
    assert instance.s == "x"
    assert instance.i == 1
    assert instance.n == 1.5
    assert instance.b is True
    assert instance.a == [1, 2]
    assert instance.o == {"k": "v"}
