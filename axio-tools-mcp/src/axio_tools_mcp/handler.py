"""Dynamic ToolHandler creation from MCP tool schemas."""

from __future__ import annotations

from typing import Any, ClassVar

import pydantic
from axio.tool import ToolHandler
from mcp.types import TextContent

from .session import MCPSession

_JSON_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_fields(
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Convert JSON Schema properties to Pydantic field definitions."""
    properties: dict[str, Any] = input_schema.get("properties", {})
    required: set[str] = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(json_type, str)
        description = prop_schema.get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, pydantic.Field(description=description))
        else:
            default = prop_schema.get("default")
            fields[prop_name] = (
                py_type | None,
                pydantic.Field(default=default, description=description),
            )

    return fields


def build_handler(
    tool_name: str,
    mcp_tool_name: str,
    description: str,
    input_schema: dict[str, Any],
    session: MCPSession,
) -> type[ToolHandler]:
    """Create a dynamic ToolHandler subclass for an MCP tool.

    The returned class forwards calls to the MCP session.
    """
    fields = _build_fields(input_schema)
    class_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Handler"

    async def __call__(self: Any) -> str:
        mcp_session: MCPSession = self.__class__._mcp_session
        name: str = self.__class__._mcp_tool_name
        data = self.model_dump(exclude_none=True)
        result = await mcp_session.call_tool(name, data)
        if result.isError:
            parts = [c.text for c in result.content if isinstance(c, TextContent)]
            raise RuntimeError("\n".join(parts) or "MCP tool error")
        parts = [c.text for c in result.content if isinstance(c, TextContent)]
        return "\n".join(parts) or ""

    handler_cls: type[ToolHandler] = pydantic.create_model(
        class_name,
        __base__=ToolHandler,
        __doc__=description,
        **fields,
    )

    handler_cls.__call__ = __call__  # type: ignore[method-assign]
    handler_cls._mcp_session = session  # type: ignore[attr-defined]
    handler_cls._mcp_tool_name = mcp_tool_name  # type: ignore[attr-defined]

    # Add ClassVar annotations so mypy knows about them
    handler_cls.__annotations__["_mcp_session"] = ClassVar[MCPSession]
    handler_cls.__annotations__["_mcp_tool_name"] = ClassVar[str]

    return handler_cls
