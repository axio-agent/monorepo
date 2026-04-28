"""MCP tool handler factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mcp.types import TextContent

from .session import MCPSession


def build_handler(
    tool_name: str,
    mcp_tool_name: str,
    description: str,
    session: MCPSession,
) -> Callable[..., Awaitable[str]]:
    """Return a plain async handler that forwards calls to the MCP session.

    The schema is provided separately via ``Tool(schema=MappingProxyType(input_schema))``;
    no annotation injection is needed here.
    """

    async def handler(**kwargs: Any) -> str:
        result = await session.call_tool(mcp_tool_name, kwargs)
        if result.isError:
            parts = [c.text for c in result.content if isinstance(c, TextContent)]
            raise RuntimeError("\n".join(parts) or "MCP tool error")
        parts = [c.text for c in result.content if isinstance(c, TextContent)]
        return "\n".join(parts) or ""

    handler.__doc__ = description
    handler.__name__ = tool_name
    handler.__annotations__ = {"return": str}
    return handler
