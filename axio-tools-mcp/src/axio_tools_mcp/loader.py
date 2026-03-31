"""Load MCP tools from server configurations."""

from __future__ import annotations

import logging

from axio.tool import Tool

from .config import MCPServerConfig
from .handler import build_handler
from .session import MCPSession

logger = logging.getLogger(__name__)


async def load_mcp_tools(
    servers: list[MCPServerConfig],
) -> tuple[list[Tool], list[MCPSession]]:
    """Connect to MCP servers and discover their tools.

    Returns ``(tools, sessions)``. The caller is responsible for closing sessions.
    Failed servers are logged and skipped.
    """
    all_tools: list[Tool] = []
    sessions: list[MCPSession] = []

    for config in servers:
        session = MCPSession(config)
        try:
            await session.connect()
            mcp_tools = await session.list_tools()
        except Exception:
            logger.error("Failed to connect to MCP server %r", config.name, exc_info=True)
            try:
                await session.close()
            except Exception:
                pass
            continue

        sessions.append(session)
        for mcp_tool in mcp_tools:
            tool_name = f"{config.name}__{mcp_tool.name}"
            description = mcp_tool.description or mcp_tool.name
            input_schema = mcp_tool.inputSchema if isinstance(mcp_tool.inputSchema, dict) else {}
            handler = build_handler(
                tool_name=tool_name,
                mcp_tool_name=mcp_tool.name,
                description=description,
                input_schema=input_schema,
                session=session,
            )
            all_tools.append(Tool(name=tool_name, description=description, handler=handler))
            logger.info("Loaded MCP tool %r from server %r", tool_name, config.name)

    return all_tools, sessions
