"""MCPRegistry: manages MCP server instances with config DB persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from axio.tool import Tool

from .config import MCPServerConfig
from .handler import build_handler
from .session import MCPSession

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MCPRegistry:
    """Manages MCP server instances, discovers tools, persists to config DB."""

    _configs: dict[str, MCPServerConfig] = field(default_factory=dict)
    _sessions: dict[str, MCPSession] = field(default_factory=dict)
    _tools: dict[str, list[Tool]] = field(default_factory=dict)
    _errors: dict[str, str] = field(default_factory=dict)
    _config_db: Any = field(default=None)
    _global_config_db: Any = field(default=None)
    _server_scope: dict[str, Any] = field(default_factory=dict)

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        """Load saved MCP server configs from both DBs, connect, discover tools."""
        self._config_db = config
        self._global_config_db = global_config

        for scope_label, db in [("global", global_config), ("project", config)]:
            if db is None:
                continue
            raw = await db.get_prefix("mcp.")
            servers: dict[str, dict[str, str]] = {}
            for full_key, value in raw.items():
                parts = full_key.split(".", 2)
                if len(parts) != 3:
                    continue
                _, server_name, key = parts
                servers.setdefault(server_name, {})[key] = value

            for name, data in servers.items():
                if name in self._configs:
                    continue  # project cannot shadow global
                try:
                    cfg = MCPServerConfig.from_dict(name, data)
                except Exception:
                    logger.warning("Invalid saved MCP config %r, skipping", name, exc_info=True)
                    continue
                self._configs[name] = cfg
                self._server_scope[name] = db
                await self._connect_server(cfg)

    async def _connect_server(self, config: MCPServerConfig) -> list[Tool]:
        """Connect to a single server and discover its tools."""
        session = MCPSession(config)
        tools: list[Tool] = []
        try:
            await session.connect()
            mcp_tools = await session.list_tools()
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
                tools.append(Tool(name=tool_name, description=description, handler=handler))
            self._sessions[config.name] = session
            self._tools[config.name] = tools
            self._errors.pop(config.name, None)
            logger.info("Connected to MCP server %r (%d tools)", config.name, len(tools))
        except Exception as exc:
            logger.error("Failed to connect to MCP server %r", config.name, exc_info=True)
            self._errors[config.name] = str(exc)
            self._tools[config.name] = []
            try:
                await session.close()
            except Exception:
                pass
        return tools

    def _resolve_scope(self, name: str, scope: Any = None) -> Any:
        """Return the config DB for a server, falling back to global then project."""
        if scope is not None:
            return scope
        return self._server_scope.get(name) or self._global_config_db or self._config_db

    async def _persist_config(self, config: MCPServerConfig, scope: Any = None) -> None:
        """Save a server config to the config DB."""
        db = self._resolve_scope(config.name, scope)
        if db is None:
            return
        prefix = f"mcp.{config.name}."
        await db.delete_prefix(prefix)
        for key, value in config.to_dict().items():
            await db.set(f"{prefix}{key}", value)
        self._server_scope[config.name] = db

    async def _delete_config(self, name: str) -> None:
        """Remove a server config from the config DB."""
        db = self._resolve_scope(name)
        if db is None:
            return
        await db.delete_prefix(f"mcp.{name}.")
        self._server_scope.pop(name, None)

    async def add_server(self, config: MCPServerConfig, scope: Any = None) -> list[Tool]:
        """Add new server, connect, load tools, persist config."""
        if config.name in self._configs:
            raise ValueError(f"Server {config.name!r} already exists")
        self._configs[config.name] = config
        tools = await self._connect_server(config)
        await self._persist_config(config, scope=scope)
        return tools

    async def remove_server(self, name: str) -> None:
        """Disconnect and remove server, delete from config DB."""
        if name not in self._configs:
            raise KeyError(f"Server {name!r} not found")
        session = self._sessions.pop(name, None)
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass
        self._configs.pop(name, None)
        self._tools.pop(name, None)
        self._errors.pop(name, None)
        await self._delete_config(name)

    async def update_server(self, name: str, config: MCPServerConfig, scope: Any = None) -> list[Tool]:
        """Reconnect server with new config."""
        effective_scope = scope if scope is not None else self._server_scope.get(name)
        old_scope = self._server_scope.get(name)
        old_config = self._configs.get(name)

        # Only reconnect when connection parameters actually changed
        needs_reconnect = old_config is None or name != config.name or old_config.to_dict() != config.to_dict()

        if needs_reconnect:
            session = self._sessions.pop(name, None)
            if session is not None:
                try:
                    await session.close()
                except Exception:
                    pass
            self._tools.pop(name, None)
            self._errors.pop(name, None)

        if name != config.name:
            self._configs.pop(name, None)
            await self._delete_config(name)
        elif effective_scope is not old_scope and old_scope is not None:
            # Scope changed but name didn't — clean up old scope DB
            await old_scope.delete_prefix(f"mcp.{name}.")

        self._configs[config.name] = config

        if needs_reconnect:
            tools = await self._connect_server(config)
        else:
            tools = self._tools.get(config.name, [])

        await self._persist_config(config, scope=effective_scope)
        return tools

    @property
    def all_tools(self) -> list[Tool]:
        """Flat list of all MCP tools across all servers."""
        result: list[Tool] = []
        for tools in self._tools.values():
            result.extend(tools)
        return result

    @property
    def server_names(self) -> list[str]:
        """Names of all configured servers."""
        return list(self._configs)

    def server_status(self, name: str) -> str:
        """Return 'connected', 'error', or 'disconnected'."""
        if name in self._errors:
            return "error"
        if name in self._sessions:
            return "connected"
        return "disconnected"

    def server_tool_count(self, name: str) -> int:
        """Return the number of tools for a given server."""
        return len(self._tools.get(name, []))

    def server_config(self, name: str) -> MCPServerConfig:
        """Return the config for a given server."""
        return self._configs[name]

    def get_server_scope(self, name: str) -> Any:
        """Return the config DB instance a server is persisted to."""
        return self._server_scope.get(name)

    async def close(self) -> None:
        """Close all sessions."""
        for session in self._sessions.values():
            try:
                await session.close()
            except Exception:
                pass
        self._sessions.clear()
