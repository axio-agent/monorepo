"""Plugin discovery via entry points."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from axio.permission import PermissionGuard
from axio.tool import Tool, ToolHandler

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolsPlugin(Protocol):
    """Protocol for dynamic tool provider plugins.

    Plugins register via the ``axio.tools.settings`` entry point group.
    The TUI discovers them, calls ``init()``, collects tools, and shows
    settings screens — without knowing anything about the plugin internals.
    """

    @property
    def label(self) -> str: ...

    async def init(self, config: Any = None, global_config: Any = None) -> None: ...

    @property
    def all_tools(self) -> list[Tool]: ...

    def settings_screen(self) -> Any: ...

    async def close(self) -> None: ...


TOOLS_GROUP = "axio.tools"
TRANSPORT_GROUP = "axio.transport"
TRANSPORT_SETTINGS_GROUP = "axio.transport.settings"
GUARDS_GROUP = "axio.guards"
TOOLS_SETTINGS_GROUP = "axio.tools.settings"


def _handler_description(handler: type[ToolHandler]) -> str:
    """Extract handler docstring as tool description."""
    doc = handler.__doc__
    if doc:
        return doc.strip()
    return handler.__name__


def discover_tools() -> list[Tool]:
    """Load ToolHandler classes from 'axio.tools' entry points, build Tool objects."""
    tools: list[Tool] = []
    for ep in entry_points(group=TOOLS_GROUP):
        try:
            handler = ep.load()
        except Exception:
            logger.warning("Failed to load tool entry point %r", ep.name, exc_info=True)
            continue
        if not (isinstance(handler, type) and issubclass(handler, ToolHandler)):
            logger.warning("Entry point %r is not a ToolHandler subclass, skipping", ep.name)
            continue
        concurrency: int | None = getattr(handler, "_tool_concurrency", None)
        tools.append(
            Tool(
                name=ep.name,
                description=_handler_description(handler),
                handler=handler,
                concurrency=concurrency,
            )
        )
    return tools


def discover_transports() -> dict[str, type]:
    """Load transport classes from 'axio.transport' entry points."""
    transports: dict[str, type] = {}
    for ep in entry_points(group=TRANSPORT_GROUP):
        try:
            cls = ep.load()
        except Exception:
            logger.warning("Failed to load transport entry point %r", ep.name, exc_info=True)
            continue
        transports[ep.name] = cls
    return transports


def discover_transport_settings() -> dict[str, type]:
    """Load settings screens from 'axio.transport.settings' entry points."""
    screens: dict[str, type] = {}
    for ep in entry_points(group=TRANSPORT_SETTINGS_GROUP):
        try:
            cls = ep.load()
        except Exception:
            logger.debug("Settings screen %r unavailable (likely missing textual)", ep.name)
            continue
        screens[ep.name] = cls
    return screens


def discover_tools_plugins() -> dict[str, ToolsPlugin]:
    """Load and instantiate tool plugins from 'axio.tools.settings' entry points."""
    plugins: dict[str, ToolsPlugin] = {}
    for ep in entry_points(group=TOOLS_SETTINGS_GROUP):
        try:
            cls = ep.load()
            plugin: ToolsPlugin = cls()
            plugins[ep.name] = plugin
        except Exception:
            logger.debug("Tools plugin %r unavailable", ep.name)
            continue
    return plugins


def discover_guards() -> dict[str, type[PermissionGuard]]:
    """Load guard classes from 'axio.guards' entry points."""
    guards: dict[str, type[PermissionGuard]] = {}
    for ep in entry_points(group=GUARDS_GROUP):
        try:
            cls = ep.load()
        except Exception:
            logger.warning("Failed to load guard entry point %r", ep.name, exc_info=True)
            continue
        if not (isinstance(cls, type) and issubclass(cls, PermissionGuard)):
            logger.warning("Entry point %r is not a PermissionGuard subclass, skipping", ep.name)
            continue
        guards[ep.name] = cls
    return guards
