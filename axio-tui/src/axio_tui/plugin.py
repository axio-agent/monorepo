"""Plugin discovery via entry points."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from axio.permission import PermissionGuard
from axio.tool import Tool

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolsPlugin(Protocol):
    """Protocol for dynamic tool provider plugins.

    Plugins register via the ``axio.tools.settings`` entry point group.
    The TUI discovers them, calls ``init()``, collects tools, and shows
    settings screens - without knowing anything about the plugin internals.
    """

    @property
    def label(self) -> str: ...

    async def init(self, config: Any = None, global_config: Any = None) -> None: ...

    @property
    def all_tools(self) -> list[Tool[Any]]: ...

    def settings_screen(self) -> Any: ...

    async def close(self) -> None: ...


TOOLS_GROUP = "axio.tools"
TRANSPORT_GROUP = "axio.transport"
TRANSPORT_SETTINGS_GROUP = "axio.transport.settings"
GUARDS_GROUP = "axio.guards"
TOOLS_SETTINGS_GROUP = "axio.tools.settings"
SELECTOR_GROUP = "axio.selector"


def _make_tool(ep_name: str, handler: Callable[..., Any]) -> Tool[Any]:
    """Build a Tool from an entry-point handler (plain async function or callable)."""
    concurrency: int | None = getattr(handler, "_tool_concurrency", None)
    return Tool(name=ep_name, handler=handler, concurrency=concurrency)


def discover_tools() -> list[Tool[Any]]:
    """Load handler callables from 'axio.tools' entry points, build Tool objects."""
    tools: list[Tool[Any]] = []
    for ep in entry_points(group=TOOLS_GROUP):
        try:
            handler = ep.load()
        except Exception:
            logger.warning("Failed to load tool entry point %r", ep.name, exc_info=True)
            continue
        if not callable(handler):
            logger.warning("Entry point %r is not callable, skipping", ep.name)
            continue
        if not inspect.iscoroutinefunction(handler):
            logger.warning("Entry point %r is not an async callable, skipping", ep.name)
            continue
        tools.append(_make_tool(ep.name, handler))
    return tools


def discover_tools_by_package() -> dict[str, list[Tool[Any]]]:
    """Return tools from 'axio.tools' entry points grouped by distribution package name."""
    groups: dict[str, list[Tool[Any]]] = {}
    for ep in entry_points(group=TOOLS_GROUP):
        try:
            handler = ep.load()
        except Exception:
            logger.warning("Failed to load tool entry point %r", ep.name, exc_info=True)
            continue
        if not callable(handler) or not inspect.iscoroutinefunction(handler):
            continue
        pkg = ep.dist.name if ep.dist else "unknown"
        groups.setdefault(pkg, []).append(_make_tool(ep.name, handler))
    return groups


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


def discover_selectors() -> dict[str, type]:
    """Return {ep_name: cls} from 'axio.selector' entry points."""
    result: dict[str, type] = {}
    for ep in entry_points(group=SELECTOR_GROUP):
        try:
            result[ep.name] = ep.load()
        except Exception:
            logger.warning("Failed to load selector EP %r", ep.name, exc_info=True)
    return result


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
