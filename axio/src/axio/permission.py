"""Permission system: guards that gate tool execution."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .exceptions import GuardError

if TYPE_CHECKING:
    from .tool import Tool


class PermissionGuard(ABC):
    """Gate for tool calls. Return modified kwargs to allow, raise to deny.

    Tool invokes guards via ``await guard(tool, **kwargs)``.

    Guards receive the ``Tool`` object and the raw keyword arguments before
    execution.  Return the (possibly modified) dict to allow; raise
    ``GuardError`` to deny.  Because guards see all inputs up front, they are
    also the right place for **logging and auditing**::

        class AuditGuard(PermissionGuard):
            async def check(self, tool: Tool, **kwargs: Any) -> dict[str, Any]:
                logger.info("tool=%s args=%s", tool.name, kwargs)
                return kwargs  # always allow

    See ``examples/agent_swarm/agent_swarm/__main__.py`` (``RoleGuard``) for a
    production example.
    """

    async def __call__(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        return await self.check(tool, **kwargs)

    @abstractmethod
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]: ...


class ConcurrentGuard(PermissionGuard, ABC):
    """Guard with concurrency control.

    Subclass and override ``check()``.  ``__call__`` acquires the semaphore
    then delegates to ``check()``.  Set ``concurrency`` to control parallelism
    (default 1 - one check at a time).
    """

    concurrency: int = 1

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def __call__(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        async with self._semaphore:
            return await self.check(tool, **kwargs)


class AllowAllGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)


class DenyAllGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        raise GuardError("denied")
