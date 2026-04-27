"""Permission system: guards that gate tool execution."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from .exceptions import GuardError


class PermissionGuard(ABC):
    """Gate for tool calls. Return handler to allow, raise to deny.

    Tool calls guards via ``await guard(instance)``.

    Guards are not limited to access control.  Because ``check()`` receives the
    fully-parsed ``ToolHandler`` instance before the tool executes, guards are
    also the right place for **logging and auditing**::

        class AuditGuard(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                logger.info("tool=%s args=%s", type(handler).__name__, handler.model_dump())
                return handler  # always allow

    See ``examples/agent_swarm/agent_swarm/__main__.py`` (``RoleGuard``) for a production example.
    """

    async def __call__(self, handler: Any) -> Any:
        return await self.check(handler)

    @abstractmethod
    async def check(self, handler: Any) -> Any: ...


class ConcurrentGuard(PermissionGuard, ABC):
    """Guard with concurrency control.

    Subclass and override ``check()``.  ``__call__`` acquires the semaphore
    then delegates to ``check()``.  Set ``concurrency`` to control parallelism
    (default 1 — one check at a time).
    """

    concurrency: int = 1

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def __call__(self, handler: Any) -> Any:
        async with self._semaphore:
            return await self.check(handler)


class AllowAllGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        return handler


class DenyAllGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        raise GuardError("denied")
