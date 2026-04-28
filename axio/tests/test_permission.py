"""Tests for axio.permission: guards."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from axio.exceptions import GuardError
from axio.permission import AllowAllGuard, ConcurrentGuard, DenyAllGuard, PermissionGuard
from axio.tool import Tool


async def _noop() -> str:
    return ""


_tool: Tool[Any] = Tool(name="t", description="t", handler=_noop)


class TestAllowAllGuard:
    async def test_returns_kwargs(self) -> None:
        guard = AllowAllGuard()
        result = await guard.check(_tool, x=1, y="hello")
        assert result == {"x": 1, "y": "hello"}

    def test_satisfies_protocol(self) -> None:
        assert isinstance(AllowAllGuard(), PermissionGuard)


class TestDenyAllGuard:
    async def test_raises(self) -> None:
        guard = DenyAllGuard()
        with pytest.raises(GuardError, match="denied"):
            await guard.check(_tool)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(DenyAllGuard(), PermissionGuard)


class TestConcurrentGuard:
    def test_satisfies_protocol(self) -> None:
        class _G(ConcurrentGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                return kwargs

        assert isinstance(_G(), PermissionGuard)

    async def test_limits_parallel_checks(self) -> None:
        """ConcurrentGuard.concurrency must cap simultaneous check() executions."""
        running = 0
        max_running = 0

        class _CountGuard(ConcurrentGuard):
            concurrency = 2

            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                nonlocal running, max_running
                running += 1
                max_running = max(max_running, running)
                await asyncio.sleep(0)
                running -= 1
                return kwargs

        guard = _CountGuard()
        await asyncio.gather(*[guard(_tool) for _ in range(10)])
        assert max_running <= 2

    async def test_semaphore_released_on_guard_error(self) -> None:
        """Semaphore must be released even when check() raises GuardError."""

        class _FailGuard(ConcurrentGuard):
            concurrency = 1

            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                raise GuardError("nope")

        guard = _FailGuard()
        for _ in range(3):
            with pytest.raises(GuardError):
                await guard(_tool)
        # If the semaphore leaked, this would deadlock; completing proves it didn't
        assert guard._semaphore._value == 1

    async def test_default_concurrency_is_one(self) -> None:
        class _G(ConcurrentGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                return kwargs

        assert _G.concurrency == 1
        assert _G()._semaphore._value == 1
