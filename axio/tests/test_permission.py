"""Tests for axon.permission: guards."""

from __future__ import annotations

import pytest

from axio.exceptions import GuardError
from axio.permission import AllowAllGuard, DenyAllGuard, PermissionGuard


class TestAllowAllGuard:
    async def test_returns_handler(self) -> None:
        guard = AllowAllGuard()
        sentinel = object()
        result = await guard.check(sentinel)
        assert result is sentinel

    def test_satisfies_protocol(self) -> None:
        assert isinstance(AllowAllGuard(), PermissionGuard)


class TestDenyAllGuard:
    async def test_raises(self) -> None:
        guard = DenyAllGuard()
        with pytest.raises(GuardError, match="denied"):
            await guard.check(object())

    def test_satisfies_protocol(self) -> None:
        assert isinstance(DenyAllGuard(), PermissionGuard)
