"""Tests for axio.selector: ToolSelector protocol."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from axio.messages import Message
from axio.selector import ToolSelector
from axio.tool import Tool


class _TopK:
    """Simple concrete ToolSelector that keeps first N tools."""

    def __init__(self, k: int) -> None:
        self._k = k

    async def select(self, messages: Iterable[Message], tools: Iterable[Tool[Any]]) -> Iterable[Tool[Any]]:
        return list(tools)[: self._k]


async def _empty() -> str:
    return ""


def _make_tools(n: int) -> list[Tool[Any]]:
    return [Tool(name=f"t{i}", handler=_empty) for i in range(n)]


class TestToolSelectorProtocol:
    def test_runtime_checkable_positive(self) -> None:
        assert isinstance(_TopK(1), ToolSelector)

    def test_runtime_checkable_negative(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), ToolSelector)

    def test_runtime_checkable_missing_select(self) -> None:
        class _NoSelect:
            def other(self) -> None: ...

        assert not isinstance(_NoSelect(), ToolSelector)


class TestToolSelectorBehavior:
    async def test_returns_subset(self) -> None:
        tools = _make_tools(5)
        result = list(await _TopK(3).select([], tools))
        assert len(result) == 3
        assert result == tools[:3]

    async def test_returns_all_when_k_exceeds_count(self) -> None:
        tools = _make_tools(2)
        result = list(await _TopK(10).select([], tools))
        assert result == tools

    async def test_returns_empty_for_k_zero(self) -> None:
        tools = _make_tools(3)
        result = list(await _TopK(0).select([], tools))
        assert result == []

    async def test_empty_tools_list(self) -> None:
        result = list(await _TopK(5).select([], []))
        assert result == []
