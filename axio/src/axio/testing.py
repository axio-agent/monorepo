"""Shared test helpers: StubTransport, fixtures, response builders."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .context import MemoryContextStore
from .events import IterationEnd, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from .messages import Message
from .tool import Tool
from .types import StopReason, Usage


async def _msg_input(msg: str) -> str:
    return json.dumps({"msg": msg})


class StubTransport:
    """A CompletionTransport that yields pre-configured event sequences.

    Each call to stream() pops the next sequence from the list.
    """

    def __init__(self, responses: list[list[StreamEvent]] | None = None) -> None:
        self._responses: list[list[StreamEvent]] = list(responses or [])
        self._call_count = 0

    async def _generate(self, events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        idx = min(self._call_count, len(self._responses) - 1)
        events = self._responses[idx]
        self._call_count += 1
        return self._generate(events)


def make_tool_use_response(
    tool_name: str = "echo",
    tool_id: str = "call_1",
    tool_input: dict[str, Any] | None = None,
    iteration: int = 1,
    usage: Usage | None = None,
) -> list[StreamEvent]:
    """Build a standard tool_use response event sequence."""
    inp = tool_input or {"msg": "hi"}
    u = usage or Usage(10, 5)
    return [
        ToolUseStart(0, tool_id, tool_name),
        ToolInputDelta(0, tool_id, json.dumps(inp)),
        IterationEnd(iteration, StopReason.tool_use, u),
    ]


def make_text_response(text: str = "Done", iteration: int = 2, usage: Usage | None = None) -> list[StreamEvent]:
    """Build a standard end_turn text response event sequence."""
    u = usage or Usage(10, 5)
    return [
        TextDelta(0, text),
        IterationEnd(iteration, StopReason.end_turn, u),
    ]


def make_stub_transport() -> StubTransport:
    return StubTransport(
        [
            [
                TextDelta(0, "Hello"),
                TextDelta(0, " world"),
                IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
            ]
        ]
    )


def make_ephemeral_context() -> MemoryContextStore:
    return MemoryContextStore()


def make_echo_tool() -> Tool[Any]:
    return Tool(name="echo", description="Returns input as JSON", handler=_msg_input)
