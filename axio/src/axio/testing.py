"""Shared test helpers: StubTransport, fixtures, response builders."""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from .context import MemoryContextStore
from .events import IterationEnd, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from .messages import Message
from .models import Capability, ModelRegistry, ModelSpec
from .tool import Tool
from .types import StopReason, Usage


async def _msg_input(msg: str) -> str:
    return json.dumps({"msg": msg})


class _StaticModelTransport(Protocol):
    model: ModelSpec
    models: ModelRegistry


class StubTransport:
    """A CompletionTransport that yields pre-configured event sequences.

    Each call to stream() pops the next sequence from the list.
    """

    def __init__(self, responses: Sequence[Sequence[StreamEvent | BaseException]] | None = None) -> None:
        #: An exception among the events is raised where it sits, which is how a real transport
        #: reports a failure. ``IterationEnd`` cannot carry ``StopReason.error``.
        self._responses: list[Sequence[StreamEvent | BaseException]] = list(responses or [])
        self._call_count = 0

    async def _generate(self, events: Sequence[StreamEvent | BaseException]) -> AsyncIterator[StreamEvent]:
        for event in events:
            if isinstance(event, BaseException):
                raise event
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


def assert_static_model_transport_contract(transport: _StaticModelTransport) -> None:
    """Check invariants shared by transports with a ready, bundled model catalog."""
    models = transport.models
    current = transport.model

    assert isinstance(models, ModelRegistry), f"models must be a ModelRegistry, got {type(models).__name__}"
    assert models, "the model registry must not be empty"
    assert current.id in models, f"the selected model {current.id!r} is absent from its registry"
    assert models[current.id] == current, f"the selected model {current.id!r} has stale registry metadata"

    for model_id, spec in models.items():
        assert isinstance(spec, ModelSpec), f"{model_id!r} is not a ModelSpec"
        assert model_id == spec.id, f"registry key {model_id!r} does not match ModelSpec.id {spec.id!r}"
        assert spec.id and spec.id == spec.id.strip(), f"invalid model id {spec.id!r}"
        assert isinstance(spec.capabilities, frozenset), f"{model_id!r} capabilities must be a frozenset"
        assert spec.capabilities and all(isinstance(cap, Capability) for cap in spec.capabilities), (
            f"{model_id!r} has invalid capabilities"
        )
        assert Capability.text in spec.capabilities, f"{model_id!r} cannot accept text"
        assert spec.context_window > 0, f"{model_id!r} has a non-positive context window"
        assert spec.max_output_tokens > 0, f"{model_id!r} has a non-positive output token limit"
        assert math.isfinite(spec.input_cost) and spec.input_cost >= 0, f"{model_id!r} has invalid input cost"
        assert math.isfinite(spec.output_cost) and spec.output_cost >= 0, f"{model_id!r} has invalid output cost"


def assert_stream_contract(events: Sequence[StreamEvent]) -> None:
    """Check what every ``CompletionTransport.stream()`` must produce.

    A transport that breaks one of these still passes its own tests, because the agent papers over
    the difference. Call this from each transport's tests on whatever its fake server produced.
    """
    # StopReason.error is not checked here: `IterationEnd.__post_init__` refuses it, so no such
    # event can reach this function. A transport that tries raises where it builds one.
    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 1, f"a stream ends with exactly one IterationEnd, got {len(ends)}"
    assert events[-1] is ends[0], "IterationEnd is the last event"
    usage = ends[0].usage
    assert usage.cache_read_tokens + usage.cache_write_tokens <= usage.input_tokens, (
        f"the cache slices are inside input_tokens, got {usage}"
    )
    assert usage.reasoning_tokens <= usage.output_tokens, f"reasoning is inside output_tokens, got {usage}"
