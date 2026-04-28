# Writing Transports

A transport connects Axio to an LLM provider. Implement the
`CompletionTransport` protocol to add support for any API.

## The protocol

<!-- name: test_completion_transport_protocol -->
```python
from typing import runtime_checkable, Protocol
from collections.abc import AsyncIterator
from axio.messages import Message
from axio import Tool, StreamEvent


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]: ...
```

Your transport must yield `StreamEvent` values as they arrive from the LLM.
The agent expects the stream to end with an `IterationEnd` event.

## Minimal implementation

<!-- name: test_echo_transport -->
```python
import asyncio
from collections.abc import AsyncIterator
from axio import CompletionTransport, Tool, TextDelta, IterationEnd, StreamEvent, StopReason, Usage
from axio.messages import Message
from axio.blocks import TextBlock


class EchoTransport:
    """Transport that echoes the last user message (for testing)."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        # Find the last user message text
        last_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                for block in msg.content:
                    if hasattr(block, "text"):
                        last_text = block.text
                        break
                break

        # Yield it back as a text delta
        yield TextDelta(index=0, delta=f"Echo: {last_text}")

        # Always end with IterationEnd
        yield IterationEnd(
            iteration=1,
            stop_reason=StopReason.end_turn,
            usage=Usage(input_tokens=0, output_tokens=0),
        )


async def main():
    transport = EchoTransport()
    msgs = [Message(role="user", content=[TextBlock(text="ping")])]
    events = [e async for e in transport.stream(msgs, [], "")]
    assert isinstance(events[0], TextDelta)
    assert events[0].delta == "Echo: ping"
    assert isinstance(events[1], IterationEnd)
    assert events[1].stop_reason == StopReason.end_turn

asyncio.run(main())
```

In the example above `stream` is declared as an `async def` with `yield`
statements, making it an async generator. Production transports (e.g.
`AnthropicTransport`, `OpenAITransport`) instead declare `stream` as a plain
`def` that returns a call to a separate `async def _do_stream(...)` generator.
Both approaches satisfy the `CompletionTransport` protocol because both return
an `AsyncIterator[StreamEvent]`.

## Event contract

Your transport should yield these events in order:

1. **Content events** - any mix of:
   - `TextDelta` for text chunks
   - `ReasoningDelta` for reasoning/thinking chunks
   - `ToolUseStart` followed by `ToolInputDelta` for tool calls

2. **`IterationEnd`** - exactly once at the end, with:
   - `iteration`: the agent passes this, but transports can use `1`
   - `stop_reason`: `end_turn`, `tool_use`, `max_tokens`, or `error`
   - `usage`: token counts for this call

### Tool calls

When the LLM wants to call a tool, yield:

<!-- name: test_tool_call_events -->
```python
import asyncio
from axio import ToolUseStart, ToolInputDelta, IterationEnd, StopReason, Usage


async def example_tool_call_stream():
    usage = Usage(input_tokens=10, output_tokens=5)
    yield ToolUseStart(index=0, tool_use_id="call_abc", name="my_tool")
    yield ToolInputDelta(index=0, tool_use_id="call_abc", partial_json='{"arg": "value"}')
    yield IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=usage)


async def main():
    events = [e async for e in example_tool_call_stream()]
    assert len(events) == 3

asyncio.run(main())
```

The agent assembles `ToolInputDelta` fragments into complete JSON. You can
yield multiple `ToolInputDelta` events for the same tool call if the API
streams the JSON incrementally.

### Multiple tool calls

For parallel tool calls, use different `index` values:

<!-- name: test_multiple_tool_calls -->
```python
import asyncio
from axio import ToolUseStart, ToolInputDelta, IterationEnd, StopReason, Usage


async def example_parallel_stream():
    usage = Usage(input_tokens=10, output_tokens=5)
    yield ToolUseStart(index=0, tool_use_id="call_1", name="tool_a")
    yield ToolUseStart(index=1, tool_use_id="call_2", name="tool_b")
    yield ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"x": 1}')
    yield ToolInputDelta(index=1, tool_use_id="call_2", partial_json='{"y": 2}')
    yield IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=usage)


async def main():
    events = [e async for e in example_parallel_stream()]
    assert len(events) == 5

asyncio.run(main())
```

## TUI integration

Transports that integrate with the TUI should implement three additional
conventions: a `name` field, a `session` field, and `fetch_models()`,
`to_dict()`, and `from_dict()` methods. None of these are part of the core
`CompletionTransport` protocol, but the TUI expects them when loading and
displaying transports.

### `name` field

Declare a `name: str` dataclass field so the TUI can display the transport's
name in the welcome screen and command palette:

```python
from dataclasses import dataclass, field
import os


@dataclass(slots=True)
class MyTransport:
    name: str = "My Provider"
    api_key: str = field(default_factory=lambda: os.environ.get("MY_API_KEY", ""))
```

### `session` field

Declare a `session: aiohttp.ClientSession | None` field. The TUI creates a
single `aiohttp.ClientSession` and injects it into the transport before any
`stream()` calls. Using a shared session enables connection pooling and lets
the TUI manage the session lifetime (opening it on startup and closing it on
shutdown).

```python
import aiohttp
from dataclasses import dataclass, field


@dataclass(slots=True)
class MyTransport:
    name: str = "My Provider"
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)
```

Inside `stream()` (or the internal `_do_stream()` generator), assert that the
session is set before using it:

```python
assert self.session is not None, "session is required for streaming"
async with self.session.post(url, json=payload, headers=headers) as resp:
    ...
```

### `fetch_models()` method

The TUI calls `await transport.fetch_models()` during startup to verify that
the transport is reachable and to populate its model list. If the call raises,
the transport is marked as unavailable in the UI.

```python
async def fetch_models(self) -> None:
    """Refresh the available model list.

    May query the API (e.g. GET /models) or simply reset to a static registry.
    """
    self.models = MY_STATIC_MODEL_REGISTRY
```

The built-in transports (`AnthropicTransport`, `OpenAITransport`) use a static
model registry and simply reassign it in `fetch_models()`, but a transport
could also make a live API call here to discover available models.

### `to_dict()` and `from_dict()` methods

The TUI serialises transport configuration (API keys, base URLs, selected
model, etc.) to persistent storage using `to_dict()`, and deserialises it back
with the `from_dict()` class method. The serialised dictionary must be
JSON-compatible.

```python
from typing import Any, Self
import aiohttp


@dataclass(slots=True)
class MyTransport:
    name: str = "My Provider"
    base_url: str = "https://api.example.com/v1"
    api_key: str = ""
    session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> Self:
        return cls(
            name=str(data.get("name", "")),
            base_url=str(data.get("base_url", "")) or "https://api.example.com/v1",
            api_key=str(data.get("api_key", "")),
            session=session,
        )
```

Note that `from_dict` accepts `session` as a keyword-only argument so the
TUI can inject the shared session when reconstructing a transport from saved
configuration.

## Registering as a plugin

Add entry points to your `pyproject.toml`:

```toml
[project.entry-points."axio.transport"]
my_llm = "my_package:MyTransport"
```

Optionally provide a settings screen for the TUI:

```toml
[project.entry-points."axio.transport.settings"]
my_llm = "my_package:MySettingsScreen"
```

## Tips

- Stream tokens as they arrive - don't buffer the full response.
- Track token usage accurately for cost monitoring.
- Handle API errors gracefully: yield `IterationEnd` with
  `stop_reason=StopReason.error` rather than letting exceptions propagate.
- For retryable errors (HTTP 429, 5xx), implement exponential backoff with
  respect to the `Retry-After` response header when present.
- Look at `axio-transport-openai` and `axio-transport-anthropic` for
  production-grade reference implementations using `aiohttp` and SSE parsing.
