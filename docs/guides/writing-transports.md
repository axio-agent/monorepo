# Writing Transports

A transport connects Axio to an LLM provider. Implement the
`CompletionTransport` protocol to add support for any API.

## The protocol

```python
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

```python
from collections.abc import AsyncIterator
from axio import CompletionTransport, Message, Tool, StreamEvent
from axio.events import TextDelta, IterationEnd
from axio.types import StopReason, Usage


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
```

## Event contract

Your transport should yield these events in order:

1. **Content events** — any mix of:
   - `TextDelta` for text chunks
   - `ReasoningDelta` for reasoning/thinking chunks
   - `ToolUseStart` followed by `ToolInputDelta` for tool calls

2. **`IterationEnd`** — exactly once at the end, with:
   - `iteration`: the agent passes this, but transports can use `1`
   - `stop_reason`: `end_turn`, `tool_use`, `max_tokens`, or `error`
   - `usage`: token counts for this call

### Tool calls

When the LLM wants to call a tool, yield:

```python
yield ToolUseStart(index=0, tool_use_id="call_abc", name="my_tool")
yield ToolInputDelta(index=0, tool_use_id="call_abc", partial_json='{"arg": "value"}')
yield IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=usage)
```

The agent assembles `ToolInputDelta` fragments into complete JSON. You can
yield multiple `ToolInputDelta` events for the same tool call if the API
streams the JSON incrementally.

### Multiple tool calls

For parallel tool calls, use different `index` values:

```python
yield ToolUseStart(index=0, tool_use_id="call_1", name="tool_a")
yield ToolUseStart(index=1, tool_use_id="call_2", name="tool_b")
yield ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"x": 1}')
yield ToolInputDelta(index=1, tool_use_id="call_2", partial_json='{"y": 2}')
yield IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=usage)
```

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

- Stream tokens as they arrive — don't buffer the full response.
- Track token usage accurately for cost monitoring.
- Handle API errors gracefully: yield `IterationEnd` with
  `stop_reason=StopReason.error` rather than letting exceptions propagate.
- Look at `axio-transport-openai` for a production-grade reference
  implementation using `aiohttp` and SSE parsing.
