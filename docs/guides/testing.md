# Testing

Axio ships with testing helpers in `axio.testing` that make it easy to write
fast, deterministic tests for agents, tools, and custom components.

## StubTransport

`StubTransport` is a fake transport that yields pre-configured event
sequences instead of calling a real LLM:

<!-- name: test_stub_transport -->
```python
from axio.testing import StubTransport, make_text_response

transport = StubTransport([
    make_text_response("Hello!"),
])
assert len(transport._responses) == 1
assert transport._call_count == 0
```

Each entry in the list is one transport call. The stub cycles through them
in order, repeating the last one if the agent makes more calls than expected.

## Factory functions

### make_text_response

Create an event sequence for a simple text reply:

<!-- name: test_make_text_response -->
```python
from axio import TextDelta, IterationEnd, StopReason, Usage
from axio.testing import make_text_response

events = make_text_response(text="Hello world", iteration=1)
assert events == [
    TextDelta(index=0, delta="Hello world"),
    IterationEnd(
        iteration=1, 
        stop_reason=StopReason.end_turn,
        usage=Usage(input_tokens=10, output_tokens=5),
    ),
]
```

### make_tool_use_response

Create an event sequence for a tool call:

<!-- name: test_make_tool_use_response -->
```python
from axio import ToolUseStart, ToolInputDelta, IterationEnd, StopReason
from axio.testing import make_tool_use_response

events = make_tool_use_response(
    tool_name="greet",
    tool_id="call_1",
    tool_input={"name": "Alice"},
    iteration=1,
)
assert len(events) == 3
assert isinstance(events[0], ToolUseStart)
assert events[0].name == "greet"
assert events[0].tool_use_id == "call_1"
assert isinstance(events[1], ToolInputDelta)
assert "Alice" in events[1].partial_json
assert isinstance(events[2], IterationEnd)
assert events[2].stop_reason == StopReason.tool_use
```

### make_stub_transport

Create transport that returns a single "Hello world" text response:

<!-- name: test_make_stub_transport -->
```python
from axio import TextDelta, IterationEnd
from axio.testing import make_stub_transport

transport = make_stub_transport()
assert len(transport._responses) == 1
assert isinstance(transport._responses[0][0], TextDelta)
assert isinstance(transport._responses[0][-1], IterationEnd)
```

### make_ephemeral_context

Create a fresh in-memory context store:

<!-- name: test_make_ephemeral_context -->
```python
from axio import MemoryContextStore
from axio.testing import make_ephemeral_context

context = make_ephemeral_context()
assert isinstance(context, MemoryContextStore)
assert context.session_id is not None
```

### make_echo_tool

Create a test tool that echoes its input as JSON:

<!-- name: test_make_echo_tool -->
```python
from axio.testing import make_echo_tool

tool = make_echo_tool()
assert tool.name == "echo"
assert "JSON" in tool.description
```

## Testing an agent with tools

A typical test sets up a stub that first requests a tool call, then returns
text after seeing the result:

<!-- name: test_agent_calls_tool -->
```python
import asyncio
from axio import Agent
from axio.testing import (
    StubTransport,
    make_tool_use_response,
    make_text_response,
    make_ephemeral_context,
    make_echo_tool,
)


async def test_agent_calls_tool():
    transport = StubTransport([
        make_tool_use_response("echo", tool_input={"text": "hello"}),
        make_text_response("Done!"),
    ])
    agent = Agent(
        system="You are a test agent.",
        tools=[make_echo_tool()],
        transport=transport,
    )
    result = await agent.run("Say hello", make_ephemeral_context())
    assert result == "Done!"

asyncio.run(test_agent_calls_tool())
```

No `@pytest.mark.asyncio` decorator needed - the project uses
`asyncio_mode = "auto"`.

## Testing tools in isolation

Test a tool handler directly:

<!-- name: test_word_count_isolation -->
```python
import asyncio
from axio import Tool


async def word_count(text: str) -> str:
    """Count words in text."""
    count = len(text.split())
    return f"The text contains {count} words."


async def test_word_count():
    tool = Tool(name="word_count", handler=word_count)
    result = await tool(text="one two three")
    assert "3" in result

asyncio.run(test_word_count())
```

Or test through the `Tool` wrapper to exercise guards:

<!-- name: test_word_count_via_tool -->
```python
import asyncio
from axio import Tool


async def word_count(text: str) -> str:
    """Count words in text."""
    count = len(text.split())
    return f"The text contains {count} words."


async def test_word_count_tool():
    tool = Tool(name="word_count", handler=word_count)
    result = await tool(text="one two three")
    assert "3" in result

asyncio.run(test_word_count_tool())
```

## Testing guards

<!-- name: test_guard_testing -->
```python
import asyncio
import pytest
from typing import Any
from axio import Tool, PermissionGuard, GuardError


async def word_count(text: str) -> str:
    """Count words."""
    return str(len(text.split()))


_tool: Tool[Any] = Tool(name="word_count", handler=word_count)


class MaxLengthGuard(PermissionGuard):
    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        for name, value in kwargs.items():
            if isinstance(value, str) and len(value) > self.max_length:
                raise GuardError(f"Field '{name}' exceeds {self.max_length} characters")
        return kwargs


async def test_guard_allows_short_input():
    guard = MaxLengthGuard(max_length=100)
    result = await guard.check(_tool, text="short")
    assert result == {"text": "short"}


async def test_guard_denies_long_input():
    guard = MaxLengthGuard(max_length=5)
    with pytest.raises(GuardError):
        await guard.check(_tool, text="this is way too long")

asyncio.run(test_guard_allows_short_input())
asyncio.run(test_guard_denies_long_input())
```

## Testing context stores

<!-- name: test_context_stores -->
```python
import asyncio
from axio import MemoryContextStore
from axio.messages import Message
from axio.blocks import TextBlock


async def test_context_append_and_history():
    ctx = MemoryContextStore()
    msg = Message(role="user", content=[TextBlock(text="hello")])
    await ctx.append(msg)
    history = await ctx.get_history()
    assert len(history) == 1
    assert history[0].role == "user"

asyncio.run(test_context_append_and_history())
```
