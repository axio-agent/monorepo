# Testing

Axio ships with testing helpers in `axio.testing` that make it easy to write
fast, deterministic tests for agents, tools, and custom components.

## StubTransport

`StubTransport` is a fake transport that yields pre-configured event
sequences instead of calling a real LLM:

```python
from axio.testing import StubTransport, make_text_response

transport = StubTransport([
    make_text_response("Hello!"),
])
```

Each entry in the list is one transport call. The stub cycles through them
in order, repeating the last one if the agent makes more calls than expected.

## Factory functions

### make_text_response

Create an event sequence for a simple text reply:

```python
from axio.testing import make_text_response

events = make_text_response(text="Hello world", iteration=1)
# [TextDelta(index=0, delta="Hello world"),
#  IterationEnd(iteration=1, stop_reason=StopReason.end_turn, usage=...)]
```

### make_tool_use_response

Create an event sequence for a tool call:

```python
from axio.testing import make_tool_use_response

events = make_tool_use_response(
    tool_name="greet",
    tool_id="call_1",
    tool_input={"name": "Alice"},
    iteration=1,
)
# [ToolUseStart(index=0, tool_use_id="call_1", name="greet"),
#  ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"name":"Alice"}'),
#  IterationEnd(iteration=1, stop_reason=StopReason.tool_use, usage=...)]
```

### make_stub_transport

Create a transport that returns a single "Hello world" text response:

```python
from axio.testing import make_stub_transport

transport = make_stub_transport()
```

### make_ephemeral_context

Create a fresh in-memory context store:

```python
from axio.testing import make_ephemeral_context

context = make_ephemeral_context()
```

### make_echo_tool

Create a test tool that echoes its input as JSON:

```python
from axio.testing import make_echo_tool

tool = make_echo_tool()
```

## Testing an agent with tools

A typical test sets up a stub that first requests a tool call, then returns
text after seeing the result:

```python
import pytest
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
```

No `@pytest.mark.asyncio` decorator needed — the project uses
`asyncio_mode = "auto"`.

## Testing tools in isolation

Test a tool handler directly:

```python
from my_package.tools import WordCount


async def test_word_count():
    handler = WordCount(text="one two three")
    result = await handler()
    assert "3" in result
```

Or test through the `Tool` wrapper to exercise guards:

```python
from axio import Tool
from my_package.tools import WordCount


async def test_word_count_tool():
    tool = Tool(name="word_count", description="Count words", handler=WordCount)
    result = await tool(text="one two three")
    assert "3" in result
```

## Testing guards

```python
from axio.exceptions import GuardError
from my_package.guards import MaxLengthGuard


async def test_guard_allows_short_input():
    guard = MaxLengthGuard(max_length=100)
    handler = WordCount(text="short")
    result = await guard.check(handler)
    assert result is handler


async def test_guard_denies_long_input():
    guard = MaxLengthGuard(max_length=5)
    handler = WordCount(text="this is way too long")
    with pytest.raises(GuardError):
        await guard.check(handler)
```

## Testing context stores

```python
from axio import MemoryContextStore, Message
from axio.blocks import TextBlock


async def test_context_append_and_history():
    ctx = MemoryContextStore()
    msg = Message(role="user", content=[TextBlock(text="hello")])
    await ctx.append(msg)
    history = await ctx.get_history()
    assert len(history) == 1
    assert history[0].role == "user"
```
