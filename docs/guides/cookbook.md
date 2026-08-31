# Cookbook

Practical recipes for common Axio patterns.

## Agent with memory persistence

Save and restore conversation history:

<!-- name: test_memory_context -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response


transport = StubTransport([make_text_response("Hello!")])


async def main() -> None:
    context = MemoryContextStore()

    agent = Agent(
        system="You are a helpful assistant.",
        tools=[],
        transport=transport,
    )

    reply = await agent.run("Hi", context)
    print(reply)


asyncio.run(main())
```

## Streaming in FastAPI

Build a web API with streaming events:

<!-- name: test_streaming_pattern -->
```python
from axio import Agent, MemoryContextStore, TextDelta
from axio.testing import StubTransport, make_text_response


transport = StubTransport([make_text_response("Hello!")])


async def stream_events(message: str, agent: Agent, context: MemoryContextStore):
    """Pattern for streaming - yields events."""
    async for event in agent.run_stream(message, context):
        yield event
```

## RAG with custom tools

Combine retrieval and generation:

<!-- name: test_rag_tools -->
```python
from axio import Tool


async def retrieve_context(query: str) -> str:
    """Retrieve relevant context from a knowledge base."""
    return f"Results for: {query}"


async def generate_response(context: str, question: str) -> str:
    """Generate a response using retrieved context."""
    return f"Generated: {context[:50]}"


# Create tools
retrieve_tool = Tool(name="retrieve", handler=retrieve_context)
generate_tool = Tool(name="generate", handler=generate_response)
```

## Multi-agent workflow

Coordinate multiple agents:

<!-- name: test_multi_agent -->
```python
from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response


async def main():
    shared_context = MemoryContextStore()

    transport = StubTransport([
        make_text_response("Research result"),
        make_text_response("Final summary"),
    ])

    research_agent = Agent(
        system="Research the topic.",
        tools=[],
        transport=transport,
        context=shared_context,
    )

    writer_agent = Agent(
        system="Write a summary.",
        tools=[],
        transport=transport,
        context=shared_context,
    )

    research_result = await research_agent.run("What is async?")
    final_result = await writer_agent.run(f"Summary: {research_result}")
    print(final_result)
```

## Retrying another transport

A transport that wraps another one and retries it. Retrying is only safe before the first event
reaches the caller. Once a delta has been yielded the turn is half-delivered, and starting over
repeats it.

<!-- name: test_retry_transport -->
```python
import asyncio
from collections.abc import AsyncIterator
from typing import Any
from axio import Agent, MemoryContextStore, StopReason, StreamEvent, TextDelta, Tool, Usage
from axio.events import IterationEnd
from axio.exceptions import StreamError
from axio.messages import Message


class RetryTransport:
    """Retries the wrapped transport on StreamError, with exponential backoff."""

    def __init__(self, inner: Any, max_retries: int = 3, base_delay: float = 0.01) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        for attempt in range(1, self.max_retries + 1):
            delivered = False
            try:
                async for event in self.inner.stream(messages, tools, system):
                    delivered = True
                    yield event
                return
            except StreamError:
                # Nothing to retry once the caller has seen part of the turn.
                if delivered or attempt == self.max_retries:
                    raise
                await asyncio.sleep(self.base_delay * 2 ** (attempt - 1))


class Flaky:
    """Fails twice, then answers."""

    def __init__(self) -> None:
        self.calls = 0

    def stream(
        self,
        messages: list[Message],
        tools: list[Tool[Any]],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        return self._answer()

    async def _answer(self) -> AsyncIterator[StreamEvent]:
        if self.calls < 3:
            raise StreamError("503 Service Unavailable")
        yield TextDelta(index=0, delta="ok")
        yield IterationEnd(iteration=0, stop_reason=StopReason.end_turn, usage=Usage(1, 1))


async def main() -> None:
    flaky = Flaky()
    agent = Agent(system="", transport=RetryTransport(flaky))
    assert await agent.run("hi", MemoryContextStore()) == "ok"
    assert flaky.calls == 3

asyncio.run(main())
```

The shipped transports retry inside themselves rather than through a wrapper. They prefer the
`Retry-After` header over their own backoff when the response carries one. See
{doc}`writing-transports` for what a transport is expected to do with errors.

## Rate limiting tool

<!-- name: test_rate_limit -->
```python
import asyncio
from axio import Tool, CONTEXT

RATE_LIMIT = 10
TIME_WINDOW = 60


async def rate_limited_action(data: str) -> str:
    """Tool with rate limiting."""
    calls: list[float] = CONTEXT.get()
    now = asyncio.get_event_loop().time()
    # Prune old calls outside the window
    calls[:] = [t for t in calls if now - t < TIME_WINDOW]
    if len(calls) >= RATE_LIMIT:
        raise RuntimeError(f"Rate limit: {RATE_LIMIT}/{TIME_WINDOW}s")
    calls.append(now)
    return "done"

call_log: list[float] = []
tool = Tool(name="rate_limited_action", handler=rate_limited_action, context=call_log)
```

## API key guard

Check for required environment variables:

<!-- name: test_api_key_guard -->
```python
import os
from typing import Any
from axio import PermissionGuard, GuardError


class ApiKeyGuard(PermissionGuard):
    """Ensure required environment variables are set."""
    required_keys = ("OPENAI_API_KEY",)

    async def check(self, handler: Any) -> Any:
        missing = [k for k in self.required_keys if not os.environ.get(k)]
        if missing:
            raise GuardError(f"Missing: {', '.join(missing)}")
        return handler
```

## Tool with guards

Apply guards to specific tools:

<!-- name: test_tool_with_guards -->
```python
from typing import Any
from axio import Tool, PermissionGuard


async def sensitive_operation(data: str) -> str:
    """Process sensitive data."""
    return f"Processed: {data}"


class AllowGuard(PermissionGuard):
    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        return kwargs


sensitive_tool = Tool(
    name="sensitive_operation",
    handler=sensitive_operation,
    guards=(AllowGuard(),),
)

assert len(sensitive_tool.guards) == 1
```