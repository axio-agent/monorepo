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

## Custom transport with retry

Transport with exponential backoff:

<!-- name: test_retry_transport -->
```python
from typing import AsyncIterator
from axio import Tool, StreamEvent, TextDelta, IterationEnd, StopReason, Usage
from axio.messages import Message


class RetryTransport:
    """Transport with retry logic."""
    max_retries = 3
    base_delay = 0.01

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str,
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(index=0, delta="response")
        yield IterationEnd(
            iteration=1,
            stop_reason=StopReason.end_turn,
            usage=Usage(0, 0),
        )
```

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