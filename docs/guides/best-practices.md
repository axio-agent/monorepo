# Best Practices

Guidelines for building reliable, maintainable Axio applications.

## Tool Handlers

### Keep handlers focused

Each tool should do one thing well. If you find yourself writing "and also...", split into multiple
tools.

<!-- name: test_focused_tools -->
```python
from axio import Tool


async def fetch_url(url: str) -> str:
    """Fetch a URL and return its content."""
    ...


async def parse_json(data: str) -> dict:
    """Parse a JSON string and return a dict."""
    ...
```

### Use descriptive names and docstrings

The LLM uses these to decide when to call your tool.

<!-- name: test_descriptive_docstrings -->
```python
from axio import Tool


async def geo_locate(ip: str = "auto") -> str:
    """Get geographic location from IP address using ip-api.com.

    Returns city, country, and coordinates as JSON.
    """
    return '{"city": "NYC", "country": "US"}'
```

### Validate inputs with Field

Use `Annotated` + `Field` from `axio.field` for validation:

<!-- name: test_field_validation -->
```python
from typing import Annotated
from axio import Tool, Field


async def fetch_url(
    url: Annotated[str, Field(description="HTTP or HTTPS URL")],
    timeout: Annotated[int, Field(default=10, ge=1, le=60)] = 10,
) -> str:
    """Fetch a URL."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return f"fetched {url}"
```

### Return structured data as JSON

Tool results are always coerced to `str`. Return `json.dumps(...)` for machine-readable output:

<!-- name: test_structured_result -->
```python
import json
from axio import Tool


async def calculate(a: float, b: float, operation: str) -> str:
    """Perform calculations."""
    ops = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else 0,
    }
    result = ops[operation](a, b)
    return json.dumps({"result": result, "operation": operation})
```

## Error Handling

### Raise HandlerError for expected failures

<!-- name: test_handler_error -->
```python
from pathlib import Path
from axio import Tool, HandlerError


async def read_file(path: str) -> str:
    """Read a file and return its content."""
    p = Path(path)
    if not p.exists():
        raise HandlerError(f"File not found: {path}")
    return p.read_text()
```

### Use guards for validation

Move input validation to guards to keep handlers clean:

<!-- name: test_guard_validation -->
```python
from typing import Any
from axio import PermissionGuard


class SanitizeInput(PermissionGuard):
    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            k: v.replace("<script>", "") if isinstance(v, str) else v
            for k, v in kwargs.items()
        }
```

## Testing

### Test tools in isolation

<!-- name: test_isolated_tool -->
```python
import asyncio
from axio import Tool


async def fetch_url(url: str) -> str:
    """Fetch a URL."""
    return f"Fetched: {url}"


async def test_fetch():
    tool = Tool(name="fetch_url", handler=fetch_url)
    result = await tool(url="https://example.com")
    assert "example.com" in result

asyncio.run(test_fetch())
```

### Use StubTransport for agent tests

<!-- name: test_stub_transport_agent -->
```python
from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_tool_use_response, make_text_response


async def test_agent_with_tool():
    transport = StubTransport([
        make_tool_use_response("fetch", tool_input={"url": "..."}),
        make_text_response("Done"),
    ])
    agent = Agent(tools=[], transport=transport)
    result = await agent.run("Fetch example.com", MemoryContextStore())
    assert "Done" in result
```

### Test guards separately

<!-- name: test_guard_testing_separate -->
```python
import asyncio
import pytest
from typing import Any
from axio import Tool, PermissionGuard, GuardError


async def word_count(text: str) -> str:
    """Count words."""
    return str(len(text.split()))


class MaxLengthGuard(PermissionGuard):
    def __init__(self, max_length: int = 10000) -> None:
        self.max_length = max_length

    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        for name, value in kwargs.items():
            if isinstance(value, str) and len(value) > self.max_length:
                raise GuardError(f"Field '{name}' exceeds {self.max_length}")
        return kwargs


_tool: Tool[Any] = Tool(name="word_count", handler=word_count)


async def test_max_length_guard_allows():
    guard = MaxLengthGuard(max_length=100)
    result = await guard(_tool, text="short")
    assert result == {"text": "short"}


async def test_max_length_guard_denies():
    guard = MaxLengthGuard(max_length=5)
    with pytest.raises(GuardError):
        await guard(_tool, text="this is way too long")

asyncio.run(test_max_length_guard_allows())
asyncio.run(test_max_length_guard_denies())
```

## Configuration

### Use environment variables for secrets

<!-- name: test_env_variables -->
```python
import os
from dataclasses import dataclass, field


@dataclass
class MyTransport:
    api_key: str = field(default_factory=lambda: os.environ.get("MY_API_KEY", ""))
```

### Separate config from code

For complex applications, load configuration from files:

<!-- name: test_pydantic_settings -->
```python
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    database_url: str
    openai_api_key: str
    default_model: str = "gpt-4"
```

## Performance

### Limit concurrency on expensive tools

<!-- name: test_concurrency_limit -->
```python
from axio import Tool


async def slow_api_call() -> str:
    """Slow external API call."""
    return "done"


tool = Tool(
    name="slow_api_call",
    handler=slow_api_call,
    concurrency=2,
)

assert tool.concurrency == 2
```

### Reuse context stores

Don't create a new context store for each request:

<!-- name: test_reuse_context -->
```python
from axio import MemoryContextStore


# Bad: new store each time
async def handle_request_bad(msg: str) -> None:
    context = MemoryContextStore()


# Good: reuse store
context = MemoryContextStore()


async def handle_request_good(msg: str) -> None:
    pass
```

## Security

### Always use guards for sensitive operations

<!-- name: test_sensitive_tool_guards -->
```python
from typing import Any
from axio import Tool, PermissionGuard


async def run_sql(query: str) -> str:
    """Execute a SQL query."""
    return "result"


class ApiKeyGuard(PermissionGuard):
    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        return kwargs


class RateLimitGuard(PermissionGuard):
    def __init__(self, max_per_minute: int = 10) -> None:
        self.max_per_minute = max_per_minute

    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        return kwargs


tool = Tool(
    name="exec_sql",
    handler=run_sql,
    guards=(
        ApiKeyGuard(),
        RateLimitGuard(max_per_minute=10),
    ),
)

assert len(tool.guards) == 2
```

### Validate file path guards

If your tool accesses files, validate paths:

<!-- name: test_path_guard -->
```python
from pathlib import Path
from typing import Any
from axio import PermissionGuard, GuardError


class PathGuard(PermissionGuard):
    allowed_dirs: tuple[str, ...] = ("/tmp",)

    async def check(self, tool: Any, **kwargs: Any) -> dict[str, Any]:
        path = Path(kwargs.get("path", "")).resolve()
        allowed = [Path(d).resolve() for d in self.allowed_dirs]
        if not any(path.is_relative_to(a) for a in allowed):
            raise GuardError(f"Path not allowed: {kwargs.get('path')}")
        return kwargs
```

## Code Organization

### Use type hints everywhere

Axio uses strict typing. Type hints help catch errors early:

<!-- name: test_type_hints -->
```python
from axio import Tool


async def my_tool(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search and return results."""
    return [{"id": "1", "name": "test"}]
```

## Production Deployment

### Use SQLite for production

MemoryContextStore loses data on shutdown. Use SQLiteContextStore for persistence. The caller owns
the connection, so open it once and close it on shutdown:

<!-- name: test_sqlite_for_production -->
```python
import asyncio
from axio_context_sqlite import SQLiteContextStore, connect


async def main() -> None:
    conn = await connect(":memory:")  # a file path in production
    try:
        context = SQLiteContextStore(conn, session_id="user-42")
        assert context.session_id == "user-42"
    finally:
        await conn.close()

asyncio.run(main())
```

### Monitor token usage

`IterationEnd.usage` is one request. `SessionEndEvent.total_usage` is the whole run, every slice
summed. `input_tokens` and `output_tokens` are inclusive grand totals. `cache_read_tokens`,
`cache_write_tokens` and `reasoning_tokens` are disjoint slices of one of them. That holds
whichever provider answered, because each transport converts into that rule.

`Usage` is counts, never money. A cached token and a written one bill at different multipliers, so
price the slices separately against your own per-model rates:

<!-- name: test_monitor_usage -->
```python
import asyncio
from axio import Agent, MemoryContextStore, StopReason, TextDelta, Usage
from axio.events import IterationEnd, SessionEndEvent
from axio.testing import StubTransport

# Dollars per million tokens, from the provider's price list.
RATES = {"cached_input": 0.30, "input": 3.00, "output": 15.00}


def cost(usage: Usage) -> float:
    return (
        usage.uncached_input_tokens * RATES["input"]
        + usage.cache_read_tokens * RATES["cached_input"]
        + usage.output_tokens * RATES["output"]
    ) / 1_000_000


async def main() -> None:
    transport = StubTransport([[
        TextDelta(index=0, delta="done"),
        IterationEnd(
            iteration=0,
            stop_reason=StopReason.end_turn,
            usage=Usage(1000, 200, cache_read_tokens=800, reasoning_tokens=150),
        ),
    ]])
    agent = Agent(system="", transport=transport)

    total = Usage(0, 0)
    async for event in agent.run_stream("hi", MemoryContextStore()):
        if isinstance(event, SessionEndEvent):
            total = event.total_usage

    # 800 of the 1000 input tokens were served from cache, and 150 of the 200 output tokens
    # were reasoning the caller never saw.
    assert total.uncached_input_tokens == 200
    assert total.answer_tokens == 50
    assert round(cost(total), 6) == round((200 * 3.0 + 800 * 0.30 + 200 * 15.0) / 1_000_000, 6)

asyncio.run(main())
```

A zero slice means the provider billed none of it, or reported no breakdown at all. Axio cannot
tell those apart, and neither can a cost line built on them.

### Read the stop reason, not just the text

`StopReason` has eight members and four of them are neither success nor a broken transport. Match
on `SessionEndEvent.stop_reason` before treating a short answer as a failure:

<!-- name: test_stop_reason_policy -->
```python
import asyncio
from axio import Agent, MemoryContextStore, Refusal, StopReason, Usage
from axio.events import IterationEnd, SessionEndEvent
from axio.testing import StubTransport


async def main() -> None:
    transport = StubTransport([[
        Refusal(index=0, text="I can't help with that.", category="policy"),
        IterationEnd(iteration=0, stop_reason=StopReason.refusal, usage=Usage(12, 5)),
    ]])
    agent = Agent(system="", transport=transport)

    stream = agent.run_stream("...", MemoryContextStore())
    said, end = "", None
    async for event in stream:
        if isinstance(event, Refusal):
            said = event.text
        if isinstance(event, SessionEndEvent):
            end = event

    assert end is not None and end.stop_reason is StopReason.refusal
    # A decline is a finished turn, not an error: retrying the same prompt cannot succeed.
    assert said == "I can't help with that."

asyncio.run(main())
```

`agent.run()` returns the refusal text too, so a declined turn no longer looks like an empty answer.
`refusal`, `context_window_exceeded` and `cancelled` all end the run, and so do `unknown` and
`repetition`. `pause_turn` does not — the agent resumes on it. A provider reason the transport
could not map arrives as `StopReason.unknown`, with the provider's own word in `IterationEnd.raw`;
`StopReason.error` is kept for the transport itself failing, and carries an `Error` event beside
it.
