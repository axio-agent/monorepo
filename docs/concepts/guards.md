# Permission Guards

Guards gate tool execution. They sit between parameter validation and handler
invocation, forming a composable middleware chain that can allow, deny, or
modify tool calls.

## PermissionGuard ABC

<!-- name: test_permission_guard_abc -->
```python
from abc import ABC, abstractmethod
from typing import Any
from axio import Tool


class PermissionGuard(ABC):
    async def __call__(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        return await self.check(tool, **kwargs)

    @abstractmethod
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]: ...
```

A guard receives the `Tool` object and the raw keyword arguments, and must either:

- **Return** a `dict` of (possibly modified) kwargs to allow execution.
- **Raise** `GuardError` to deny execution.

## Guard chain

When a `Tool` has multiple guards, they run **sequentially**. Each guard's
output kwargs become the next guard's input:

<!-- name: test_guard_chain -->
```python
import asyncio
from typing import Any
from axio.permission import AllowAllGuard
from axio import Tool


async def echo(text: str) -> str:
    """Echo text."""
    return text


_tool: Tool[Any] = Tool(name="echo", handler=echo)


async def main():
    guards = (AllowAllGuard(),)
    kwargs: dict[str, Any] = {"text": "hello"}
    for guard in guards:
        kwargs = await guard(_tool, **kwargs)
    assert kwargs["text"] == "hello"

asyncio.run(main())
```

This lets you compose guards freely. For example, a path-validation guard
followed by an LLM-based risk-assessment guard:

<!--
name: test_tool_with_guards
```python
from axio.permission import AllowAllGuard

async def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return "ok"

# Note: PathGuard and LLMGuard are provided by the axio-tui-guards package
# For testing, you can use AllowAllGuard as a placeholder:
PathGuard = AllowAllGuard
LLMGuard = AllowAllGuard
```
-->
<!-- name: test_tool_with_guards -->
```python
from axio import Tool

tool = Tool(
    name="write_file",
    handler=write_file,
    guards=(PathGuard(), LLMGuard()),
)
assert tool.name == "write_file"
assert len(tool.guards) == 2
```

## ConcurrentGuard

For guards that need concurrency control (e.g., rate-limiting or guards that
call external services), subclass `ConcurrentGuard`:

<!-- name: test_concurrent_guard -->
```python
import asyncio
from abc import ABC, abstractmethod
from typing import Any
from axio import PermissionGuard, Tool


class ConcurrentGuard(PermissionGuard, ABC):
    concurrency: int = 1

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def __call__(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        async with self._semaphore:
            return await self.check(tool, **kwargs)
```

Set the `concurrency` class variable to control how many tool calls can
pass through the guard simultaneously.

## Built-in guards

`AllowAllGuard`
: Always returns kwargs unchanged. Useful as a default.

`DenyAllGuard`
: Always raises `GuardError("denied")`. Useful for testing or disabling tools.

## Shipped guard plugins

The `axio-tui-guards` package provides two guards registered via the
`axio.guards` entry point group:

`PathGuard`
: Validates file paths against an allowed directory tree. Prevents tools
  from accessing files outside a configured root.

`LLMGuard`
: Uses a secondary LLM call to assess whether a tool call is safe.
  Provides a natural-language explanation when denying.

See [Writing Guards](../guides/writing-guards.md) for a step-by-step
guide to creating your own.
