# Permission Guards

Guards gate tool execution. They sit between parameter validation and handler
invocation, forming a composable middleware chain that can allow, deny, or
modify tool calls.

## PermissionGuard ABC

<!-- name: test_permission_guard_abc -->
```python
from abc import ABC, abstractmethod
from typing import Any


class PermissionGuard(ABC):
    async def __call__(self, handler: Any) -> Any:
        return await self.check(handler)

    @abstractmethod
    async def check(self, handler: Any) -> Any: ...
```

A guard receives the validated `ToolHandler` instance and must either:

- **Return** the handler (possibly modified) to allow execution.
- **Raise** `GuardError` to deny execution.

## Guard chain

When a `Tool` has multiple guards, they run **sequentially**. Each guard's
output becomes the next guard's input:

<!-- name: test_guard_chain -->
```python
import asyncio
from typing import Any
from axio.permission import AllowAllGuard
from axio.tool import ToolHandler


class EchoHandler(ToolHandler[Any]):
    """Echo text."""
    text: str
    async def __call__(self, context: Any) -> str:
        return self.text


async def main():
    guards = (AllowAllGuard(),)
    instance = EchoHandler(text="hello")
    for guard in guards:
        instance = await guard(instance)
    assert instance.text == "hello"

asyncio.run(main())
```

This lets you compose guards freely. For example, a path-validation guard
followed by an LLM-based risk-assessment guard:

<!--
name: test_tool_with_guards
```python
from typing import Any
from axio.tool import ToolHandler
from axio.permission import AllowAllGuard
class WriteFile(ToolHandler[Any]):
    path: str
    content: str
    async def __call__(self, context: Any) -> str: return "ok"

# Note: PathGuard and LLMGuard are provided by the axio-tui-guards package
# For testing, you can use AllowAllGuard as a placeholder:
PathGuard = AllowAllGuard
LLMGuard = AllowAllGuard
```
-->
<!-- name: test_tool_with_guards -->
```python
from axio.tool import Tool

tool = Tool(
    name="write_file",
    description="Write a file",
    handler=WriteFile,
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
from axio.permission import PermissionGuard


class ConcurrentGuard(PermissionGuard, ABC):
    concurrency: int = 1

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrency)

    async def __call__(self, handler: Any) -> Any:
        async with self._semaphore:
            return await self.check(handler)
```

Set the `concurrency` class variable to control how many tool calls can
pass through the guard simultaneously.

## Built-in guards

`AllowAllGuard`
: Always returns the handler unchanged. Useful as a default.

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
