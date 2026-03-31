# Permission Guards

Guards gate tool execution. They sit between parameter validation and handler
invocation, forming a composable middleware chain that can allow, deny, or
modify tool calls.

## PermissionGuard ABC

```python
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

```python
# Inside Tool.__call__
for guard in self.guards:
    instance = await guard(instance)
```

This lets you compose guards freely. For example, a path-validation guard
followed by an LLM-based risk-assessment guard:

```python
Tool(
    name="write_file",
    description="Write a file",
    handler=WriteFile,
    guards=(PathGuard(), LLMGuard()),
)
```

## ConcurrentGuard

For guards that need concurrency control (e.g., rate-limiting or guards that
call external services), subclass `ConcurrentGuard`:

```python
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
