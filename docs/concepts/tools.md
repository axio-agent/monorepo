# Tool System

The tool system has two layers: **ToolHandler** (a Pydantic model defining
parameters and execution logic) and **Tool** (a frozen dataclass that wraps a
handler with metadata and guards).

## ToolHandler

<!--
name: test_tool_handler_interface
```python
from pydantic import BaseModel
```
-->
```python
class ToolHandler[T](BaseModel):
    """Subclass fields define JSON-schema for input parameters."""

    async def __call__(self, context: T) -> str:
        raise NotImplementedError
```

A tool handler is a Pydantic `BaseModel`. Its fields become the tool's input
schema automatically via `model_json_schema()`. The `__call__` method implements
the actual execution.

<!-- name: test_write_file_handler -->
```python
from typing import Any
from pathlib import Path
from axio.tool import ToolHandler

class WriteFile(ToolHandler[Any]):
    """Write content to a file at the given path."""
    path: str
    content: str

    async def __call__(self, context: Any) -> str:
        Path(self.path).write_text(self.content)
        return f"Wrote {len(self.content)} bytes to {self.path}"
```

The handler's **docstring** becomes the tool description sent to the LLM.

## Tool

```python
@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName
    description: str
    handler: type[ToolHandler[T]]
    guards: tuple[PermissionGuard, ...] = ()
    concurrency: int | None = None
    context: T = ...  # runtime default: empty MappingProxyType
```

`handler`
: The handler **class**, not an instance. The tool creates a new instance
  for each invocation via `handler.model_validate(kwargs)`.

`guards`
: A tuple of permission guards that run sequentially before execution.

`concurrency`
: Optional semaphore limit. When set, at most `concurrency` invocations
  of this tool can run simultaneously.

### Input schema

The `input_schema` property returns the Pydantic-generated JSON schema:

```python
@property
def input_schema(self) -> dict[str, Any]:
    return self.handler.model_json_schema()
```

Transports send this schema to the LLM so it knows how to call the tool.

## Execution flow

```{mermaid}
sequenceDiagram
    participant Agent
    participant Tool
    participant Guard
    participant Handler

    Agent->>Tool: __call__(**kwargs)
    Tool->>Tool: Acquire semaphore (if concurrency set)
    Tool->>Tool: handler.model_validate(kwargs)
    loop For each guard
        Tool->>Guard: check(handler_instance)
        Guard-->>Tool: handler (or raise GuardError)
    end
    Tool->>Handler: await handler_instance(context)
    Handler-->>Tool: result string
    Tool-->>Agent: result
```

1. The agent calls `tool(**kwargs)` with the input the model provided.
2. If the tool has a concurrency limit, it acquires the semaphore.
3. The kwargs are validated by creating a handler instance via Pydantic's
   `model_validate`.
4. Each guard in the `guards` tuple is called sequentially. A guard can
   modify the handler instance or raise `GuardError` to deny execution.
5. The handler's `__call__` method runs and returns a string result.
6. If the handler raises any exception, it is wrapped in `HandlerError`.

## Exception hierarchy

```
AxioError
└── ToolError
    ├── GuardError    # Guard denied or crashed
    └── HandlerError  # Handler raised during execution
```

The agent catches both and wraps the error message in a `ToolResultBlock`
with `is_error=True`, so the model can see what went wrong and retry or
adjust its approach.

## ToolSelector

The `ToolSelector` protocol (defined in `axio.selector`) allows a component
to filter or rank the full set of available tools before each LLM call. The
agent passes the current message history and the full tool list to the
selector, which returns the subset of tools that should be offered to the
model for that turn.

<!--
name: test_tool_selector_protocol
```python
from dataclasses import dataclass
from axio.tool import ToolHandler
from axio.permission import PermissionGuard
ToolName = str
```
-->
<!-- name: test_tool_selector_protocol -->
```python
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable
from axio.messages import Message
from axio.tool import Tool


@runtime_checkable
class ToolSelector(Protocol):
    async def select(
        self,
        messages: Iterable[Message],
        tools: Iterable[Tool[Any]],
    ) -> Iterable[Tool[Any]]: ...
```

A selector is useful when you have a large tool catalogue and want to avoid
sending every tool's schema to the model on every turn — for example, by
using embeddings or keyword matching to pick only the relevant tools.

`ToolSelector` implementations are registered via the `axio.selector` entry
point group and discovered by `discover_selectors()` from `axio_tui.plugin`.
