# Tool System

Tools are plain `async def` functions. Parameters are function arguments;
the docstring becomes the tool description sent to the LLM.

## Plain async function

<!-- name: test_write_file_handler -->
```python
from pathlib import Path
from axio import Tool

async def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    Path(path).write_text(content)
    return f"Wrote {len(content)} bytes to {path}"

tool = Tool(name="write_file", handler=write_file)
```

The handler's **docstring** becomes the tool description sent to the LLM.
Function annotations are converted to a JSON schema object automatically - no
decorators or schema registration needed.

Use `Annotated` + `Field` to add descriptions, defaults, or numeric bounds:

<!-- name: test_write_file_with_field -->
```python
from typing import Annotated
from axio import Tool, Field

async def search(
    query: Annotated[str, Field(description="Search query")],
    limit: Annotated[int, Field(default=10, ge=1, le=100)] = 10,
) -> str:
    """Search the knowledge base."""
    return f"results for {query!r} (limit={limit})"

tool = Tool(name="search", handler=search)
result_default = search.__defaults__
assert result_default == (10,)
```

## Context injection

When a tool needs access to runtime state (a database connection, a sandbox
object, etc.), use `CONTEXT.get()` inside the function and pass the value
via `Tool(context=...)`:

<!-- name: test_tool_decorator -->

```python
import asyncio
from typing import Annotated
from axio import Tool, CONTEXT, Field


async def search(
        query: Annotated[str, Field(description="Search query")],
        limit: Annotated[int, Field(default=10, ge=1, le=100)] = 10,
) -> str:
    """Search a list of documents."""
    documents: list[str] = CONTEXT.get()
    results = [s for s in documents if query.lower() in s.lower()]
    return "\n".join(results[:limit]) or "no results"


documents = ["Axio is async", "Pydantic is great", "Axio uses protocols"]
t = Tool(name="search", handler=search, context=documents)

result = asyncio.run(t(query="axio"))
assert "Axio" in result
```

Nested helpers that cannot receive arguments can also call `CONTEXT.get()`:

<!-- name: test_tool_context_var -->

```python
import asyncio
from axio import Tool, CONTEXT


def helper() -> str:
    return str(CONTEXT.get())  # works even without an explicit argument


async def ping(msg: str) -> str:
    """Echo msg with context from ContextVar."""
    return f"{msg}:{helper()}"


t = Tool(name="ping", handler=ping, context="ctx-42")
assert asyncio.run(t(msg="hello")) == "hello:ctx-42"
```

## Tool dataclass

Every handler function is wrapped in a `Tool` frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: str
    handler: Callable[..., Awaitable[Any]]
    description: str = ""       # defaults to handler.__doc__
    guards: tuple[PermissionGuard, ...] = ()
    concurrency: int | None = None
    context: T = ...   # default: empty mapping
```

`handler`
: An `async def` function.  A fresh call is made per invocation with
  validated kwargs.

`description`
: Defaults to `handler.__doc__`.  Pass an explicit string to override.

`guards`
: Guards run sequentially before the handler.  Each receives the `Tool` object
  and the raw kwargs, and either returns a (possibly modified) kwargs dict
  (allow) or raises `GuardError` (deny).

`concurrency`
: Limits parallel invocations of this tool via an `asyncio.Semaphore`.

`context`
: Arbitrary runtime state available via `CONTEXT.get()` inside the handler.
  Use this to inject a database connection, a sandbox object, or any other
  state the handler needs without touching global state.

### Input schema

```python
@property
def input_schema(self) -> dict[str, Any]:
    return dict(self.schema)
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
    Tool->>Tool: Acquire semaphore (if set)
    Tool->>Tool: Inject defaults + validate types/bounds
    loop For each guard
        Tool->>Guard: check(tool, **kwargs)
        Guard-->>Tool: kwargs (or raise GuardError)
    end
    Tool->>Handler: handler(**kwargs)
    Handler-->>Tool: result string
    Tool-->>Agent: result
```

1. The agent calls `tool(**kwargs)` with the input the model provided.
2. If the tool has a concurrency limit, it acquires the semaphore.
3. Missing fields with defaults are injected; provided fields are validated (type, Literal, bounds).
4. Each guard in the `guards` tuple is called sequentially with the fully materialised kwargs.
5. Guards return a (possibly modified) kwargs dict to allow, or raise `GuardError` to deny.
6. The handler is called with the materialised kwargs (stray keys stripped unless handler accepts `**kwargs`).
7. Any exception from the handler is wrapped in `HandlerError`.

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

The `ToolSelector` protocol allows a component to filter or rank the full set
of available tools before each LLM call.

<!--
name: test_tool_selector_protocol
```python
from axio import PermissionGuard
ToolName = str
```
-->
<!-- name: test_tool_selector_protocol -->
```python
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable
from axio.messages import Message
from axio import Tool


@runtime_checkable
class ToolSelector(Protocol):
    async def select(
        self,
        messages: Iterable[Message],
        tools: Iterable[Tool[Any]],
    ) -> Iterable[Tool[Any]]: ...
```

A selector is useful when you have a large tool catalogue and want to avoid
sending every tool's schema to the model on every turn - for example, by
using embeddings or keyword matching to pick only the relevant tools.

`ToolSelector` implementations are registered via the `axio.selector` entry
point group and discovered by `discover_selectors()` from `axio_tui.plugin`.
