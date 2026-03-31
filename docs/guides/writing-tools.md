# Writing Tools

This guide walks through creating a custom tool from scratch and registering
it as a plugin.

## 1. Create the handler

A tool handler is a Pydantic `BaseModel` subclass. Fields become the tool's
input parameters; the `__call__` method implements execution.

```python
# my_tools/word_count.py
from axio import ToolHandler


class WordCount(ToolHandler):
    """Count the number of words in the given text."""

    text: str

    async def __call__(self) -> str:
        count = len(self.text.split())
        return f"The text contains {count} words."
```

Key points:

- The **docstring** becomes the tool description sent to the LLM.
- Fields support all Pydantic features: defaults, validators, `Field()`
  metadata.
- `__call__` must be async and return a string.

## 2. Wrap it in a Tool

```python
from axio import Tool

word_count_tool = Tool(
    name="word_count",
    description="Count words in text",
    handler=WordCount,
)
```

The `handler` parameter takes the **class**, not an instance. Axio creates a
fresh instance for each invocation via `model_validate()`.

## 3. Use it with an agent

```python
agent = Agent(
    system="You are a helpful assistant.",
    tools=[word_count_tool],
    transport=my_transport,
)
```

## 4. Register as a plugin

To make your tool discoverable by the TUI and other Axio applications, add an
entry point to your `pyproject.toml`:

```toml
[project.entry-points."axio.tools"]
word_count = "my_tools.word_count:WordCount"
```

After installing or syncing, `discover_tools()` will find it automatically.

## Adding guards

Attach guards to control when the tool can run:

```python
from axio import AllowAllGuard

tool = Tool(
    name="word_count",
    description="Count words in text",
    handler=WordCount,
    guards=(AllowAllGuard(),),
)
```

See [Guards](../concepts/guards.md) for more on the guard system.

## Concurrency control

Limit how many instances of your tool can run simultaneously:

```python
tool = Tool(
    name="web_fetch",
    description="Fetch a URL",
    handler=WebFetch,
    concurrency=3,  # at most 3 concurrent fetches
)
```

## Error handling

If your handler raises an exception, Axio wraps it in `HandlerError` and
sends the error message back to the model as a `ToolResultBlock` with
`is_error=True`. The model sees the error and can adjust its approach.

For expected failures, raise `HandlerError` directly with a clear message:

```python
from axio.exceptions import HandlerError

class ReadFile(ToolHandler):
    """Read a file."""
    path: str

    async def __call__(self) -> str:
        p = Path(self.path)
        if not p.exists():
            raise HandlerError(f"File not found: {self.path}")
        return p.read_text()
```

## Dynamic tool providers

If your package needs to provide a variable number of tools based on
configuration (like MCP servers or Docker containers), implement the
`ToolsPlugin` protocol instead:

```python
from axio.plugin import ToolsPlugin

class MyPlugin:
    async def get_tools(self) -> list[Tool]:
        # Build tools dynamically
        return [...]
```

Register under `axio.tools.settings`:

```toml
[project.entry-points."axio.tools.settings"]
my_plugin = "my_package.plugin:MyPlugin"
```
