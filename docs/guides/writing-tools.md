# Writing Tools

This guide walks through creating a custom tool from scratch and registering
it as a plugin.

## 1. Create the handler

A tool handler is a plain `async def` function. Parameters become the tool's
input parameters; the docstring becomes the description.

<!-- name: test_word_count_tool -->
```python
# my_tools/word_count.py
from axio import Tool


async def word_count(text: str) -> str:
    """Count the number of words in the given text."""
    count = len(text.split())
    return f"The text contains {count} words."
```

Key points:

- The **docstring** becomes the tool description sent to the LLM.
- Parameters support all standard Python type annotations. Use `Annotated` +
  `Field` from `axio.field` for descriptions, defaults, or numeric bounds.
- The function must be `async`. It can return a `str`, a `dict`, or any
  JSON-serialisable value. The agent coerces non-string return values to
  JSON when building the `ToolResultBlock`.

## Annotating parameters

Use `Annotated` together with `Field` from `axio.field` to attach metadata
to individual parameters. This controls what the LLM sees in the generated
JSON schema: descriptions, optional defaults, and numeric constraints.

### Descriptions and optional parameters

Parameter descriptions are included in the JSON schema sent to the LLM with
every tool call. Clear descriptions help the model understand what each
parameter expects and produce correct values - especially for parameters
whose purpose isn't obvious from the name alone.

<!-- name: test_annotated_parameters -->
```python
from typing import Annotated
from axio import Field, Tool


async def search(
    query: Annotated[str, Field(description="Search query string")],
    limit: Annotated[int, Field(description="Maximum results to return", default=10)],
) -> str:
    """Search for items matching the query."""
    return f"Found results for '{query}' (limit={limit})"


tool = Tool(name="search", handler=search)
schema = tool.input_schema

assert schema["properties"]["query"]["description"] == "Search query string"
assert schema["properties"]["limit"]["description"] == "Maximum results to return"
# 'query' is required; 'limit' has a default so it is optional
assert "query" in schema["required"]
assert "limit" not in schema.get("required", [])
```

Parameters with a `default` value are omitted from `required` in the schema.
When the LLM omits an optional parameter, the default is applied automatically
before the handler is called - no `None` check needed.

### Numeric constraints

Use `ge` (≥) and `le` (≤) to add bounds that are included in the JSON schema
and enforced at call time:

<!-- name: test_annotated_constraints -->
```python
from typing import Annotated
from axio import Field, Tool


async def resize(
    width: Annotated[int, Field(description="Width in pixels", ge=1, le=4096)],
    height: Annotated[int, Field(description="Height in pixels", ge=1, le=4096)],
) -> str:
    """Resize an image."""
    return f"Resized to {width}x{height}"


tool = Tool(name="resize", handler=resize)
schema = tool.input_schema

assert schema["properties"]["width"]["minimum"] == 1
assert schema["properties"]["width"]["maximum"] == 4096
```

### Strict string parameters

`StrictStr` rejects values that are not already a `str` (no silent coercion
from `int` or other types). Import it from `axio.field`:

<!-- name: test_strict_str -->
```python
from axio import StrictStr, Tool


async def echo(message: StrictStr) -> str:
    """Echo the message back."""
    return message


tool = Tool(name="echo", handler=echo)
schema = tool.input_schema

assert schema["properties"]["message"]["type"] == "string"
```

`StrictStr` is equivalent to `Annotated[str, FieldInfo(strict=True)]`. The LLM
always sends strings, so `StrictStr` is mainly useful when you call a tool from
Python code and want to catch accidental non-string inputs early.

## 2. Wrap it in a Tool

<!-- name: test_word_count_tool -->
```python
from axio import Tool

word_count_tool = Tool(
    name="word_count",
    handler=word_count,
)
```

`Tool` reads the description from `handler.__doc__` automatically.
Pass an explicit `description=` string to override it.

## 3. Use it with an agent

<!--
name: test_word_count_tool
```python
from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response
my_transport = StubTransport([make_text_response("ok")])
context = MemoryContextStore()
```
-->
<!-- name: test_word_count_tool -->
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
word_count = "my_tools.word_count:word_count"
```

After installing or syncing, `discover_tools()` will find it automatically.

## Adding guards

Attach guards to control when the tool can run:

<!--
name: test_tool_with_guard
```python
from axio import Tool

async def word_count(text: str) -> str:
    """Count words."""
    return str(len(text.split()))
```
-->
<!-- name: test_tool_with_guard -->
```python
from axio.permission import AllowAllGuard

tool = Tool(
    name="word_count",
    handler=word_count,
    guards=(AllowAllGuard(),),
)
```

See [Guards](../concepts/guards.md) for more on the guard system.

## Concurrency control

Limit how many instances of your tool can run simultaneously:

```python
async def web_fetch(url: str) -> str:
    """Fetch a URL."""
    ...

tool = Tool(
    name="web_fetch",
    handler=web_fetch,
    concurrency=3,  # at most 3 concurrent fetches
)
```

## Error handling

If your handler raises an exception, Axio wraps it in `HandlerError` and
sends the error message back to the model as a `ToolResultBlock` with
`is_error=True`. The model sees the error and can adjust its approach.

For expected failures, raise `HandlerError` directly with a clear message:

<!--
name: test_error_handling
```python
from pathlib import Path
```
-->
<!-- name: test_error_handling -->
```python
from axio import HandlerError, Tool


async def read_file(path: str) -> str:
    """Read a file."""
    p = Path(path)
    if not p.exists():
        raise HandlerError(f"File not found: {path}")
    return p.read_text()
```

## Dynamic tool providers

If your package needs to provide a variable number of tools based on
configuration (like MCP servers or Docker containers), implement the
`ToolsPlugin` protocol instead. `ToolsPlugin` is defined in the
`axio_tui` package:

```python
from typing import Any
from axio_tui.plugin import ToolsPlugin
from axio import Tool


class MyPlugin:
    """A dynamic tool provider."""

    @property
    def label(self) -> str:
        return "My Plugin"

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        # Load configuration, connect to external services, etc.
        pass

    @property
    def all_tools(self) -> list[Tool]:
        # Build and return tools based on current configuration
        return [...]

    def settings_screen(self) -> Any:
        # Return a Textual Screen for configuring this plugin in the TUI
        return None

    async def close(self) -> None:
        # Clean up connections and resources
        pass
```

Register under `axio.tools.settings`:

```toml
[project.entry-points."axio.tools.settings"]
my_plugin = "my_package.plugin:MyPlugin"
```

See [Plugin System](../concepts/plugins.md) for the full `ToolsPlugin` protocol
and lifecycle documentation.
