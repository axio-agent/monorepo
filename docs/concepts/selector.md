# Tool Selection

Tool selection is the process of dynamically filtering the set of available tools before each LLM call. By implementing the `ToolSelector` protocol, you can reduce context size, enforce capability restrictions, or route tool access based on conversation context.

## Why use tool selection?

Sending every tool definition to the LLM on every iteration has downsides:

- **Token cost**: Tool schemas consume input tokens even when irrelevant
- **Context pollution**: Large tool sets can confuse the model
- **Security**: Some users or sessions may only be authorized for specific tools
- **Performance**: Parsing and serializing many tool schemas adds latency

A `ToolSelector` addresses these by trimming the active tool list before each iteration passes to the transport.

## ToolSelector protocol

The `ToolSelector` protocol defines a single method:

<!--
name: test_tool_selector_protocol
-->
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

**Parameters:**

`messages`
: The current conversation history from the {class}`ContextStore`. Inspect this to understand what the user has asked and what tools have been used.

`tools`
: The full set of registered tools from the agent's `tools` list.

**Returns:**

An `Iterable[Tool[Any]]` containing the subset of tools to pass to the transport for this iteration. The order does not matter - only membership.

When `selector` is `None` (the default), all tools are passed on every iteration.

## When to use selectors

Use a `ToolSelector` when:

- You have **10+ tools** and want to reduce noise in the model's context
- You need **dynamic access control** based on user role or session state
- You want to implement **intent-based routing** (e.g., only file tools for file-related questions)
- You're building a **plugin system** where tool availability changes at runtime

For small, fixed tool sets, a selector is usually unnecessary.

## Example implementations

### Keyword-based filtering

Filter tools based on whether the user's latest message mentions relevant keywords:

<!-- name: test_keyword_selector -->
```python
import asyncio
from collections.abc import Iterable
from typing import Any
from axio.messages import Message
from axio import Tool


class KeywordSelector:
    """Select tools based on keyword matching in the latest user message."""

    def __init__(self, keywords: dict[str, list[str]]):
        """
        keywords: dict mapping tool_name -> list of triggering keywords
        """
        self.keywords = {name.lower(): kwlist for name, kwlist in keywords.items()}

    async def select(
        self,
        messages: Iterable[Message],
        tools: Iterable[Tool[Any]],
    ) -> Iterable[Tool[Any]]:
        # Get the latest user message
        message_list = list(messages)
        if not message_list:
            return []  # No messages, no tools

        latest = message_list[-1]
        if latest.role != "user":
            return tools  # Default to all if not a user message

        text = "".join(
            block.text for block in latest.content
            if hasattr(block, "text")
        ).lower()

        # Match any keyword
        selected: list[Tool[Any]] = []
        for tool in tools:
            tool_keywords = self.keywords.get(tool.name.lower(), [])
            if any(kw.lower() in text for kw in tool_keywords):
                selected.append(tool)

        # If no match, return empty to avoid tool calls
        return selected if selected else []


# Usage
selector = KeywordSelector({
    "write_file": ["file", "write", "save", "create"],
    "read_file": ["file", "read", "open", "load"],
    "search": ["search", "find", "query"],
})
```

### Permission-based filtering

Combine with guards to enforce role-based access:

<!-- name: test_permission_selector -->
```python
from collections.abc import Iterable
from typing import Any
from axio.messages import Message
from axio import Tool


class RoleBasedSelector:
    """Select tools based on user role."""

    ALLOWED_TOOLS: dict[str, set[str]] = {
        "admin": {"write_file", "delete_file", "shell"},
        "editor": {"write_file", "read_file"},
        "viewer": {"read_file"},
    }

    def __init__(self, role: str):
        self.role = role.lower()

    async def select(
        self,
        messages: Iterable[Message],
        tools: Iterable[Tool[Any]],
    ) -> Iterable[Tool[Any]]:
        allowed_names = self.ALLOWED_TOOLS.get(self.role, set())
        return [t for t in tools if t.name in allowed_names]
```

### Embedding-based selection

For large tool catalogues, use semantic similarity:

<!-- name: test_embedding_selector -->
```python
from collections.abc import Iterable
from typing import Any
from axio.messages import Message
from axio import Tool


class EmbeddingSelector:
    """Select tools using embedding similarity to the query."""

    def __init__(self, model: str = "text-embedding-3-small", top_k: int = 5):
        self.model = model
        self.top_k = top_k
        self._tool.embeddings: dict[str, list[float]] = {}

    async def _embed(self, text: str) -> list[float]:
        # Placeholder - use your embedding provider here
        return [0.0] * 1536

    async def select(
        self,
        messages: Iterable[Message],
        tools: Iterable[Tool[Any]],
    ) -> Iterable[Tool[Any]]:
        message_list = list(messages)
        if not message_list:
            return []

        query = message_list[-1].content[0].text if message_list[-1].content else ""

        # Build embeddings for tools not yet cached
        for tool in tools:
            if tool.name not in self._tool.embeddings:
                desc = f"{tool.name}: {tool.description}"
                self._tool.embeddings[tool.name] = await self._embed(desc)

        query_embedding = await self._embed(query)
        # ... similarity scoring logic ...
        # return top_k tools
        return list(tools)[: self.top_k]
```

## Integration with Agent

Pass a selector via the `selector` field when constructing an {class}`Agent`:
```python
import asyncio
from axio import Agent, Tool, MemoryContextStore
from axio.testing import StubTransport, make_text_response


async def echo(msg: str) -> str:
    """Echo the message."""
    return msg


async def _run():
    selector = KeywordSelector({"echo": ["echo", "repeat", "say"]})
    transport = StubTransport([make_text_response("ok")])
    agent = Agent(
        system="You are helpful.",
        transport=transport,
        tools=[Tool(name="echo", handler=echo)],
        selector=selector,
    )

    context = MemoryContextStore()
    result = await agent.run("echo hello", context)
    return result

asyncio.run(_run())
```

## Plugin discovery

`ToolSelector` implementations can be registered via entry points in `pyproject.toml`:

```toml
[project.entry-points."axio.selector"]
keyword = "my_package.selectors:KeywordSelector"
role_based = "my_package.selectors:RoleBasedSelector"
```

Then discovered at runtime via `axio_tui.plugin.discover_selectors()`.

## Related

- {doc}`agent` - The `Agent` class uses the selector to filter tools each iteration
- {doc}`tools` - Tool definitions and the handler function signature
- {doc}`guards` - Permission guards run *after* selection, at call time
