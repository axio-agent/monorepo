# axio

[![PyPI](https://img.shields.io/pypi/v/axio)](https://pypi.org/project/axio/)
[![Python](https://img.shields.io/pypi/pyversions/axio)](https://pypi.org/project/axio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Minimal, streaming-first, protocol-driven foundation for LLM-powered agents.

No dependencies. Three protocols. An agent loop that just works.

## Features

- **Streaming agent loop** - `run_stream()` yields typed events as they arrive; no buffering, no polling
- **Three clean protocols** - `CompletionTransport`, `ContextStore`, `PermissionGuard`; swap any piece without touching the rest
- **Concurrent tool dispatch** - all tool calls in a turn run via `asyncio.gather` automatically
- **Context compaction** - `compact_context()` summarises old history to stay within token limits
- **Testing helpers** - `StubTransport`, `make_tool_use_response()`, `make_echo_tool()` ship in `axio.testing`
- **Plugin-ready** - entry-point groups (`axio.tools`, `axio.transport`, `axio.guards`) for drop-in extensions

## Installation

```bash
pip install axio
```

## Quick start

<!--
name: test_readme_quick_start
```python
import sys, types
from axio.testing import StubTransport, make_text_response

_m = types.ModuleType("axio_transport_openai")
class OpenAITransport(StubTransport):
    def __init__(self, api_key: str = "", model: str = "") -> None:
        super().__init__([make_text_response("Hello, Alice!")])
_m.OpenAITransport = OpenAITransport  # type: ignore[attr-defined]
sys.modules["axio_transport_openai"] = _m
```
-->
<!-- name: test_readme_quick_start -->
```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.tool import Tool

# 1. Define a tool
async def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"

greet_tool = Tool(name="greet", description="Greet someone by name", handler=greet)

# 2. Wire up the agent (transport comes from an axio-transport-* package)
from axio_transport_openai import OpenAITransport

transport = OpenAITransport(api_key="sk-...", model="gpt-4o-mini")
agent = Agent(system="You are helpful.", transport=transport, tools=[greet_tool])

# 3. Run
async def main() -> None:
    ctx = MemoryContextStore()
    async for event in agent.run_stream("Please greet Alice", ctx):
        print(event)

asyncio.run(main())
```

## Architecture

```
  User message
       │
       ▼
  ┌─────────┐   stream()    ┌─────────────────────┐
  │  Agent  │ ────────────▶ │ CompletionTransport  │
  │  loop   │ ◀──────────── │ (Anthropic, OpenAI, …) │
  └─────────┘  StreamEvent  └─────────────────────┘
       │
       │ tool_use?
       ▼
  ┌──────────┐   check()   ┌─────────────────┐
  │   Tool   │ ──────────▶ │ PermissionGuard │
  │ handler  │             │ (path, LLM, …)  │
  └──────────┘             └─────────────────┘
       │
       ▼
  ┌──────────────┐
  │ ContextStore │  append() / get_history() / fork() / compact()
  └──────────────┘
```

## Protocols

### CompletionTransport

<!-- name: test_readme_completion_transport -->
```python
from typing import Protocol, runtime_checkable
from collections.abc import AsyncIterator
from axio.events import StreamEvent
from axio.messages import Message
from axio.tool import Tool

@runtime_checkable
class CompletionTransport(Protocol):
    def stream(
        self, messages: list[Message], tools: list[Tool], system: str
    ) -> AsyncIterator[StreamEvent]: ...
```

### ContextStore

<!-- name: test_readme_context_store -->
```python
from axio.context import ContextStore
from axio.messages import Message

class MyContextStore(ContextStore):
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def append(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self) -> list[Message]:
        return list(self._messages)

    # Everything else - session_id, close(), fork(), clear(),
    # get/set_context_tokens(), add_context_tokens(), list_sessions()
    # - has a default implementation.
```

### PermissionGuard

`PermissionGuard` is an abstract base class (ABC). Subclass it and implement
`check()`:

<!-- name: test_readme_permission_guard -->
```python
from typing import Any
from axio.permission import PermissionGuard
from axio.tool import Tool

class MyGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        # return kwargs to allow, raise GuardError to deny
        return kwargs
```

## Stream events

| Event | Description |
|---|---|
| `TextDelta` | Incremental assistant text chunk |
| `ToolUseStart` | Tool call begins (name + id) |
| `ToolInputDelta` | Streaming JSON fragment for tool arguments |
| `ToolResult` | Tool execution result |
| `IterationEnd` | One LLM round complete - carries `Usage` + `StopReason` |
| `Error` | Transport or tool exception |
| `SessionEndEvent` | Agent loop finished - carries total `Usage` |

## Tools

<!-- name: test_readme_tools -->
```python
from axio.tool import Tool

async def summarise(text: str, max_words: int = 20) -> str:
    """Summarise the given text in one sentence."""
    # your implementation
    return "..."

tool = Tool(
    name="summarise",
    description="Summarise text",   # overrides docstring if set
    handler=summarise,
    concurrency=4,                   # max parallel executions
)
```

## Testing

<!-- name: test_readme_testing -->
```python
from axio.agent import Agent
from axio.testing import (
    StubTransport,
    make_tool_use_response,
    make_text_response,
    make_ephemeral_context,
    make_echo_tool,
)

async def test_agent_calls_tool():
    transport = StubTransport([
        make_tool_use_response("echo", tool_input={"msg": "hi"}),
        make_text_response("Done"),
    ])
    agent = Agent(system="", tools=[make_echo_tool()], transport=transport)
    result = await agent.run("say hi", make_ephemeral_context())
    assert result == "Done"
```

## Plugin entry points

```toml
[project.entry-points."axio.tools"]
my_tool = "my_package:MyHandler"

[project.entry-points."axio.transport"]
my_backend = "my_package:MyTransport"

[project.entry-points."axio.guards"]
my_guard = "my_package:MyGuard"
```

## Ecosystem

| Package | Purpose |
|---|---|
| [axio-transport-anthropic](https://github.com/mosquito/axio-agent) | Anthropic Claude transport |
| [axio-transport-openai](https://github.com/mosquito/axio-agent) | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) |
| [axio-transport-codex](https://github.com/mosquito/axio-agent) | ChatGPT OAuth transport |
| [axio-context-sqlite](https://github.com/mosquito/axio-agent) | SQLite-backed persistent context store |
| [axio-tools-local](https://github.com/mosquito/axio-agent) | Shell, file, Python tools |
| [axio-tools-mcp](https://github.com/mosquito/axio-agent) | MCP server bridge |
| [axio-tools-docker](https://github.com/mosquito/axio-agent) | Docker sandbox tools |
| [axio-tui](https://github.com/mosquito/axio-agent) | Textual TUI application |
| [axio-tui-guards](https://github.com/mosquito/axio-agent) | Permission guard plugins |

## License

MIT
