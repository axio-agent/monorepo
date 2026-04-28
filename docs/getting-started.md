# Getting Started

**Axio** (*Asynchronous eXtensible Intelligent Orchestration*) gives you a minimal
but complete foundation for building LLM-powered agents - see [Why Axio?](index.md#why-axio)
for the story behind the name.

## Installation

Install Axio from PyPI:

```bash
pip install axio
```

To install the terminal UI with extras:

```bash
pip install "axio-tui[all]"
# or with uv (global tool install):
uv tool install "axio-tui[anthropic,openai,codex,local,mcp,guards]"
```

Available TUI extras: `anthropic`, `openai`, `codex`, `local`, `mcp`, `guards`, `all`.

### From source (development)

If you want to work on Axio itself, clone the monorepo and sync dependencies:

```bash
git clone https://github.com/mosquito/axio-agent
cd axio-agent
uv sync --all-packages
```

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - for workspace management and tool installation
## Minimal Agent

The smallest possible agent needs three things: a **transport** to talk to an LLM,
a **context store** to hold conversation history, and an **Agent** to tie them together.

<!-- name: test_minimal_agent -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio.testing import StubTransport, make_text_response

async def main() -> None:
    transport = StubTransport([
        make_text_response("Hello! I'm a stub agent."),
    ])
    context = MemoryContextStore()
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[],
        transport=transport,
    )
    reply = await agent.run("Hi there!", context)
    return reply

assert asyncio.run(main()) == "Hello! I'm a stub agent."
```

Replace `StubTransport` with real transport like `OpenAITransport` to connect to
a live LLM. The agent loop, tool dispatch, and streaming all work the same way
regardless of which transport you use - that's the power of the protocol-driven
design.

## Adding Tools

Tools are plain `async def` functions. Define parameters as arguments and return
a string (or JSON-serialisable value):

<!--
name: test_adding_tools
-->
<!-- name: test_adding_tools -->
```python
from axio import Agent, MemoryContextStore, Tool
from axio.testing import StubTransport, make_text_response

# Use real transport in real code; StubTransport is just for example
transport = StubTransport([make_text_response("ok")])
context = MemoryContextStore()


async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


agent = Agent(
    system="You are a helpful assistant.",
    tools=[Tool(name="greet", handler=greet)],
    transport=transport,
)
```

## Streaming Events

`run_stream()` returns an `AgentStream` that yields `StreamEvent` objects as the
agent runs. This lets you react to text tokens, tool calls, and session-end
signals as they arrive rather than waiting for the full response.

<!-- name: test_streaming_example -->
```python
import asyncio
from axio import Agent, MemoryContextStore, TextDelta
from axio.testing import StubTransport, make_text_response
from axio.events import SessionEndEvent

async def main() -> None:
    transport = StubTransport([
        make_text_response("Streaming works!"),
    ])
    context = MemoryContextStore()
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[],
        transport=transport,
    )
    collected = []
    async for event in agent.run_stream("Hello!", context):
        if isinstance(event, TextDelta):
            collected.append(event.delta)
        elif isinstance(event, SessionEndEvent):
            break
    return "".join(collected)

assert asyncio.run(main()) == "Streaming works!"
```

## Running the TUI

The `axio-tui` package provides a terminal UI built with Textual:

```bash
uv tool install "axio-tui[all]"
uv tool run axio
```

```{image} _static/tui-screenshot.svg
:alt: Axio TUI - terminal interface showing a conversation with tool calls
:width: 100%
```

The TUI discovers available transports, tools, and guards automatically
via the [plugin system](concepts/plugins.md).

## Next Steps

- Read [Core Concepts](concepts/index.md) to understand the architecture
- Follow the [Writing Tools](guides/writing-tools.md) guide to create your own
- See [Packages](packages.md) for an overview of all available packages
