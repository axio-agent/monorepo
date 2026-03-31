# Getting Started

## Installation

Axio is distributed as a uv workspace. Clone the repository and sync dependencies:

```bash
git clone https://github.com/axio-agent/axio.git
cd axio
uv sync
```

To include optional packages (transports, tools, guards):

```bash
uv sync --all-extras
```

## Minimal Agent

The smallest possible agent needs three things: a **transport** to talk to an LLM,
a **context store** to hold conversation history, and an **Agent** to tie them together.

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
    print(reply)

asyncio.run(main())
```

Replace `StubTransport` with a real transport like `OpenAITransport` to connect to
a live LLM. The agent loop, tool dispatch, and streaming all work the same way
regardless of which transport you use — that's the power of the protocol-driven
design.

## Adding Tools

Tools are Pydantic models. Define fields for parameters and implement `__call__`:

```python
from axio import Tool, ToolHandler

class Greet(ToolHandler):
    """Greet someone by name."""
    name: str

    async def __call__(self) -> str:
        return f"Hello, {self.name}!"

agent = Agent(
    system="You are a helpful assistant.",
    tools=[Tool(name="greet", description="Greet someone", handler=Greet)],
    transport=transport,
)
```

## Running the TUI

The `axio-tui` package provides a terminal UI built with Textual:

```bash
uv tool install "axio-tui[all]"
uv tool run axio
```

```{image} _static/tui-screenshot.svg
:alt: Axio TUI — terminal interface showing a conversation with tool calls
:width: 100%
```

The TUI discovers available transports, tools, and guards automatically
via the [plugin system](concepts/plugins.md).

## Next Steps

- Read [Core Concepts](concepts/index.md) to understand the architecture
- Follow the [Writing Tools](guides/writing-tools.md) guide to create your own
- See [Packages](packages.md) for an overview of all available packages
