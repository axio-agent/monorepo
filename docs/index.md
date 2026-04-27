# Axio

[![GitHub](https://img.shields.io/badge/github-mosquito%2Faxio--agent-181717?logo=github&logoColor=white)](https://github.com/mosquito/axio-agent)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/mosquito/axio-agent/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Docs](https://img.shields.io/github/actions/workflow/status/mosquito/axio-agent/docs.yml?label=docs&logo=readthedocs&logoColor=white)](https://github.com/mosquito/axio-agent/actions)
[![PyPI axio](https://img.shields.io/pypi/v/axio?label=axio&logo=pypi&logoColor=white)](https://pypi.org/project/axio/)
[![PyPI axio-tui](https://img.shields.io/pypi/v/axio-tui?label=axio-tui&logo=pypi&logoColor=white)](https://pypi.org/project/axio-tui/)

**Axio** (*Asynchronous eXtensible Intelligent Orchestration*) — a minimal but complete
foundation for building LLM-powered agents in Python.

Every integration point is a protocol — bring your own transport, context store,
tools, and permission guards.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Quick Start
:link: quick-start
:link-type: doc

Install the TUI and start chatting with an LLM agent in under a minute.
:::

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Write a minimal agent from scratch with the core library.
:::

:::{grid-item-card} Core Concepts
:link: concepts/index
:link-type: doc

Understand the agent loop, protocols, tools, events, and the plugin system.
:::

:::{grid-item-card} How-To Guides
:link: guides/index
:link-type: doc

Step-by-step guides for writing custom tools, transports, and guards.
:::

:::{grid-item-card} Packages
:link: packages
:link-type: doc

Overview of every package in the monorepo and their entry points.
:::

::::

## How to draw an owl

::::{grid} 1 1 2 2
:gutter: 2
:class-row: owl-row

:::{grid-item}
:columns: 12 12 2 2
:class: owl-caption

```{image} _static/axio-circles.svg
:alt: Step 1
:width: 80px
:align: center
```

**Step 1.**
:::

:::{grid-item}
:columns: 12 12 10 10

<!-- name: test_index_example -->
```python
import aiohttp
from typing import Any
from axio.tool import Tool, ToolHandler


class Fetch(ToolHandler[Any]):
    """Fetch the text content of a URL."""
    url: str

    async def __call__(self, context: Any) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as r:
                return (await r.text())[:2000]

fetch = Tool(name="fetch", description=Fetch.__doc__, handler=Fetch)
assert fetch.name == "fetch"
```
:::

::::

::::{grid} 1 1 2 2
:gutter: 2
:class-row: owl-row

:::{grid-item}
:columns: 12 12 2 2
:class: owl-caption

```{image} _static/logo.svg
:alt: Step 2
:width: 80px
:align: center
```

**Step 2.**
:::

:::{grid-item}
:columns: 12 12 10 10

<!--
name: test_index_example
```python
import axio_transport_openai
from axio.testing import StubTransport, make_text_response
axio_transport_openai.OpenAITransport = lambda: StubTransport([make_text_response("The weather is sunny.")])
```
-->
<!-- name: test_index_example -->
```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport


async def main() -> None:
    agent = Agent(
        system="You are a helpful assistant.",
        tools=[fetch],
        transport=OpenAITransport(),
    )
    reply = await agent.run(
        "What's the weather tomorrow? Use geoip for detect my location and wttr.in for weather.",
        MemoryContextStore(),
    )
    assert reply

asyncio.run(main())
```
:::

::::

## Why Axio?

The name is a backronym — each letter describes what the framework actually does:

**A — Asynchronous.**
The agent loop is built on `asyncio` end-to-end. Tool calls from a single
LLM response are dispatched concurrently via `asyncio.gather` so results
arrive in parallel, not sequentially. Every transport, tool, and context
store uses `async def` throughout — no thread pools or blocking I/O hidden
beneath the surface.

**X — eXtensible.**
Every integration point is a runtime-checkable `Protocol` or abstract base class.
You can swap the transport (OpenAI, Anthropic, any custom endpoint), the context
store (in-memory, SQLite, your own database), the permission guards, and the tools
without touching a single line of framework code. The plugin system — based on
Python entry points — lets separate packages register transports, tools, and
guards that are discovered automatically at runtime. Extensible is capitalised
because it is the core design decision from which everything else follows.

**I — Intelligent.**
The LLM drives the decision loop. Axio stays out of the way: it presents tools
to the model, faithfully delivers tool results back, and keeps iterating until the
model decides it is done. No hard-coded routing, no fixed decision trees — the
intelligence lives in the model, not the framework.

**O — Orchestration.**
Axio coordinates agents, tools, context, and permissions into a coherent execution
flow. Sub-agents can be spawned and composed via the built-in `subagent` tool;
context stores are shared across agents; permission guards form a composable
chain that gates every tool call. Complex multi-agent workflows emerge from
simple, well-defined primitives.

Extensible by design
: Every integration point is a runtime-checkable `Protocol` or ABC.
  Swap transports, context stores, tools, and guards without touching framework code.

Streaming by default
: All LLM I/O flows through typed `StreamEvent` values.
  No hidden buffering — you see every token, tool call, and result as it happens.

Tools are Pydantic models
: Define parameters as fields, get JSON schema for free, override `__call__` for execution.
  Guards gate every tool call through a composable permission chain.

Multi-agent orchestration built-in
: Spawn sub-agents via the `subagent` tool, share context between agents,
  compose complex workflows — all without external dependencies.

### Architecture

```{mermaid}
flowchart TB
    subgraph User["User Code"]
        A[Agent]
    end

    subgraph Core["axio - Core Framework"]
        B[Tool Handler<br/>Pydantic model]
        C[Permission Guard<br/>Protocol]
        D[StreamEvent<br/>Typed events]
    end

    subgraph Transport["Transport (pluggable)"]
        E[OpenAI Transport]
        F[Anthropic Transport]
        G[Custom Transport]
    end

    subgraph Context["Context Store (pluggable)"]
        H[MemoryContextStore]
        I[SQLiteContextStore]
    end

    subgraph LLM["LLM Provider"]
        J[OpenAI]
        K[Claude]
        L[Custom]
    end

    A -->|1. configures| B
    A -->|2. uses| C
    A -->|3. builds| D
    A -->|4. sends to| E
    A -->|5. stores in| H
    E -->|SSE| D
    D -->|tool call| B
    B -->|result| D
    D -->|text| J
    J -->|response| E
    E -->|reply| A
```

The agent loop:
1. You configure the agent with tools (Pydantic models) and guards (permission chain)
2. User sends a message
3. Agent sends to transport → LLM
4. LLM responds with text or tool calls
5. Tools execute → results return → LLM generates final response
6. Events stream back to you (tokens, tool calls, results)

## How does Axio compare?

Here's how Axio compares to other popular Python agent frameworks:

| | Axio | pydantic-ai | LangChain / LangGraph | AutoGen |
|---|---|---|---|---|
| **Architecture** | Minimal core + protocols | Pydantic-native, validation-centric | Heavy abstraction layer | Multi-agent orchestration |
| **Streaming** | All events typed, full tool visibility | Text streaming works; tool calls and final answer can't stream simultaneously | Added later, inconsistent | Limited |
| **Tool definition** | Pydantic model subclass | Decorator + function signature → auto JSON schema | Functions + decorators | Class-based agents |
| **Transport** | Pluggable protocol; you bring the client | Built-in 20+ providers, trivial to swap | Built-in, harder to swap | Azure OpenAI focused |
| **Multi-agent** | Built-in (`subagent` tool, shared context stores) | Agent-as-tool pattern; not the primary focus | Via LangGraph | Native |
| **Learning curve** | Low — ~100 lines for an agent | Low for Pydantic/FastAPI users; moderate otherwise | Medium — many abstractions | High — complex configs |
| **Scope** | Agent loop + extensions | Agent + structured outputs; no RAG, no built-in memory | Full stack (RAG, chains, etc.) | Multi-agent scenarios |
| **API stability** | Stable | Beta (v0.x, breaking changes possible) | Stable | Stable |

### When to choose each

**Choose Axio if:**
- You want a minimal foundation and full control over integrations
- Full, real-time streaming visibility into every token, tool call, and result matters
- You prefer explicit protocols over decorator magic
- You're building a custom agent UI or embedding agents in a larger system

**Choose pydantic-ai if:**
- You already use Pydantic/FastAPI and want the same patterns for agents
- Your agent must return strongly typed, validated structured outputs
- You need trivial provider swapping across 20+ LLM backends
- You're okay with beta-stage API stability

**Choose LangChain if:**
- You need RAG, text splitters, and other built-in utilities
- You want batteries-included with less wiring code
- You're prototyping quickly and can accept abstraction overhead

**Choose AutoGen if:**
- You're building complex multi-agent scenarios with conversation flows
- You need built-in support for human-in-the-loop
- You're okay with Azure OpenAI as the primary backend

**Actually — Axio also supports multi-agent:**
- Sub-agents via the built-in `subagent` tool
- Shared context stores for agent-to-agent communication
- Composable workflows without external dependencies

Axio's philosophy is thin abstraction over the prompt-completion loop,
not a full framework with opinions about how you should structure your application.
If that aligns with your needs — welcome.

```{toctree}
:maxdepth: 2
:hidden:

quick-start
getting-started
concepts/index
guides/index
packages
api
troubleshooting
glossary
```
