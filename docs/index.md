# Axio

[![GitHub org](https://img.shields.io/badge/github-axio--agent-181717?logo=github&logoColor=white)](https://github.com/axio-agent)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/axio-agent/axio/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Docs](https://img.shields.io/github/actions/workflow/status/axio-agent/docs/docs.yml?label=docs&logo=readthedocs&logoColor=white)](https://github.com/axio-agent/docs/actions)
[![PyPI axio](https://img.shields.io/pypi/v/axio?label=axio&logo=pypi&logoColor=white)](https://pypi.org/project/axio/)
[![PyPI axio-tui](https://img.shields.io/pypi/v/axio-tui?label=axio-tui&logo=pypi&logoColor=white)](https://pypi.org/project/axio-tui/)

**A highly extensible, streaming-first agent framework for Python.**

Axio gives you a minimal but complete foundation for building LLM-powered agents.
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

```python
import aiohttp
from axio import Tool, ToolHandler

class Fetch(ToolHandler):
    """Fetch the text content of a URL."""
    url: str

    async def __call__(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as r:
                return (await r.text())[:2000]

fetch = Tool(name="fetch", description=Fetch.__doc__, handler=Fetch)
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

```python
import asyncio
from axio import Agent, MemoryContextStore
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
    print(reply)

asyncio.run(main())
```
:::

::::

## Why Axio?

Extensible by design
: Every integration point is a runtime-checkable `Protocol` or ABC.
  Swap transports, context stores, tools, and guards without touching framework code.

Streaming by default
: All LLM I/O flows through typed `StreamEvent` values.
  No hidden buffering — you see every token, tool call, and result as it happens.

Tools are Pydantic models
: Define parameters as fields, get JSON schema for free, override `__call__` for execution.
  Guards gate every tool call through a composable permission chain.

```{toctree}
:maxdepth: 2
:hidden:

quick-start
getting-started
concepts/index
guides/index
packages
```
