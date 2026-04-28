# Quick Start

Get a working TUI agent running in your terminal in under a minute.

## Install

Install the TUI as an isolated tool with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "axio-tui[all]"
```

Or pick only the extras you need:

```bash
# OpenAI transport (includes Nebius, OpenRouter, custom) + local filesystem tools
uv tool install "axio-tui[openai,local]"

# Anthropic + guards
uv tool install "axio-tui[anthropic,guards]"
```

Available extras: `anthropic`, `openai`, `codex`, `local`, `mcp`,
`guards`, `all`.

:::{dropdown} Alternative: pip install
```bash
pip install "axio-tui[all]"
```
:::

### From source (development)

```bash
git clone https://github.com/mosquito/axio-agent
cd axio-agent
uv sync --all-packages
```

## Set your API key

Export the API key for your chosen transport:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Nebius AI Studio
export NEBIUS_API_KEY="..."
```

## Launch the TUI

```bash
axio
```

```{image} _static/tui-screenshot.svg
:alt: Axio TUI - terminal interface showing a conversation with tool calls
:width: 100%
```

The TUI automatically discovers all installed transports, tools, and guards
via the [plugin system](concepts/plugins.md). Select a model, start a
conversation, and watch the agent call tools in real time.

## Key features

- **Model selection** - switch between any discovered transport and model
- **Session persistence** - conversations are stored in SQLite and survive
  restarts
- **Tool visibility** - every tool call is shown with its input and output
- **Permission guards** - guards prompt for approval before executing
  sensitive operations
- **Sub-agents** - spawn child agents for parallel tasks

## Web mode

Serve the TUI over HTTP for remote access:

```bash
axio --serve
```

Opens on `localhost:8086` by default. Access it from any browser.

## What's next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Write a minimal agent from scratch with the core library.
:::

:::{grid-item-card} Core Concepts
:link: concepts/index
:link-type: doc

Understand protocols, tools, events, and the plugin system.
:::

::::
