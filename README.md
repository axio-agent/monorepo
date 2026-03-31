# axio-transport-openai

[![PyPI](https://img.shields.io/pypi/v/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

OpenAI-compatible streaming transport for [axio](https://github.com/axio-agent/axio).

Works with any API that speaks the OpenAI chat completions format — OpenAI itself, Azure OpenAI, local models via Ollama/vLLM/LM Studio, and compatible cloud providers.

## Features

- **Full SSE streaming** — parses `data:` chunks incrementally; no waiting for full responses
- **Automatic retry** — configurable backoff on transient HTTP errors
- **Tool calling** — streams tool-use JSON fragments as `ToolInputDelta` events
- **aiohttp-based** — zero blocking I/O
- **Optional TUI settings screen** — install with `[tui]` extra for a Textual configuration UI

## Installation

```bash
pip install axio-transport-openai
```

With TUI settings screen:

```bash
pip install "axio-transport-openai[tui]"
```

## Usage

```python
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport

transport = OpenAITransport(
    api_key="sk-...",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",  # default; override for local models
    max_tokens=4096,
)

agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)

async def main() -> None:
    ctx = MemoryContextStore()
    result = await agent.run("What is 2 + 2?", ctx)
    print(result)
```

### Local models (Ollama, vLLM, LM Studio)

```python
transport = OpenAITransport(
    api_key="ollama",                        # any non-empty string
    model="llama3.2",
    base_url="http://localhost:11434/v1",
)
```

### Streaming events

```python
from axio.events import TextDelta, SessionEndEvent

async for event in agent.run_stream("Tell me a joke", ctx):
    match event:
        case TextDelta(delta=text):
            print(text, end="", flush=True)
        case SessionEndEvent(total_usage=usage):
            print(f"\n[{usage.input_tokens}in / {usage.output_tokens}out tokens]")
```

## Plugin registration

When installed, this package registers itself via entry points so `axio-tui` discovers it automatically:

```toml
[project.entry-points."axio.transport"]
openai = "axio_transport_openai:OpenAITransport"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-transport-nebius](https://github.com/axio-agent/axio-transport-nebius) · [axio-transport-codex](https://github.com/axio-agent/axio-transport-codex) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
