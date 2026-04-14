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

<!-- name: test_readme_usage -->
```python
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport, OPENAI_MODELS

transport = OpenAITransport(
    api_key="sk-...",
    model=OPENAI_MODELS["gpt-4o-mini"],
    base_url="https://api.openai.com/v1",  # default; override for local models
)

agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)

async def main() -> None:
    ctx = MemoryContextStore()
    result = await agent.run("What is 2 + 2?", ctx)
    print(result)
```

### Local models (Ollama, vLLM, LM Studio)

<!-- name: test_readme_usage -->
```python
from axio.models import ModelSpec, Capability

transport = OpenAITransport(
    api_key="ollama",                        # any non-empty string
    model=ModelSpec(id="llama3.2", capabilities=frozenset({Capability.text})),
    base_url="http://localhost:11434/v1",
)
```

### Streaming events

<!-- name: test_readme_streaming -->
```python
import asyncio
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.testing import StubTransport, make_text_response
from axio.events import TextDelta, SessionEndEvent

agent = Agent(
    system="",
    tools=[],
    transport=StubTransport([make_text_response("Why did the chicken cross the road?")]),
)

async def main() -> None:
    ctx = MemoryContextStore()
    async for event in agent.run_stream("Tell me a joke", ctx):
        match event:
            case TextDelta(delta=text):
                print(text, end="", flush=True)
            case SessionEndEvent(total_usage=usage):
                print(f"\n[{usage.input_tokens}in / {usage.output_tokens}out tokens]")

asyncio.run(main())
```

## Plugin registration

When installed, this package registers itself via entry points so `axio-tui` discovers it automatically:

```toml
[project.entry-points."axio.transport"]
openai = "axio_transport_openai:OpenAITransport"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-transport-codex](https://github.com/axio-agent/axio-transport-codex) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
