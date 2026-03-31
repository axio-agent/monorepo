# axio-transport-codex

[![PyPI](https://img.shields.io/pypi/v/axio-transport-codex)](https://pypi.org/project/axio-transport-codex/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-codex)](https://pypi.org/project/axio-transport-codex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ChatGPT OAuth transport for [axio](https://github.com/axio-agent/axio) using the OpenAI Responses API.

Authenticates via the same OAuth flow used by ChatGPT — no API key required; your ChatGPT subscription covers usage. Implements the OpenAI Responses API (not the legacy completions endpoint).

## Features

- **No API key** — authenticates with your ChatGPT account via OAuth
- **Responses API** — uses OpenAI's newer stateful Responses endpoint
- **Full streaming** — incremental text and tool-call events via SSE
- **Tool calling** — works with all axio tool handlers
- **aiohttp-based** — non-blocking I/O throughout

## Installation

```bash
pip install axio-transport-codex
```

## Usage

```python
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_codex import CodexTransport

transport = CodexTransport(
    access_token="your-chatgpt-oauth-token",
    model="gpt-4o",
)

agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)

async def main() -> None:
    ctx = MemoryContextStore()
    async for event in agent.run_stream("Summarise the Rust ownership model", ctx):
        from axio.events import TextDelta
        if isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)
    print()
```

## Plugin registration

```toml
[project.entry-points."axio.transport"]
codex = "axio_transport_codex:CodexTransport"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-transport-openai](https://github.com/axio-agent/axio-transport-openai) · [axio-transport-nebius](https://github.com/axio-agent/axio-transport-nebius) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
