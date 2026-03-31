# axio-transport-nebius

[![PyPI](https://img.shields.io/pypi/v/axio-transport-nebius)](https://pypi.org/project/axio-transport-nebius/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-nebius)](https://pypi.org/project/axio-transport-nebius/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Nebius AI Studio](https://studio.nebius.ai/) transport for [axio](https://github.com/axio-agent/axio).

Extends `axio-transport-openai` with Nebius-specific API endpoint and dynamic model discovery — list available models directly from the Studio API.

## Features

- **Nebius AI Studio** endpoint pre-configured
- **Dynamic model list** — fetch available models at runtime from the Nebius API
- **Full OpenAI compatibility** — inherits streaming, retry, and tool-calling from `axio-transport-openai`
- **Optional TUI settings screen** — install with `[tui]` extra

## Installation

```bash
pip install axio-transport-nebius
```

## Usage

```python
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_nebius import NebiusTransport

transport = NebiusTransport(
    api_key="your-nebius-api-key",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
)

agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)

async def main() -> None:
    ctx = MemoryContextStore()
    result = await agent.run("Explain transformers in one paragraph.", ctx)
    print(result)
```

### Discover available models

```python
from axio_transport_nebius import NebiusTransport

async def list_models() -> None:
    transport = NebiusTransport(api_key="your-key", model="")
    models = await transport.list_models()
    for m in models:
        print(m)
```

## Plugin registration

```toml
[project.entry-points."axio.transport"]
nebius = "axio_transport_nebius:NebiusTransport"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-transport-openai](https://github.com/axio-agent/axio-transport-openai) · [axio-transport-codex](https://github.com/axio-agent/axio-transport-codex) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
