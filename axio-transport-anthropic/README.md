# axio-transport-anthropic

[![PyPI](https://img.shields.io/pypi/v/axio-transport-anthropic)](https://pypi.org/project/axio-transport-anthropic/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-anthropic)](https://pypi.org/project/axio-transport-anthropic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Anthropic Claude transport for [axio](https://github.com/mosquito/axio-agent).

Streams Claude responses over the Anthropic Messages API using `aiohttp` and
SSE parsing. Supports prompt caching, extended thinking, and automatic retry
on rate-limit and overload errors.

## Features

- **All Claude models** - Opus, Sonnet, Haiku; configurable via `ANTHROPIC_MODELS`
- **Prompt caching** - `cache_control: ephemeral` applied automatically to the
  system prompt and the last tool definition
- **Extended thinking** - `ReasoningDelta` events emitted for thinking blocks
- **Retry logic** - automatic backoff on 429 and all 5xx errors; honours `Retry-After` header
- **TUI integration** - settings screen for API key and model selection

## Installation

```bash
pip install axio-transport-anthropic
```

With the TUI settings screen:

```bash
pip install "axio-transport-anthropic[tui]"
```

Or as part of the full TUI bundle:

```bash
pip install "axio-tui[anthropic]"
```

## Usage

<!-- name: test_readme_usage -->
```python
import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio_transport_anthropic import AnthropicTransport, ANTHROPIC_MODELS

async def main() -> None:
    async with aiohttp.ClientSession() as session:
        transport = AnthropicTransport(
            api_key="sk-ant-...",
            model=ANTHROPIC_MODELS["claude-sonnet-4-6"],
            session=session,
        )
        agent = Agent(system="You are helpful.", tools=[], transport=transport)
        ctx = MemoryContextStore()
        print(await agent.run("Hello!", ctx))
```

Set the API key via environment variable instead of passing it directly:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Override the base URL (e.g. for a proxy or private endpoint):

```bash
export ANTHROPIC_BASE_URL="https://your-proxy.example.com/v1"
```

## Models

| Model ID | Context | Max output | Notes |
|---|---|---|---|
| `claude-opus-4-6` | 1 M | 128 k | Most capable |
| `claude-sonnet-4-6` | 1 M | 64 k | Balanced (default) |
| `claude-haiku-4-5-20251001` | 200 k | 64 k | Fastest / cheapest |
| `claude-opus-4-5` | 200 k | 64 k | - |
| `claude-sonnet-4-5` | 200 k | 64 k | - |

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `""` | Anthropic API key |
| `model` | `claude-sonnet-4-6` | Active model |
| `base_url` | `https://api.anthropic.com/v1` | API base URL |
| `max_retries` | `10` | Max retry attempts on 429 and 5xx errors |
| `retry_base_delay` | `5.0` | Base delay (seconds) for exponential backoff |

## `fetch_models()`

`await transport.fetch_models()` resets `transport.models` to the built-in `ANTHROPIC_MODELS` registry. It does not make a network request. Override `model` directly to switch the active model.

## Serialisation

`AnthropicTransport` supports JSON round-trip for storing and restoring configuration:

```python
# Serialise
data = transport.to_dict()   # -> {"name": ..., "base_url": ..., "api_key": ..., "models": [...]}

# Restore
transport = AnthropicTransport.from_dict(data, session=session)
```

`from_dict` falls back to `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` environment variables if the stored values are empty.

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent) · [axio-transport-openai](https://github.com/mosquito/axio-agent) · [axio-transport-codex](https://github.com/mosquito/axio-agent)

## License

MIT
