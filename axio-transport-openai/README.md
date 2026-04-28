# axio-transport-openai

[![PyPI](https://img.shields.io/pypi/v/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

OpenAI-compatible streaming transport for [axio](https://github.com/mosquito/axio-agent).

Works with any API that speaks the OpenAI chat completions format - OpenAI itself, local models via Ollama/vLLM/LM Studio, Nebius AI Studio, OpenRouter, and any other compatible provider.

## Features

- **Full SSE streaming** - parses `data:` chunks incrementally; no waiting for full responses
- **Automatic retry** - configurable backoff on 429 and 5xx responses; honours `Retry-After` header
- **Tool calling** - streams tool-use JSON fragments as `ToolInputDelta` events; parallel tool calls supported
- **Reasoning support** - `<think>...</think>` blocks emitted as `ReasoningDelta` events
- **Embeddings** - `embed()` method for models that support `/v1/embeddings`
- **Sub-transports** - `NebiusTransport`, `OpenRouterTransport`, and `OpenAICompatibleTransport` for common providers
- **aiohttp-based** - zero blocking I/O
- **Optional TUI settings screen** - install with `[tui]` extra for a Textual configuration UI

## Installation

```bash
pip install axio-transport-openai
```

With TUI settings screens:

```bash
pip install "axio-transport-openai[tui]"
```

## Usage

```python
import asyncio
import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import TextDelta
from axio_transport_openai import OpenAITransport, OPENAI_MODELS

async def main() -> None:
    async with aiohttp.ClientSession() as session:
        transport = OpenAITransport(
            api_key="sk-...",
            model=OPENAI_MODELS["gpt-4.1-mini"],
            session=session,
        )
        agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)
        ctx = MemoryContextStore()
        async for event in agent.run_stream("What is 2 + 2?", ctx):
            if isinstance(event, TextDelta):
                print(event.delta, end="", flush=True)
        print()

asyncio.run(main())
```

The `session` parameter is **required** for streaming. Create an `aiohttp.ClientSession` in an async context and pass it to the transport.

### Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Default API key if not passed to the constructor |
| `OPENAI_BASE_URL` | Default base URL (falls back to `https://api.openai.com/v1`) |

### Local models (Ollama, vLLM, LM Studio)

```python
from axio.models import ModelSpec, Capability
from axio_transport_openai import OpenAITransport

transport = OpenAITransport(
    api_key="ollama",                        # any non-empty string
    model=ModelSpec(id="llama3.2", capabilities=frozenset({Capability.text})),
    base_url="http://localhost:11434/v1",
)
```

### Streaming events

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

## Configuration reference

`OpenAITransport` is a dataclass with the following fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"OpenAI"` | Display name (used by TUI) |
| `api_key` | `str` | `$OPENAI_API_KEY` | API key |
| `base_url` | `str` | `$OPENAI_BASE_URL` or `https://api.openai.com/v1` | API base URL |
| `model` | `ModelSpec` | `OPENAI_MODELS["gpt-4.1-mini"]` | Active model |
| `models` | `ModelRegistry` | all `OPENAI_MODELS` | Available models |
| `session` | `aiohttp.ClientSession \| None` | `None` | HTTP session (required for streaming) |
| `max_retries` | `int` | `10` | Maximum retry attempts on 429 / 5xx |
| `retry_base_delay` | `float` | `5.0` | Base delay in seconds for exponential backoff |

## Models

| Model ID | Context | Max output | Capabilities | Price (in/out per M tokens) |
|---|---|---|---|---|
| `gpt-5.4` | 1,050,000 | 128,000 | text, vision, tool use | $10 / $40 |
| `gpt-5.4-mini` | 400,000 | 128,000 | text, vision, tool use | $1.50 / $6 |
| `gpt-5.4-nano` | 400,000 | 128,000 | text, tool use | $0.30 / $1.20 |
| `gpt-5.1` | 400,000 | 128,000 | text, vision, tool use | $5 / $20 |
| `gpt-5` | 400,000 | 128,000 | text, vision, tool use | $5 / $20 |
| `gpt-5-mini` | 400,000 | 128,000 | text, vision, tool use | $1.25 / $5 |
| `gpt-5-nano` | 400,000 | 128,000 | text, tool use | $0.25 / $1 |
| `o4-mini` | 200,000 | 100,000 | text, reasoning, tool use | $1.10 / $4.40 |
| `o3` | 200,000 | 100,000 | text, reasoning, tool use | $10 / $40 |
| `o3-mini` | 200,000 | 100,000 | text, reasoning, tool use | $1.10 / $4.40 |
| `gpt-4.1` | 1,047,576 | 32,768 | text, vision, tool use | $2 / $8 |
| `gpt-4.1-mini` | 1,047,576 | 32,768 | text, vision, tool use | $0.40 / $1.60 |
| `gpt-4.1-nano` | 1,047,576 | 32,768 | text, tool use | $0.10 / $0.40 |
| `gpt-4o` | 128,000 | 16,384 | text, vision, tool use | $2.50 / $10 |
| `gpt-4o-mini` | 128,000 | 16,384 | text, vision, tool use | $0.15 / $0.60 |

The default model is `gpt-4.1-mini`.

## `fetch_models()`

`await transport.fetch_models()` resets `transport.models` to the built-in `OPENAI_MODELS` registry. It does not make a network request. Override `model` directly to switch the active model.

## Serialisation

`OpenAITransport` supports JSON round-trip for storing and restoring configuration:

```python
# Serialise
data = transport.to_dict()   # -> {"name": ..., "base_url": ..., "api_key": ..., "models": [...]}

# Restore
transport = OpenAITransport.from_dict(data, session=session)
```

`from_dict` falls back to `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables if the stored values are empty.

## Sub-transports

### NebiusTransport

`NebiusTransport` connects to [Nebius AI Studio](https://studio.nebius.com/) (`https://api.tokenfactory.nebius.com/v1`). It inherits all retry and streaming behaviour from `OpenAITransport`.

```python
from axio_transport_openai.nebius import NebiusTransport

transport = NebiusTransport(
    api_key="...",          # or set NEBIUS_API_KEY
    session=session,
)
```

| Field | Default |
|---|---|
| `name` | `"Nebius AI Studio"` |
| `api_key` | `$NEBIUS_API_KEY` |
| `base_url` | `https://api.tokenfactory.nebius.com/v1` |
| `model` | `deepseek-ai/DeepSeek-V3-0324` |

`fetch_models()` queries `/v1/models?verbose=true` and populates `transport.models` with all models returned by the API, including their context windows, output limits, capabilities (text, vision, tool use, embedding), and pricing.

### OpenRouterTransport

`OpenRouterTransport` connects to [OpenRouter](https://openrouter.ai/) (`https://openrouter.ai/api/v1`), which provides a unified API over hundreds of models from many providers.

```python
from axio_transport_openai.openrouter import OpenRouterTransport

transport = OpenRouterTransport(
    api_key="...",          # or set OPENROUTER_API_KEY
    session=session,
)
```

| Field | Default |
|---|---|
| `name` | `"OpenRouter"` |
| `api_key` | `$OPENROUTER_API_KEY` |
| `base_url` | `https://openrouter.ai/api/v1` |
| `model` | `google/gemini-2.5-pro-preview` |

`fetch_models()` queries `/v1/models` and populates `transport.models` with all models returned by the API, including their context windows, output limits, capabilities (text, vision, tool use, embedding), and pricing.

### OpenAICompatibleTransport

`OpenAICompatibleTransport` is a thin subclass of `OpenAITransport` for user-defined custom providers. Instances are created by the TUI hub screen and persisted to `~/.local/share/axio/openai-custom.json`. You can also instantiate them directly:

```python
from axio.models import ModelSpec, ModelRegistry, Capability
from axio_transport_openai.custom import OpenAICompatibleTransport

transport = OpenAICompatibleTransport(
    name="localai",
    base_url="http://localhost:8080/v1",
    api_key="",
    models=ModelRegistry([
        ModelSpec(
            id="llama3.2",
            context_window=131_072,
            max_output_tokens=4_096,
            capabilities=frozenset({Capability.text, Capability.tool_use}),
        )
    ]),
    session=session,
)
```

`fetch_models()` is a no-op for this transport - the model list is provided at construction time.

The JSON configuration format used by the TUI is:

```json
[
  {
    "name": "localai",
    "base_url": "http://localhost:8080/v1",
    "api_key": "",
    "models": [
      {
        "id": "llama3.2",
        "context_window": 131072,
        "max_output_tokens": 4096,
        "capabilities": ["text", "tool_use"],
        "input_cost": 0.0,
        "output_cost": 0.0
      }
    ]
  }
]
```

## Plugin registration

When installed, this package registers all four transports via entry points so `axio-tui` discovers them automatically:

```toml
[project.entry-points."axio.transport"]
openai         = "axio_transport_openai:OpenAITransport"
nebius         = "axio_transport_openai.nebius:NebiusTransport"
openrouter     = "axio_transport_openai.openrouter:OpenRouterTransport"
openai-custom  = "axio_transport_openai.custom:OpenAICompatibleTransport"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-transport-codex](https://github.com/mosquito/axio-agent) · [axio-transport-anthropic](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
