# axio-transport-codex

[![PyPI](https://img.shields.io/pypi/v/axio-transport-codex)](https://pypi.org/project/axio-transport-codex/)
[![Python](https://img.shields.io/pypi/pyversions/axio-transport-codex)](https://pypi.org/project/axio-transport-codex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ChatGPT OAuth transport for [axio](https://github.com/mosquito/axio-agent) using the OpenAI Responses API.

Authenticates via the same OAuth2 PKCE flow used by the ChatGPT desktop client - no API key purchase required; your ChatGPT subscription covers usage. Implements the OpenAI Responses API (not the legacy chat completions endpoint).

## Features

- **No API key purchase** - authenticates with your ChatGPT account via OAuth2 PKCE
- **Responses API** - uses OpenAI's newer Responses endpoint (`/backend-api/codex/responses`)
- **Automatic token refresh** - access tokens are silently refreshed before expiry; callers notified via `on_auth_refresh`
- **Full streaming** - incremental text, reasoning, and tool-call events via SSE
- **Tool calling** - works with all axio tool handlers; parallel tool calls enabled
- **Retry logic** - automatic backoff on 429 and 5xx responses; honours `Retry-After` header
- **aiohttp-based** - non-blocking I/O throughout

## Installation

```bash
pip install axio-transport-codex
```

With the TUI settings screen (ChatGPT sign-in dialog):

```bash
pip install "axio-transport-codex[tui]"
```

## Authentication

`CodexTransport` authenticates with ChatGPT using OAuth2 PKCE. The transport stores an access token (`api_key`), a refresh token, and an expiry timestamp. It refreshes the access token automatically when it is within 30 seconds of expiry.

### Running the OAuth flow

Use `run_oauth_flow()` from `axio_transport_codex.oauth` to obtain tokens for the first time. It opens a browser window for ChatGPT sign-in and waits for the callback on `http://localhost:1455/auth/callback`.

```python
import asyncio
from axio_transport_codex.oauth import run_oauth_flow
from axio_transport_codex import CodexTransport

async def main() -> None:
    tokens = await run_oauth_flow()
    # tokens = {
    #   "access_token": "...",
    #   "refresh_token": "...",
    #   "expires_at": "1234567890",   # Unix timestamp as string
    #   "account_id": "...",
    # }

    transport = CodexTransport(
        api_key=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_at=tokens["expires_at"],
        account_id=tokens["account_id"],
    )
```

### Persisting tokens across sessions

Pass an `on_auth_refresh` callback to receive updated credentials whenever the access token is refreshed. Save the returned dict to disk or a secrets store and restore it on the next run.

```python
import json, pathlib, aiohttp
from axio_transport_codex import CodexTransport

CRED_FILE = pathlib.Path("~/.config/axio/codex.json").expanduser()

async def save_tokens(tokens: dict[str, str]) -> None:
    CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    CRED_FILE.write_text(json.dumps(tokens))

async def main() -> None:
    creds = json.loads(CRED_FILE.read_text()) if CRED_FILE.exists() else {}

    async with aiohttp.ClientSession() as session:
        transport = CodexTransport(
            api_key=creds.get("api_key", ""),
            refresh_token=creds.get("refresh_token", ""),
            expires_at=creds.get("expires_at", ""),
            account_id=creds.get("account_id", ""),
            session=session,
            on_auth_refresh=save_tokens,
        )
        # use transport with an Agent ...
```

### Refreshing tokens manually

```python
from axio_transport_codex.oauth import refresh_access_token

tokens = await refresh_access_token(refresh_token="...")
# tokens = {"access_token": ..., "refresh_token": ..., "expires_at": ..., "account_id": ...}
```

## Usage

```python
import asyncio
import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import TextDelta
from axio_transport_codex import CodexTransport, CODEX_MODELS

async def main() -> None:
    async with aiohttp.ClientSession() as session:
        transport = CodexTransport(
            api_key="<your-chatgpt-access-token>",
            refresh_token="<your-refresh-token>",
            expires_at="<unix-timestamp-string>",
            account_id="<your-account-id>",
            model=CODEX_MODELS["gpt-4.1"],
            session=session,
        )
        agent = Agent(system="You are a helpful assistant.", tools=[], transport=transport)
        ctx = MemoryContextStore()
        async for event in agent.run_stream("Explain the Rust borrow checker.", ctx):
            if isinstance(event, TextDelta):
                print(event.delta, end="", flush=True)
        print()

asyncio.run(main())
```

The `session` parameter is **required** for streaming. Create an `aiohttp.ClientSession` in an async context and pass it to the transport.

## Configuration reference

`CodexTransport` is a dataclass with the following fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"ChatGPT (Codex)"` | Display name (used by TUI) |
| `api_key` | `str` | `""` | ChatGPT OAuth access token |
| `refresh_token` | `str` | `""` | OAuth refresh token for silent renewal |
| `expires_at` | `str` | `""` | Access token expiry as a Unix timestamp string |
| `account_id` | `str` | `""` | ChatGPT account/organisation ID (sent as `ChatGPT-Account-ID` header) |
| `base_url` | `str` | `"https://chatgpt.com/backend-api/codex"` | API base URL |
| `model` | `ModelSpec` | `CODEX_MODELS["gpt-4.1"]` | Active model |
| `models` | `ModelRegistry` | all `CODEX_MODELS` | Available models |
| `session` | `aiohttp.ClientSession \| None` | `None` | HTTP session (required for streaming) |
| `on_auth_refresh` | `Callable[[dict[str, str]], Awaitable[None]] \| None` | `None` | Callback invoked with fresh credentials after token refresh |
| `max_retries` | `int` | `10` | Maximum retry attempts on 429 / 5xx |
| `retry_base_delay` | `float` | `5.0` | Base delay in seconds for exponential backoff |

## Models

| Model ID | Context | Max output | Capabilities |
|---|---|---|---|
| `gpt-4.1` | 1,047,576 | 32,768 | text, vision, tool use |
| `gpt-4.1-mini` | 1,047,576 | 32,768 | text, vision, tool use |
| `gpt-4.1-nano` | 1,047,576 | 32,768 | text, tool use |
| `gpt-4o` | 128,000 | 16,384 | text, vision, tool use |
| `gpt-4o-mini` | 128,000 | 16,384 | text, vision, tool use |
| `o4-mini` | 200,000 | 100,000 | text, reasoning, tool use |
| `o3` | 200,000 | 100,000 | text, reasoning, tool use |
| `o3-mini` | 200,000 | 100,000 | text, reasoning, tool use |

The default model is `gpt-4.1`.

## `fetch_models()`

`await transport.fetch_models()` queries the Codex `/models` endpoint for the list of models your account has access to. If the request fails or the account is not configured, it falls back to the built-in `CODEX_MODELS` registry. Models not found in the built-in registry are added with basic `text` and `tool_use` capabilities.

`session` and `api_key` must be set before calling `fetch_models()`.

## Plugin registration

When installed, this package registers itself via entry points so `axio-tui` discovers it automatically:

```toml
[project.entry-points."axio.transport"]
codex = "axio_transport_codex:CodexTransport"

[project.entry-points."axio.transport.settings"]
codex = "axio_transport_codex.tui:CodexSettingsScreen"
```

## Part of the axio ecosystem

[axio](https://github.com/mosquito/axio-agent) · [axio-transport-openai](https://github.com/mosquito/axio-agent) · [axio-transport-anthropic](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
