# Packages

The Axio packages documented here each have a focused responsibility. They are
top-level directories in the monorepo. A uv workspace manages them.

## Overview

| Package | Purpose | Entry Point Groups |
|---------|---------|-------------------|
| `axio` | Core framework | - |
| `axio-context-sqlite` | SQLite-backed persistent context store | - |
| `axio-sse` | `text/event-stream` decoder and payload reader | - |
| `axio-responses` | OpenAI Responses API: request items and stream events | - |
| `axio-transport-anthropic` | Anthropic Claude transport | `axio.transport`, `axio.transport.settings` |
| `axio-transport-openai` | OpenAI Responses transport, plus the compatible chat endpoints (Nebius, OpenRouter, custom) | `axio.transport`, `axio.transport.realtime`, `axio.transport.settings` |
| `axio-transport-codex` | ChatGPT (Codex) OAuth transport | `axio.transport`, `axio.transport.settings` |
| `axio-transport-google` | Google Gemini transport + Gemini Live realtime | `axio.transport`, `axio.transport.realtime`, `axio.transport.settings`, `axio.tools` |
| `axio-audio` | Microphone and speaker helpers for realtime agents | - |
| `axio-tools-local` | Filesystem & shell tools | `axio.tools` |
| `axio-tools-mcp` | MCP tool loader | `axio.tools.settings` |
| `axio-tools-docker` | Docker sandbox tools | - |
| `axio-repl` | Interactive terminal coding assistant | - |
| `axio-tui` | Textual chat application | `axio.tools` |
| `axio-tui-guards` | Path and LLM guards for the TUI | `axio.guards` |

## Core

### axio

The foundation. Defines the agent loop, all protocols (`CompletionTransport`,
`ContextStore`, `PermissionGuard`), the tool system, stream events, and testing
helpers. Has no entry points. Other packages depend on it.

Dependencies: none (stdlib only)

## Wire formats

### axio-sse

Reads `text/event-stream`. `Decoder` is the format as a synchronous state machine.
Feed it chunks cut anywhere, and take the events they completed. `events()` and
`payloads()` are the async skin over it.

A stream whose events are all one shape needs nothing above `payloads()`. A stream
that names each event subclasses `Reader` and writes one `@on(...)` method per
event. `Wire` says what one payload looks like. Fields are read by declared name
and type, so a misspelled key is a type error at the place that uses it, rather
than a default quietly standing in for the value.

A `Reader` names only the events it interprets. Everything else reaches
`unmatched()`, which returns nothing by default. A reader overrides it to forward
instead of drop. The reason is that an endpoint which runs tools publishes one
event family per tool. That set therefore depends on which tools exist and which
the caller declared, not on the protocol. Named one by one, the list is stale the
day a tool is added. A new tool then reads as news about the protocol when it is
news about the tools. `strict=True` still raises `UnknownEvent` for any name no
method claims. A test can therefore hold `names()` against the schema the provider
publishes, without the reader carrying a list it cannot keep true.

The package takes chunks and never lines. `aiohttp`'s `readuntil` raises
`LineTooLong` past 131072 bytes. `LineTooLong` is not a `ClientError`. One large
reasoning event kills a turn with no answer.

Dependencies: none (stdlib only)

### axio-responses

The OpenAI Responses API as axio speaks it. `convert_messages` and `convert_tools`
build the request. `Responses` is an `axio_sse.Reader` that turns the stream into
`StreamEvent`s.

The claim `names()` supports runs one way only: every name the reader claims is
one the schema publishes. That is what catches a typo in a claimed name. The
reverse does not hold, and deliberately. The reader does not name the hosted-tool
event families. It forwards them through `unmatched()` as
`ProviderEvent(provider="openai", ...)`.

Both halves live here rather than in a transport because two transports speak this
API: `axio-transport-openai` against `/v1/responses`, and `axio-transport-codex`
against the ChatGPT backend.

Dependencies: `axio`, `axio-sse`

## Context Stores

### axio-context-sqlite

SQLite-backed persistent context store. Implements the `axio.context.ContextStore`
protocol so conversations survive process restarts. Multiple sessions can coexist
in the same database file, isolated by `session_id` and `project`.

Features:
- Automatic gzip compression for large payloads (> 512 bytes)
- WAL journal mode with a 5-second busy timeout for concurrent access
- `list_sessions()` - list all sessions for a project, ordered newest first
- `fork()` - copy a session's messages into a new session ID
- `add_context_tokens()` - atomic token-count increment via SQL UPSERT

Dependencies: `axio`, `aiosqlite>=0.20`

## Transports

### axio-transport-anthropic

Anthropic Claude transport. It posts with `aiohttp`. It reads the stream through
`axio-sse`: an `axio_sse.Reader` keyed on the format's own `event:` field, with
one `@on(...)` method per event. It supports all Claude models with prompt caching
(`cache_control`). It retries automatically on rate-limit (429) and overload (529)
responses.

Its `input_tokens` are converted into the axio rule before they leave. The API
counts only the tokens after the last cache breakpoint. The cache read and cache
write counts are therefore added back, and reported as slices of an inclusive
total.

Entry points:
- `axio.transport` → `AnthropicTransport`
- `axio.transport.settings` → `AnthropicSettingsScreen`

Dependencies: `axio`, `axio-sse`, `aiohttp>=3.11`

### axio-transport-openai

OpenAI HTTP streaming transports. They post with `aiohttp` and read the stream
through `axio-sse`. The Responses vocabulary itself lives in `axio-responses`.
Four transports are registered as entry points:

| Entry point name | Class | Provider |
|---|---|---|
| `openai` | `OpenAITransport` | OpenAI API |
| `nebius` | `NebiusTransport` | Nebius AI Studio |
| `openrouter` | `OpenRouterTransport` | OpenRouter |
| `openai-custom` | `OpenAICompatibleTransport` | Any OpenAI-compatible endpoint |

Settings screens are registered under `axio.transport.settings` for each.
`OpenAIRealtimeTransport` is registered under `axio.transport.realtime` as
`openai`.

`api: Literal["responses", "chat"]` selects the endpoint. `OpenAITransport`
defaults to `"responses"` and posts to `/v1/responses`, because
`/v1/chat/completions` refuses function tools beside any reasoning effort other
than `"none"` for a model that reasons. A request carrying tools fails there with
a 400 naming a parameter the caller never sent. The three compatible subclasses
say `"chat"`, since compatible servers rarely implement `/v1/responses`. See
{ref}`the troubleshooting entry <tools-and-reasoning-400>`.

`extra_params` is folded into the request. Its `tools` are merged rather than
substituted. A caller adding a hosted tool would otherwise take away the function
declarations the agent needs dispatched. The turn would then read as the model
simply choosing to call nothing. A declaration whose name matches one already
there wins.

Dependencies: `axio`, `axio-responses`, `axio-sse`, `aiohttp>=3.11`

### axio-transport-codex

ChatGPT (Codex) transport, speaking the Responses API against the ChatGPT backend
with OAuth authentication. It shares `axio-responses` with
`axio-transport-openai`, so it depends on `axio-sse` only through it.

Entry points:
- `axio.transport` → `CodexTransport`
- `axio.transport.settings` → `CodexSettingsScreen`

Dependencies: `axio`, `axio-responses`, `aiohttp>=3.11`

### axio-transport-google

Google GenAI (Gemini) transport for the Developer API and Vertex AI. Supports
standard completion and Gemini Live realtime sessions. Also registers image and
video generation tools when installed.

Entry points:
- `axio.transport` → `GoogleTransport`, `VertexAITransport`
- `axio.transport.realtime` → `GeminiLiveTransport` (`gemini`), `VertexLiveTransport` (`vertex`)
- `axio.transport.settings` → `GoogleSettingsScreen`, `VertexSettingsScreen`
- `axio.tools` → `generate_image`, `generate_video`

Gemini's stream carries no per-event discriminator, so this transport reads it
with `axio_sse.payloads()` and `Wire` shapes rather than with a `Reader`. Its
token counts are converted into the axio rule on the way out. Tool-use prompt
tokens are not inside `promptTokenCount`, and thinking is not inside
`candidatesTokenCount`, so both are added.

See the {doc}`guides/google-transport` guide.

Dependencies: `axio`, `axio-sse`, `axio-transport-anthropic[vertexai]`, `google-auth[urllib3]>=2.0`

`aiohttp` is not declared here. The HTTP client arrives through
`axio-transport-anthropic`, which this package builds on for Vertex AI
authentication.

## Audio

### axio-audio

Microphone capture and speaker playback for realtime voice agents.
Provides `Microphone`, `Speaker`, and `DuplexAudio` (single-clock duplex
stream for production-grade echo cancellation).

See the {doc}`guides/realtime-audio` guide.

Dependencies: `axio`, `sounddevice>=0.5`, `numpy>=2`

## Tools

### axio-tools-local

Filesystem and shell tool handlers for local development:

| Entry Point | Handler | Description |
|-------------|---------|-------------|
| `shell` | `shell` | Run shell commands |
| `run_python` | `run_python` | Execute Python code |
| `write_file` | `write_file` | Write content to a file |
| `patch_file` | `patch_file` | Apply patches to files |
| `read_file` | `read_file` | Read file contents |
| `list_files` | `list_files` | List directory contents |

Dependencies: `axio`

### axio-tools-mcp

Dynamic tool loader that connects to MCP (Model Context Protocol) servers and
returns regular `Tool` objects for explicit use in an agent.

Entry points:
- `axio.tools.settings` → `MCPPlugin`

Dependencies: `axio`, `mcp>=1.6`

### axio-tools-docker

Docker sandbox environment as an async context manager. Spins up an isolated
container via `aiodocker`. Exposes six tools that mirror `axio-tools-local`:
`shell`, `write_file`, `read_file`, `list_files`, `run_python`, `patch_file`. No
entry points. Use `DockerSandbox` directly in code.

```python
async with DockerSandbox(image="python:3.12-slim") as sandbox:
    agent = Agent(..., tools=sandbox.tools)
```

Dependencies: `axio`, `aiodocker>=0.26`

## Applications

### axio-tui

Textual chat application. Stores conversations with `axio-context-sqlite`. Keeps
per-project configuration in a database of its own. Discovers transports, tools,
guards and settings screens through the entry-point groups above. Optional extras
name the transports and tool packages it can drive: `anthropic`, `openai`,
`codex`, `local`, `mcp`, `guards`, and `all` for the lot.

Entry points:
- `axio.tools` → `status_line`, `confirm`, `subagent`, `vision`

Console script: `axio = "axio_tui.__main__:main"`

Dependencies: `axio`, `axio-context-sqlite`, `textual>=2.1.0`, `textual-serve>=1.1`, `argclass>=1.6`

### axio-tui-guards

The TUI offers two guards. `PathGuard` asks before a tool touches a path, and
remembers the answer for that directory. `LLMGuard` puts the decision to an agent,
and feeds the user's overrides back into its context. Both are registered as entry
points, so the TUI can list them without importing them.

Entry points:
- `axio.guards` → `path` (`PathGuard`), `llm` (`LLMGuard`)

Dependencies: `axio`, `axio-tui`

## REPL

### axio-repl

Terminal coding assistant. Runs an agent loop with file/shell tools. Streams every
token and tool call to the terminal. Auto-detects the transport from environment
variables. Supports model switching, streaming tool arguments and output, vision,
and workspace-level `AGENTS.md` instructions.

Console script: `axio-repl = "axio_repl:main_sync"`

See the {doc}`guides/axio-repl` guide.

Dependencies: `axio`, `axio-tools-local`, `axio-transport-openai`, `aiohttp>=3.11`
