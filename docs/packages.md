# Packages

The Axio packages documented here each have a focused responsibility. They are
top-level directories in the monorepo and are managed as a uv workspace.

## Overview

| Package | Purpose | Entry Point Groups |
|---------|---------|-------------------|
| `axio` | Core framework | - |
| `axio-context-sqlite` | SQLite-backed persistent context store | - |
| `axio-transport-anthropic` | Anthropic Claude transport | `axio.transport` |
| `axio-transport-openai` | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) | `axio.transport` |
| `axio-transport-codex` | ChatGPT (Codex) OAuth transport | `axio.transport` |
| `axio-transport-google` | Google Gemini transport + Gemini Live realtime | `axio.transport`, `axio.transport.realtime`, `axio.tools` |
| `axio-audio` | Microphone and speaker helpers for realtime agents | - |
| `axio-tools-local` | Filesystem & shell tools | `axio.tools` |
| `axio-tools-mcp` | MCP tool loader | - |
| `axio-tools-docker` | Docker sandbox tools | - |
| `axio-repl` | Interactive terminal coding assistant | - |

## Core

### axio

The foundation. Defines the agent loop, all protocols (`CompletionTransport`,
`ContextStore`, `PermissionGuard`), the tool system, stream events, and
testing helpers. Has no entry points - other packages depend on it.

Dependencies: none (stdlib only)

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

Anthropic Claude transport using `aiohttp` and SSE parsing. Supports all
Claude models with prompt caching (`cache_control`) and automatic retry on
rate-limit (429) and overload (529) responses.

Entry points:
- `axio.transport` → `AnthropicTransport`
- `axio.transport.settings` → `AnthropicSettingsScreen`

Dependencies: `axio`, `aiohttp>=3.11`

### axio-transport-openai

OpenAI-compatible HTTP streaming transport using `aiohttp` and SSE parsing.
Includes four transports registered as entry points:

| Entry point name | Class | Provider |
|---|---|---|
| `openai` | `OpenAITransport` | OpenAI API |
| `nebius` | `NebiusTransport` | Nebius AI Studio |
| `openrouter` | `OpenRouterTransport` | OpenRouter |
| `openai-custom` | `OpenAICompatibleTransport` | Any OpenAI-compatible endpoint |

Settings screens are registered under `axio.transport.settings` for each.

Dependencies: `axio`, `aiohttp>=3.11`

### axio-transport-codex

ChatGPT (Codex) transport using the Responses API with OAuth authentication.

Entry points:
- `axio.transport` → `CodexTransport`
- `axio.transport.settings` → `CodexSettingsScreen`

Dependencies: `axio`, `aiohttp>=3.11`

### axio-transport-google

Google GenAI (Gemini) transport for the Developer API and Vertex AI. Supports
standard completion and Gemini Live realtime sessions. Also registers image and
video generation tools when installed.

Entry points:
- `axio.transport` → `GoogleTransport`, `VertexAITransport`
- `axio.transport.realtime` → `GeminiLiveTransport`, `VertexLiveTransport`
- `axio.transport.settings` → `GoogleSettingsScreen`, `VertexSettingsScreen`
- `axio.tools` → `generate_image`, `generate_video`

See the {doc}`guides/google-transport` guide.

Dependencies: `axio`, `axio-transport-anthropic[vertexai]`, `google-auth[urllib3]>=2.0`, `aiohttp>=3.11`

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

Dependencies: `axio`, `mcp>=1.6`

### axio-tools-docker

Docker sandbox environment as an async context manager. Spins up an isolated
container via `aiodocker` and exposes six tools that mirror `axio-tools-local`:
`shell`, `write_file`, `read_file`, `list_files`, `run_python`, `patch_file`.
No entry points - used directly in code via `DockerSandbox`.

```python
async with DockerSandbox(image="python:3.12-slim") as sandbox:
    agent = Agent(..., tools=sandbox.tools)
```

Dependencies: `axio`, `aiodocker>=0.26`

## REPL

### axio-repl

Terminal coding assistant. Runs an agent loop with file/shell tools, streams
every token and tool call to the terminal, and auto-detects the transport from
environment variables. Supports model switching, streaming tool arguments and
output, vision, and workspace-level `AGENTS.md` instructions.

Console script: `axio-repl = "axio_repl:main_sync"`

See the {doc}`guides/axio-repl` guide.

Dependencies: `axio`, `axio-tools-local`, `axio-transport-openai`, `aiohttp>=3.11`
