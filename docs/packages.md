# Packages

The Axio monorepo contains 10 packages, each with a focused responsibility.
Each package is a top-level directory in the monorepo root (e.g., `axio/`, `axio-tui/`)
and all are managed as a uv workspace.

## Overview

| Package | Purpose | Entry Point Groups |
|---------|---------|-------------------|
| `axio` | Core framework | - |
| `axio-context-sqlite` | SQLite-backed persistent context store | - |
| `axio-transport-anthropic` | Anthropic Claude transport | `axio.transport`, `axio.transport.settings` |
| `axio-transport-openai` | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) | `axio.transport`, `axio.transport.settings` |
| `axio-transport-codex` | ChatGPT (Codex) OAuth transport | `axio.transport`, `axio.transport.settings` |
| `axio-tools-local` | Filesystem & shell tools | `axio.tools` |
| `axio-tools-mcp` | MCP tool loader | `axio.tools.settings` |
| `axio-tools-docker` | Docker sandbox tools | - |
| `axio-tui` | Textual-based TUI app | `axio.tools` |
| `axio-tui-guards` | Permission guard plugins | `axio.guards` |

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

Dynamic tool provider that loads tools from MCP (Model Context Protocol)
servers. Registered as a `ToolsPlugin` under `axio.tools.settings`.

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

## TUI & Plugins

### axio-tui

Terminal UI application built with Textual. Provides the `axio` console
command, plugin discovery, transport management, and session persistence
via SQLite.

Tools registered under `axio.tools`:
- `status_line` - Update the TUI status bar
- `confirm` - Ask user for confirmation
- `subagent` - Spawn a sub-agent
- `vision` - Analyze images

Console script: `axio = "axio_tui.__main__:main"`

Dependencies: `axio`, `textual>=2.1.0`, `aiosqlite>=0.20`

### axio-tui-guards

Permission guard plugins for the TUI.

Guards registered under `axio.guards`:
- `path` - `PathGuard` - Validates file paths against allowed directories
- `llm` - `LLMGuard` - Uses LLM to assess tool call safety

Dependencies: `axio`, `axio-tui`
