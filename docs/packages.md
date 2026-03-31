# Packages

The Axio monorepo contains 10 packages, each with a focused responsibility.
All packages live under `packages/` and are managed as a uv workspace.

## Overview

| Package | Purpose | Entry Point Groups |
|---------|---------|-------------------|
| `axio` | Core framework | — |
| `axio-transport-openai` | OpenAI-compatible transport | `axio.transport`, `axio.transport.settings` |
| `axio-transport-nebius` | Nebius AI Studio transport | `axio.transport`, `axio.transport.settings` |
| `axio-transport-codex` | ChatGPT (Codex) OAuth transport | `axio.transport`, `axio.transport.settings` |
| `axio-tools-local` | Filesystem & shell tools | `axio.tools` |
| `axio-tools-mcp` | MCP tool loader | `axio.tools.settings` |
| `axio-tools-docker` | Docker sandbox tools | `axio.tools.settings` |
| `axio-tui` | Textual-based TUI app | `axio.tools` |
| `axio-tui-rag` | RAG plugin (LanceDB) | `axio.tools` |
| `axio-tui-guards` | Permission guard plugins | `axio.guards` |

## Core

### axio

The foundation. Defines the agent loop, all protocols (`CompletionTransport`,
`ContextStore`, `PermissionGuard`), the tool system, stream events, and
testing helpers. Has no entry points — other packages depend on it.

Dependencies: `pydantic>=2`

## Transports

### axio-transport-openai

OpenAI-compatible HTTP streaming transport using `aiohttp` and SSE parsing.
Works with any OpenAI-compatible API (OpenAI, Azure, local servers).

Entry points:
- `axio.transport` → `OpenAITransport`
- `axio.transport.settings` → `OpenAISettingsScreen`

Dependencies: `axio`, `aiohttp>=3.11`

### axio-transport-nebius

Nebius AI Studio transport. Extends `axio-transport-openai` with
Nebius-specific configuration and model catalog.

Entry points:
- `axio.transport` → `NebiusTransport`
- `axio.transport.settings` → `NebiusSettingsScreen`

Dependencies: `axio-transport-openai`

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
| `shell` | `Shell` | Run shell commands |
| `run_python` | `RunPython` | Execute Python code |
| `write_file` | `WriteFile` | Write content to a file |
| `patch_file` | `PatchFile` | Apply patches to files |
| `read_file` | `ReadFile` | Read file contents |
| `list_files` | `ListFiles` | List directory contents |

Dependencies: `axio`

### axio-tools-mcp

Dynamic tool provider that loads tools from MCP (Model Context Protocol)
servers. Registered as a `ToolsPlugin` under `axio.tools.settings`.

Dependencies: `axio`, `mcp>=1.6`

### axio-tools-docker

Dynamic tool provider for Docker sandbox environments. Provides isolated
tool execution inside containers. Registered as a `ToolsPlugin` under
`axio.tools.settings`.

Dependencies: `axio`

## TUI & Plugins

### axio-tui

Terminal UI application built with Textual. Provides the `axio` console
command, plugin discovery, transport management, and session persistence
via SQLite.

Tools registered under `axio.tools`:
- `status_line` — Update the TUI status bar
- `confirm` — Ask user for confirmation
- `subagent` — Spawn a sub-agent
- `vision` — Analyze images

Console script: `axio = "axio_tui.__main__:main"`

Dependencies: `axio`, `textual>=2.1.0`, `aiosqlite>=0.20`

### axio-tui-rag

RAG (Retrieval-Augmented Generation) plugin using LanceDB for vector search.

Tools registered under `axio.tools`:
- `index_files` — Index files for semantic search
- `semantic_search` — Search indexed content

Dependencies: `axio`, `axio-tui`, `lancedb>=0.20`

### axio-tui-guards

Permission guard plugins for the TUI.

Guards registered under `axio.guards`:
- `path` — `PathGuard` — Validates file paths against allowed directories
- `llm` — `LLMGuard` — Uses LLM to assess tool call safety

Dependencies: `axio`, `axio-tui`
