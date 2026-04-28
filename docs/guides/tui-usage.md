# TUI Guide

The Axio Terminal User Interface (TUI) built with Textual.

## Installation

```bash
uv tool install "axio-tui[all]"
```

Or pick extras: `anthropic`, `openai`, `codex`, `local`, `mcp`, `guards`

## Launch

```bash
axio
```

## Interface

```{image} ../_static/tui-screenshot.svg
:alt: Axio TUI
:width: 100%
```

## Key Bindings

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+L` | Clear log |
| `Ctrl+P` | Command palette |
| `Escape` | Stop agent |
| `F12` | Toggle Dev Console |
| `Alt+Up` | Previous message |
| `Alt+Down` | Next message |

## Command Palette

Press `Ctrl+P` to access:

- Search Model
- Switch LLM
- Clear chat log
- Compact Conversation
- Download messages
- Configure transports (OpenAI, Anthropic, Codex, OpenRouter, Nebius)
- Configure Docker Sandbox
- Fork Conversation
- New Session
- Reset conversation
- Configure MCP Servers
- Manage Plugins
- Theme
- Screenshot
- Quit

## Settings

Press system shortcut to configure:
- Chat model
- Compaction model
- Sub-agent model
- Guard model
- Vision model
- Transport API keys

## Plugins

Auto-discovers via entry points:
- `axio.transport` - LLM backends
- `axio.tools` - Tools
- `axio.tools.settings` - Tools with config UI
- `axio.guards` - Permission guards

## Serve over HTTP

```bash
textual-serve axio_tui.__main__:app
```