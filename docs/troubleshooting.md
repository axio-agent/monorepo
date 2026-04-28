# Troubleshooting

Solutions for common issues when working with Axio.

## Installation

### `uv: command not found`

Install [uv](https://docs.astral.sh/uv/) first:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

Or use pip: `pip install "axio-tui[all]"`

## API Keys

### `API key not found` / `Missing API key`

Axio looks for API keys in environment variables. Set the appropriate variable:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Nebius
export NEBIUS_API_KEY="..."

# OpenRouter
export OPENROUTER_API_KEY="..."
```

For the TUI, you can also set keys via the settings screen (open the command palette with `Ctrl+P` and search for the transport settings).

### `Invalid API key` / `Authentication error`

- Verify the key is correct - no extra spaces or quotes in the env var
- Check the key has not expired
- For OpenAI: make sure you have active billing
- For Anthropic: ensure the key has the right permissions

## Transport Connection

### `Connection refused` / `Failed to connect`

- Check your internet connection
- Verify the API endpoint is correct (especially for custom endpoints)
- Some corporate networks block external APIs - try using a VPN
- For OpenAI-compatible APIs: verify the base URL in your transport config

### `Timeout error`

- The API may be slow or experiencing high load
- Try again in a few moments
- If persistent, increase the timeout in your transport settings

### `SSL certificate error`

- Update your Python version - newer versions have updated CA certificates
- On macOS: run `/Applications/Python\ 3.x/Install\ Certificates.command`
- On Linux: update ca-certificates: `sudo apt update && sudo apt install ca-certificates`

## Tools

### `Tool not found`

Tools must be registered as entry points or explicitly passed to the Agent.

1. If using the TUI: make sure the package with tools is installed (e.g., `uv tool install "axio-tui[local]"` for filesystem tools)

2. If programmatically: pass tools directly to the Agent:

```python
from my_tool import my_tool

agent = Agent(
    system="You are helpful.",
    tools=[my_tool],  # explicitly pass
    transport=transport,
)
```

### `Tool execution failed`

Check the error message:

- **Timeout**: the tool took too long - consider async optimization
- **Permission denied**: a guard blocked the tool - see "Permission guards" below
- **Import error**: check the tool handler's dependencies are installed

### `Tool returned empty result`

- Verify the tool logic is correct
- Check logs for exceptions during execution
- Add debug output in your tool handler to see what's happening

## Permission Guards

### `Permission denied` for every tool call

Guards are blocking all tool calls. Check:

1. Check which guards are attached to your agent in your configuration
2. For path guards: ensure you are running from a directory the guard will allow
3. For LLM guards: ensure you have an API key configured

### `PathGuard: path not allowed`

The path guard prompts for permission on each new directory. It does not take
a pre-configured allow list - instead it asks at runtime. To attach it:

```python
from axio_tui_guards import PathGuard

guard = PathGuard()  # uses interactive prompt by default
agent = Agent(guards=[guard], ...)
```

When prompted, answer `y` to allow access to the directory (remembered for the
session), or `deny` to permanently block that path for the session.

## Context & Storage

### `Database is locked` (SQLite)

Multiple processes are accessing the same SQLite database. Solutions:

- Use WAL mode (enabled by default in Axio)
- Ensure you're using a single process
- Increase busy timeout in connection string

### `Session not found`

- Check the `session_id` is correct
- For SQLite: verify the database file exists and has data
- The session may have been deleted or expired

## TUI

### `axio: command not found` after install

- Verify uv tool install worked: `uv tool list`
- Add uv tools to PATH: `export PATH="$HOME/.local/bin:$PATH"`
- Or run directly: `python -m axio_tui`

### TUI crashes on startup

- Check Python version - requires 3.12+
- Run with verbose logging: `axio --log-level debug` to see error output
- Try resetting config: delete `~/.local/share/axio-tui/axio.db` and restart

### `No transports found`

Install a transport package:

```bash
uv tool install "axio-tui[openai]"   # OpenAI
uv tool install "axio-tui[anthropic]" # Anthropic
```

## Development

### `Module not found` when importing axio

Ensure you're in the right environment:

```bash
cd axio
uv shell  # activate workspace
python -c "import axio; print(axio.__file__)"
```

### Type checking errors

Axio uses strict typing. Install dev dependencies:

```bash
uv sync --all-extras
mypy axio/
```

### Tests failing

Run tests with verbose output:

```bash
cd axio
uv run pytest -v
```

Check if the failure is in your code or the framework:

- If in framework: open an issue on GitHub
- If in your code: verify against the test examples in `docs/` and `axio/tests/`

## Getting Help

If your issue isn't listed here:

1. Search [GitHub issues](https://github.com/mosquito/axio-agent/issues)
2. Open a new issue with:
   - Python version
   - Axio version (`python -c "import importlib.metadata; print(importlib.metadata.version('axio'))"`)
   - Full error traceback
   - Minimal reproduction code