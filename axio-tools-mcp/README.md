# axio-tools-mcp

[![PyPI](https://img.shields.io/pypi/v/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) bridge for [axio](https://github.com/axio-agent/axio).

Connect any MCP server to your axio agent. Tools exposed by MCP servers are discovered at runtime and become first-class axio `Tool` instances — no manual wiring required.

## Features

- **Dynamic tool discovery** — connects to MCP servers and converts their tools into axio `Tool` instances automatically
- **Multiple servers** — configure and run several MCP servers simultaneously
- **Lifecycle management** — handles server startup, shutdown, and reconnection
- **TUI integration** — ships a settings screen for managing MCP server configuration from within `axio-tui`

## Installation

```bash
pip install axio-tools-mcp
```

## Usage

### With axio-tui (recommended)

```bash
pip install "axio-tui[mcp]"
uv run axio   # MCP Servers section appears in settings
```

### Standalone

```python
from axio_tools_mcp.registry import MCPRegistry
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport

async def main() -> None:
    registry = MCPRegistry()
    await registry.init(config=None)

    tools = registry.all_tools   # list[axio.Tool]
    print(f"Loaded {len(tools)} tools from MCP servers")

    agent = Agent(
        system="You are a helpful assistant.",
        tools=tools,
        transport=OpenAITransport(api_key="sk-...", model="gpt-4o"),
    )
    result = await agent.run("Use the available tools to help me", MemoryContextStore())
    print(result)

    await registry.close()
```

## MCP server configuration

MCP servers are configured via the `axio-tui` settings UI or programmatically:

```json
{
  "servers": [
    {
      "name": "filesystem",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    {
      "name": "github",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."}
    }
  ]
}
```

## Plugin registration

```toml
[project.entry-points."axio.tools.settings"]
mcp = "axio_tools_mcp.plugin:MCPPlugin"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-tools-local](https://github.com/axio-agent/axio-tools-local) · [axio-tools-docker](https://github.com/axio-agent/axio-tools-docker) · [axio-tui](https://github.com/axio-agent/axio-tui)

## License

MIT
