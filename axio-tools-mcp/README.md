# axio-tools-mcp

[![PyPI](https://img.shields.io/pypi/v/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) bridge for [axio](https://github.com/mosquito/axio-agent).

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

<!--
name: test_readme_standalone
```python
from axio.testing import StubTransport, make_text_response
transport = StubTransport([make_text_response("done")])
```
-->
<!-- name: test_readme_standalone -->
```python
import asyncio
from axio_tools_mcp.registry import MCPRegistry
from axio.agent import Agent
from axio.context import MemoryContextStore

async def main() -> None:
    registry = MCPRegistry()
    await registry.init(config=None)

    tools = registry.all_tools   # list[axio.Tool]
    print(f"Loaded {len(tools)} tools from MCP servers")

    # Pass any CompletionTransport — e.g. OpenAITransport, AnthropicTransport
    agent = Agent(
        system="You are a helpful assistant.",
        tools=tools,
        transport=transport,
    )
    result = await agent.run("Use the available tools to help me", MemoryContextStore())
    print(result)

    await registry.close()

asyncio.run(main())
```

## Transport types

Two transport types are supported, selected by which field is set in the server config:

| Transport | Config field | Protocol |
|-----------|-------------|---------|
| **stdio** | `command` | Spawns a subprocess; communicates over stdin/stdout (MCP stdio transport) |
| **HTTP** | `url` | Connects to a running HTTP server using the MCP Streamable HTTP transport (`httpx`) |

Exactly one of `command` or `url` must be set per server — providing both or neither raises a `ValueError`.

For stdio servers, stderr output from the subprocess is forwarded to the Python logger as warnings under the `mcp:<server-name>` prefix.

HTTP servers accept optional `headers` (e.g., for bearer tokens) and a configurable `timeout` (default: 30 seconds).

## Tool naming

Tools from MCP servers are named using the pattern `<server_name>__<tool_name>`. For example, a server named `filesystem` that exposes a tool called `read_file` becomes `filesystem__read_file` in axio. The tool description is taken from the MCP tool's `description` field, falling back to the tool name if no description is provided.

## Error handling

When an MCP server fails to connect or start, the error is logged at `ERROR` level and the server is skipped — no exception is raised to the caller. `registry.all_tools` will simply not include any tools from the failed server. The error message is accessible via `registry.server_status(name)` (returns `"error"`) and the raw error string is stored internally.

## Lifecycle: `close()`

Call `await registry.close()` when the agent session ends to disconnect all MCP server sessions and release resources (subprocess file descriptors, HTTP connections):

```python
await registry.close()
```

## MCP server configuration

MCP servers are configured via the `axio-tui` settings UI or programmatically:

**stdio server** (spawns a subprocess):

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

**HTTP server** (connects to a running MCP HTTP endpoint):

```json
{
  "servers": [
    {
      "name": "remote",
      "url": "https://my-mcp-server.example.com/mcp",
      "headers": {"Authorization": "Bearer my-token"},
      "timeout": 60.0
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

[axio](https://github.com/mosquito/axio-agent) · [axio-tools-local](https://github.com/mosquito/axio-agent) · [axio-tools-docker](https://github.com/mosquito/axio-agent) · [axio-tui](https://github.com/mosquito/axio-agent)

## License

MIT
