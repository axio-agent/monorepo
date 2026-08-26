# MCP Server Tools

`axio-tools-mcp` lets you plug any
[MCP (Model Context Protocol)](https://modelcontextprotocol.io) server into an
Axio agent. Tools are loaded at runtime from the server and wrapped as regular
`Tool` objects - the agent has no idea they came from MCP.

## Install

```bash
pip install axio-tools-mcp
```

## Loading tools

`load_mcp_tools` connects to one or more servers and returns a flat list of
tools plus the open sessions:

<!--
name: test_mcp_load_tools
```python
import axio_tools_mcp
import axio_transport_anthropic
from axio import Tool
from axio.testing import StubTransport, make_text_response

async def _list_files() -> str:
    """List files in the current directory."""
    return "README.md\nsrc/"

async def _fake_load(servers):
    return [Tool(name="fs__list_files", handler=_list_files)], []

axio_tools_mcp.load_mcp_tools = _fake_load
axio_transport_anthropic.AnthropicTransport = lambda: StubTransport([make_text_response("README.md and src/")])
```
-->
<!-- name: test_mcp_load_tools -->
```python
import asyncio
from axio import Agent, MemoryContextStore
from axio_transport_anthropic import AnthropicTransport
from axio_tools_mcp import load_mcp_tools, MCPServerConfig


async def main() -> None:
    servers = [
        MCPServerConfig(name="fs", command="mcp-server-filesystem", args=["--root", "."]),
    ]
    tools, sessions = await load_mcp_tools(servers)
    try:
        agent = Agent(
            system="You are a helpful assistant.",
            tools=tools,
            transport=AnthropicTransport(),
        )
        reply = await agent.run("List the files here.", MemoryContextStore())
        print(reply)
    finally:
        for session in sessions:
            await session.close()


asyncio.run(main())
```

Sessions must be closed when you're done. Use `try/finally` as above, or an
`AsyncExitStack` for cleaner lifecycle management.

## Server configuration

`MCPServerConfig` supports two transport types: **stdio** (local subprocess)
and **HTTP**.

### Stdio (local subprocess)

```python
MCPServerConfig(
    name="filesystem",
    command="mcp-server-filesystem",
    args=["--root", "/home/user/project"],
    env={"MY_VAR": "value"},   # optional extra environment variables
)
```

| Field | Description |
|---|---|
| `name` | Server identifier - used as tool name prefix |
| `command` | Executable to run |
| `args` | Arguments passed to the command |
| `env` | Extra environment variables (merged with the current environment) |
| `timeout` | Connection timeout in seconds (default: `30.0`) |

### HTTP

```python
MCPServerConfig(
    name="remote",
    url="http://mcp-server.internal:3000/mcp",
    headers={"Authorization": "Bearer my-token"},
)
```

| Field | Description |
|---|---|
| `name` | Server identifier - used as tool name prefix |
| `url` | HTTP endpoint URL |
| `headers` | HTTP headers sent with every request |
| `timeout` | Connection timeout in seconds (default: `30.0`) |

## Tool naming

Tools are prefixed with the server name and a double underscore:

```
{server_name}__{tool_name}
```

For example, a `read_file` tool from a server named `fs` becomes `fs__read_file`.
This prevents name collisions when multiple servers expose tools with the same name.

## Multiple servers

Pass multiple configs to `load_mcp_tools` - tools from all servers are merged
into a single flat list:

```python
tools, sessions = await load_mcp_tools([
    MCPServerConfig(name="fs",  command="mcp-server-filesystem", args=["--root", "."]),
    MCPServerConfig(name="git", command="mcp-server-git"),
    MCPServerConfig(name="web", url="http://localhost:4000/mcp"),
])
```

## Error handling

If a server fails to connect, `load_mcp_tools` raises immediately. To handle
per-server failures gracefully, connect sessions individually:

```python
from axio_tools_mcp import MCPSession, MCPServerConfig

session = MCPSession(MCPServerConfig(name="fs", command="mcp-server-filesystem"))
try:
    await session.connect()
    tools = await session.list_tools()
except Exception as exc:
    print(f"Server unavailable: {exc}")
    tools = []
```
