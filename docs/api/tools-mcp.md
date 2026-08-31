# `axio-tools-mcp`

Explicit MCP server connections and conversion of discovered MCP tools into
Axio `Tool` definitions.

```{eval-rst}
.. autoclass:: axio_tools_mcp.MCPServerConfig
   :members:
```

```{eval-rst}
.. autoclass:: axio_tools_mcp.MCPSession
   :members:
```

```{eval-rst}
.. autofunction:: axio_tools_mcp.load_mcp_tools
```

```{eval-rst}
.. autoclass:: axio_tools_mcp.MCPRegistry
   :members:
```

## Plugin

`MCPPlugin` is what the `axio.tools.settings` entry point resolves to, so a host
application can offer MCP server configuration without importing this package.

```{eval-rst}
.. autoclass:: axio_tools_mcp.MCPPlugin
   :members:
```
