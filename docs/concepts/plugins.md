# Plugin System

Axio uses Python's **entry point groups** for plugin discovery. Packages
register their components as entry points, and the framework discovers them
at startup — no import-time coupling, no centralized registry.

## Entry point groups

```{mermaid}
flowchart TD
    EP["Entry Points in pyproject.toml"]
    EP --> AT["axio.tools — ToolHandler classes"]
    EP --> ATS["axio.tools.settings — ToolsPlugin (dynamic providers)"]
    EP --> ATR["axio.transport — CompletionTransport classes"]
    EP --> ATRS["axio.transport.settings — Transport settings screens"]
    EP --> AG["axio.guards — PermissionGuard classes"]
```

| Group | Registers | Example |
|-------|-----------|---------|
| `axio.tools` | Individual `ToolHandler` classes | `shell = "axio_tools_local.shell:Shell"` |
| `axio.tools.settings` | `ToolsPlugin` providers (dynamic tool sets) | `mcp = "axio_tools_mcp.plugin:MCPPlugin"` |
| `axio.transport` | `CompletionTransport` classes | `openai = "axio_transport_openai:OpenAITransport"` |
| `axio.transport.settings` | Transport settings UI screens | `openai = "axio_transport_openai:OpenAISettingsScreen"` |
| `axio.guards` | `PermissionGuard` subclasses | `path = "axio_tui_guards.guards:PathGuard"` |

## Registering entry points

In your package's `pyproject.toml`:

```toml
[project.entry-points."axio.tools"]
my_tool = "my_package.tools:MyToolHandler"

[project.entry-points."axio.transport"]
my_transport = "my_package.transport:MyTransport"

[project.entry-points."axio.guards"]
my_guard = "my_package.guards:MyGuard"
```

After installing the package (or running `uv sync` in the workspace), Axio
will automatically discover and load your components.

## Discovery functions

The `axio_tui.plugin` module provides discovery functions:

```python
def discover_tools() -> dict[str, Tool]:
    """Load all tools from the axio.tools entry point group."""

def discover_tools_plugins() -> dict[str, ToolsPlugin]:
    """Load dynamic tool providers from axio.tools.settings."""

def discover_transports() -> dict[str, type[CompletionTransport]]:
    """Load all transports from axio.transport."""

def discover_guards() -> dict[str, type[PermissionGuard]]:
    """Load all guards from axio.guards."""
```

Each function iterates over `importlib.metadata.entry_points()` for its
group, loads the objects, and returns them keyed by entry point name.

## ToolsPlugin protocol

For packages that provide a **dynamic set of tools** (like MCP or Docker
sandboxes), implement the `ToolsPlugin` protocol:

```python
@runtime_checkable
class ToolsPlugin(Protocol):
    async def get_tools(self) -> list[Tool]: ...
```

Unlike static `axio.tools` entries (one handler per entry point), a
`ToolsPlugin` can return any number of tools based on runtime configuration.

Register it under `axio.tools.settings`:

```toml
[project.entry-points."axio.tools.settings"]
my_plugin = "my_package.plugin:MyPlugin"
```

## Transport display name

Each transport class declares its display name via a `name: str` field:

```python
@dataclass(slots=True)
class MyTransport(CompletionTransport):
    name: str = "My Provider"
    api_key: str = field(default_factory=lambda: os.environ.get("MY_API_KEY", ""))
    ...
```

The TUI uses `transport.name` to label the transport in the welcome screen
and command palette.  A transport is considered *available* when its
`fetch_models()` call succeeds; if it raises (e.g. because no API key is
present), the transport is shown as unavailable.  API key lookup is each
transport's own responsibility — typically via a `field(default_factory=...)`
that reads the appropriate environment variable.
