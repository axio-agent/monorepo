# Plugin System

Axio uses Python's **entry point groups** for plugin discovery. Packages
register their components as entry points, and the framework discovers them
at startup - no import-time coupling, no centralized registry.

## Entry point groups

```{mermaid}
flowchart TD
    EP["Entry Points in pyproject.toml"]
    EP --> AT["axio.tools - Tool handler functions"]
    EP --> ATS["axio.tools.settings - ToolsPlugin (dynamic providers)"]
    EP --> ATR["axio.transport - CompletionTransport classes"]
    EP --> ATRS["axio.transport.settings - Transport settings screens"]
    EP --> AG["axio.guards - PermissionGuard classes"]
    EP --> AS["axio.selector - ToolSelector classes"]
```

| Group | Registers | Example |
|-------|-----------|---------|
| `axio.tools` | Individual tool handler functions | `shell = "axio_tools_local.shell:shell"` |
| `axio.tools.settings` | `ToolsPlugin` providers (dynamic tool sets) | `mcp = "axio_tools_mcp.plugin:MCPPlugin"` |
| `axio.transport` | `CompletionTransport` classes | `openai = "axio_transport_openai:OpenAITransport"` |
| `axio.transport.settings` | Transport settings UI screens | `openai = "axio_transport_openai:OpenAISettingsScreen"` |
| `axio.guards` | `PermissionGuard` subclasses | `path = "axio_tui_guards.guards:PathGuard"` |
| `axio.selector` | `ToolSelector` classes | `smart = "my_package.selector:SmartSelector"` |

## Registering entry points

In your package's `pyproject.toml`:

```toml
[project.entry-points."axio.tools"]
my_tool = "my_package.tools:my_tool"

[project.entry-points."axio.transport"]
my_transport = "my_package.transport:MyTransport"

[project.entry-points."axio.guards"]
my_guard = "my_package.guards:MyGuard"

[project.entry-points."axio.selector"]
my_selector = "my_package.selector:MySelector"
```

After installing the package (or running `uv sync` in the workspace), Axio
will automatically discover and load your components.

## Discovery functions

The `axio_tui.plugin` module provides discovery functions:

<!-- name: test_discover_tools -->
```python
from axio import Tool, CompletionTransport, PermissionGuard


def discover_tools() -> list[Tool]:
    """Load all tools from the axio.tools entry point group."""

def discover_tools_by_package() -> dict[str, list[Tool]]:
    """Return tools from axio.tools entry points grouped by distribution package name."""

def discover_tools_plugins() -> dict[str, "ToolsPlugin"]:
    """Load and instantiate tool plugins from axio.tools.settings."""

def discover_transports() -> dict[str, type]:
    """Load transport classes from axio.transport."""

def discover_transport_settings() -> dict[str, type]:
    """Load settings screen classes from axio.transport.settings."""

def discover_selectors() -> dict[str, type]:
    """Return selector classes from axio.selector entry points."""

def discover_guards() -> dict[str, type[PermissionGuard]]:
    """Load all guards from axio.guards."""
```

The functions that return dictionaries key the results by the entry point
name. `discover_tools()` returns a flat `list[Tool]` while
`discover_tools_by_package()` returns the same tools grouped by distribution
package name.

## ToolsPlugin protocol

For packages that provide a **dynamic set of tools** (like MCP or Docker
sandboxes), implement the `ToolsPlugin` protocol. Unlike static `axio.tools`
entries (one handler per entry point), a `ToolsPlugin` can return any number
of tools based on runtime configuration, and it integrates with the TUI's
settings screens.

The full protocol (defined in `axio_tui.plugin`) is:

```python
from typing import Any, Protocol, runtime_checkable
from axio import Tool


@runtime_checkable
class ToolsPlugin(Protocol):
    """Protocol for dynamic tool provider plugins.

    Plugins register via the ``axio.tools.settings`` entry point group.
    The TUI discovers them, calls ``init()``, collects tools, and shows
    settings screens - without knowing anything about the plugin internals.
    """

    @property
    def label(self) -> str:
        """Human-readable display name for the plugin (shown in the TUI)."""
        ...

    async def init(self, config: Any = None, global_config: Any = None) -> None:
        """Initialise the plugin, optionally with saved config."""
        ...

    @property
    def all_tools(self) -> list[Tool]:
        """Return the current list of tools this plugin provides."""
        ...

    def settings_screen(self) -> Any:
        """Return a Textual Screen (or compatible object) for configuring this plugin."""
        ...

    async def close(self) -> None:
        """Tear down connections or resources held by the plugin."""
        ...
```

The TUI lifecycle for a plugin is:

1. `discover_tools_plugins()` instantiates the class (no arguments).
2. `await plugin.init(config, global_config)` is called with any saved
   configuration.
3. `plugin.all_tools` is read to obtain the tools to register with the agent.
4. `plugin.settings_screen()` is called when the user opens the plugin's
   settings in the TUI.
5. `await plugin.close()` is called on shutdown.

Register a plugin under `axio.tools.settings`:

```toml
[project.entry-points."axio.tools.settings"]
my_plugin = "my_package.plugin:MyPlugin"
```

## Transport display name

Each transport class declares its display name via a `name: str` field:

<!-- name: test_transport_display_name -->
```python
import os
from dataclasses import dataclass, field
from axio import CompletionTransport


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
transport's own responsibility - typically via a `field(default_factory=...)`
that reads the appropriate environment variable.
