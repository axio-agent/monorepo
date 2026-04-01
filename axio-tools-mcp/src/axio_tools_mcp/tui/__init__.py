"""TUI screens for axio-tools-mcp (requires textual)."""

from __future__ import annotations

try:
    from .screens import MCPHubScreen, MCPServerEditScreen

    __all__ = ["MCPHubScreen", "MCPServerEditScreen"]
except ImportError as _e:
    import warnings

    warnings.warn(
        f"axio-tools-mcp TUI screens are unavailable: {_e}. "
        "Install textual: pip install axio[tui]",
        stacklevel=1,
    )
