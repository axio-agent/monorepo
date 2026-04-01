"""TUI screens for axio-transport-codex (requires textual)."""

from __future__ import annotations

try:
    from .settings import CodexSettingsScreen

    __all__ = ["CodexSettingsScreen"]
except ImportError as _e:
    import warnings

    warnings.warn(
        f"axio-transport-codex TUI screens are unavailable: {_e}. "
        "Install textual: pip install axio[tui]",
        stacklevel=1,
    )
