"""TUI screens for axio-tools-docker (requires textual)."""

from __future__ import annotations

try:
    from .docker import DockerSettingsScreen

    __all__ = ["DockerSettingsScreen"]
except ImportError as _e:
    import warnings

    warnings.warn(
        f"axio-tools-docker TUI screens are unavailable: {_e}. "
        "Install textual: pip install axio[tui]",
        stacklevel=1,
    )
