"""TUI screens for axio-transport-openai (requires textual)."""

from __future__ import annotations

try:
    from .custom import CustomHubScreen, ModelEditScreen, ProviderEditScreen
    from .nebius import NebiusSettingsScreen
    from .openai import OpenAISettingsScreen
    from .openrouter import OpenRouterSettingsScreen

    __all__ = [
        "CustomHubScreen",
        "ModelEditScreen",
        "NebiusSettingsScreen",
        "OpenAISettingsScreen",
        "OpenRouterSettingsScreen",
        "ProviderEditScreen",
    ]
except ImportError as _e:
    import warnings

    warnings.warn(
        f"axio-transport-openai TUI screens are unavailable: {_e}. "
        "Install textual: pip install axio[tui]",
        stacklevel=1,
    )
